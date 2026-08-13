use anyhow::{Result, bail};

use super::MediaOutputFormat;
use super::media_resources::MediaFrameResources;
use super::media_resources::i420_layout;

pub(super) struct MediaConverter {
    format: MediaOutputFormat,
    conversion_pipeline: Option<wgpu::ComputePipeline>,
    i420_pack_pipeline: Option<wgpu::ComputePipeline>,
}

impl MediaConverter {
    pub(super) fn new(device: &wgpu::Device, format: MediaOutputFormat) -> Result<Self> {
        let conversion_pipeline = match format {
            MediaOutputFormat::Rgba8 => None,
            MediaOutputFormat::Nv12 => Some(create_conversion_pipeline(
                device,
                include_str!("../shader/media_nv12.wgsl"),
                &[wgpu::TextureFormat::R8Unorm, wgpu::TextureFormat::Rg8Unorm],
            )),
            MediaOutputFormat::P010 => {
                if !device
                    .features()
                    .contains(wgpu::Features::TEXTURE_FORMAT_16BIT_NORM)
                {
                    bail!("P010 requires wgpu TEXTURE_FORMAT_16BIT_NORM support")
                }
                Some(create_conversion_pipeline(
                    device,
                    include_str!("../shader/media_p010.wgsl"),
                    &[
                        wgpu::TextureFormat::R16Unorm,
                        wgpu::TextureFormat::Rg16Unorm,
                    ],
                ))
            }
            MediaOutputFormat::I420 => Some(create_conversion_pipeline(
                device,
                include_str!("../shader/media_i420.wgsl"),
                &[
                    wgpu::TextureFormat::R8Unorm,
                    wgpu::TextureFormat::R8Unorm,
                    wgpu::TextureFormat::R8Unorm,
                ],
            )),
        };
        let i420_pack_pipeline =
            (format == MediaOutputFormat::I420).then(|| create_i420_pack_pipeline(device));
        Ok(Self {
            format,
            conversion_pipeline,
            i420_pack_pipeline,
        })
    }

    pub(super) fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        resources: &MediaFrameResources,
        copy_packed_i420_to_readback: bool,
    ) -> Result<()> {
        let Some(pipeline) = &self.conversion_pipeline else {
            if copy_packed_i420_to_readback {
                bail!("packed I420 readback requires an i420-planes converter")
            }
            return Ok(());
        };
        let rgba_view = resources.rgba.create_view(&Default::default());
        let plane_views = conversion_plane_views(resources, self.format);
        let mut entries = Vec::with_capacity(plane_views.len() + 1);
        entries.push(wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(&rgba_view),
        });
        entries.extend(
            plane_views
                .iter()
                .enumerate()
                .map(|(index, view)| wgpu::BindGroupEntry {
                    binding: index as u32 + 1,
                    resource: wgpu::BindingResource::TextureView(view),
                }),
        );
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("headless-three media conversion bind group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("headless-three media conversion pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(
                (resources.width / 2).div_ceil(8),
                (resources.height / 2).div_ceil(8),
                1,
            );
        }
        if let Some(pack_pipeline) = &self.i420_pack_pipeline {
            encode_i420_pack(device, encoder, pack_pipeline, resources);
            if copy_packed_i420_to_readback {
                let layout = i420_layout(resources.width, resources.height);
                encoder.copy_buffer_to_buffer(
                    resources.packed_i420.as_ref().expect("packed I420 buffer"),
                    0,
                    resources
                        .i420_readback
                        .as_ref()
                        .expect("I420 readback buffer"),
                    0,
                    layout.gpu_readback_bytes as u64,
                );
            }
        } else if copy_packed_i420_to_readback {
            bail!("packed I420 readback requires an i420-planes converter")
        }
        Ok(())
    }
}

fn create_conversion_pipeline(
    device: &wgpu::Device,
    source: &'static str,
    plane_formats: &[wgpu::TextureFormat],
) -> wgpu::ComputePipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("headless-three media conversion shader"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let mut entries = Vec::with_capacity(plane_formats.len() + 1);
    entries.push(sampled_texture_entry(0));
    entries.extend(
        plane_formats
            .iter()
            .enumerate()
            .map(|(index, format)| storage_entry(index as u32 + 1, *format)),
    );
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("headless-three media conversion layout"),
        entries: &entries,
    });
    compute_pipeline(
        device,
        &module,
        &layout,
        "headless-three media conversion pipeline",
    )
}

fn create_i420_pack_pipeline(device: &wgpu::Device) -> wgpu::ComputePipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("headless-three packed I420 shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../shader/media_i420_pack.wgsl").into()),
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("headless-three packed I420 layout"),
        entries: &[
            sampled_texture_entry(0),
            sampled_texture_entry(1),
            sampled_texture_entry(2),
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    compute_pipeline(
        device,
        &module,
        &layout,
        "headless-three packed I420 pipeline",
    )
}

fn compute_pipeline(
    device: &wgpu::Device,
    module: &wgpu::ShaderModule,
    layout: &wgpu::BindGroupLayout,
    label: &'static str,
) -> wgpu::ComputePipeline {
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(layout)],
        immediate_size: 0,
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&pipeline_layout),
        module,
        entry_point: Some("convert"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    })
}

fn conversion_plane_views(
    resources: &MediaFrameResources,
    format: MediaOutputFormat,
) -> Vec<wgpu::TextureView> {
    let mut views = vec![
        resources
            .y
            .as_ref()
            .expect("media Y plane")
            .create_view(&Default::default()),
    ];
    if format == MediaOutputFormat::I420 {
        views.push(
            resources
                .u
                .as_ref()
                .expect("media U plane")
                .create_view(&Default::default()),
        );
        views.push(
            resources
                .v
                .as_ref()
                .expect("media V plane")
                .create_view(&Default::default()),
        );
    } else {
        views.push(
            resources
                .uv
                .as_ref()
                .expect("media UV plane")
                .create_view(&Default::default()),
        );
    }
    views
}

fn encode_i420_pack(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::ComputePipeline,
    resources: &MediaFrameResources,
) {
    let views = [
        resources
            .y
            .as_ref()
            .expect("I420 Y plane")
            .create_view(&Default::default()),
        resources
            .u
            .as_ref()
            .expect("I420 U plane")
            .create_view(&Default::default()),
        resources
            .v
            .as_ref()
            .expect("I420 V plane")
            .create_view(&Default::default()),
    ];
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("headless-three packed I420 bind group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&views[0]),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&views[1]),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&views[2]),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::Buffer(
                    resources
                        .packed_i420
                        .as_ref()
                        .expect("packed I420 buffer")
                        .as_entire_buffer_binding(),
                ),
            },
        ],
    });
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("headless-three packed I420 pass"),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    let word_count = i420_layout(resources.width, resources.height)
        .gpu_readback_bytes
        .div_ceil(4) as u32;
    let group_count = word_count.div_ceil(64);
    let groups_x = group_count.min(65_535);
    pass.dispatch_workgroups(groups_x, group_count.div_ceil(groups_x), 1);
}

fn sampled_texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn storage_entry(binding: u32, format: wgpu::TextureFormat) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format,
            view_dimension: wgpu::TextureViewDimension::D2,
        },
        count: None,
    }
}
