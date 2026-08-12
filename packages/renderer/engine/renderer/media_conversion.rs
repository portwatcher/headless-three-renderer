use anyhow::{Result, bail};

use super::{MediaFrameResources, MediaOutputFormat};

pub(super) struct MediaConverter {
    pipeline: Option<wgpu::ComputePipeline>,
}

impl MediaConverter {
    pub(super) fn new(device: &wgpu::Device, format: MediaOutputFormat) -> Result<Self> {
        let (source, y_format, uv_format) = match format {
            MediaOutputFormat::Rgba8 => return Ok(Self { pipeline: None }),
            MediaOutputFormat::Nv12 => (
                include_str!("../shader/media_nv12.wgsl"),
                wgpu::TextureFormat::R8Unorm,
                wgpu::TextureFormat::Rg8Unorm,
            ),
            MediaOutputFormat::P010 => {
                if !device
                    .features()
                    .contains(wgpu::Features::TEXTURE_FORMAT_16BIT_NORM)
                {
                    bail!("P010 requires wgpu TEXTURE_FORMAT_16BIT_NORM support")
                }
                (
                    include_str!("../shader/media_p010.wgsl"),
                    wgpu::TextureFormat::R16Unorm,
                    wgpu::TextureFormat::Rg16Unorm,
                )
            }
        };
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("headless-three media conversion shader"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("headless-three media conversion layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                storage_entry(1, y_format),
                storage_entry(2, uv_format),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("headless-three media conversion pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("headless-three media conversion pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("convert"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        Ok(Self {
            pipeline: Some(pipeline),
        })
    }

    pub(super) fn convert(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        resources: &MediaFrameResources,
    ) -> Result<()> {
        let Some(pipeline) = &self.pipeline else {
            return Ok(());
        };
        let y = resources.y.as_ref().expect("media Y plane");
        let uv = resources.uv.as_ref().expect("media UV plane");
        let rgba_view = resources.rgba.create_view(&Default::default());
        let y_view = y.create_view(&Default::default());
        let uv_view = uv.create_view(&Default::default());
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("headless-three media conversion bind group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&rgba_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&y_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&uv_view),
                },
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("headless-three media conversion encoder"),
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
        queue.submit([encoder.finish()]);
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|error| {
                anyhow::anyhow!("failed while waiting for media conversion: {error}")
            })?;
        Ok(())
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
