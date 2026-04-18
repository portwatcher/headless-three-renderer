use anyhow::{Context, Result, anyhow, bail};
use bytemuck::{Pod, Zeroable};
use glam::Mat4;
use wgpu::util::DeviceExt;

use crate::ibl::IblMaps;
use crate::lights::{GpuLight, MAX_LIGHTS};
use crate::mesh::{MeshSide, PreparedMesh, Topology, Vertex, WrapMode, prepare_meshes};
use crate::settings::{OutputFormat, RenderSettings};
use crate::shader::SHADER;
use crate::types::{Camera, RenderScene};
use crate::util::{align_to, encode_png};
use crate::{COLOR_FORMAT, DEPTH_FORMAT};

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Uniforms {
    pub mvp: [[f32; 4]; 4],
    pub model: [[f32; 4]; 4],
    pub normal_matrix: [[f32; 4]; 4],
    pub camera_pos: [f32; 4],
    pub base_color: [f32; 4],
    pub emissive: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
    pub ambient_intensity: f32,
    pub num_lights: u32,
    pub ambient_color: [f32; 4],
    // x = normal_scale.x, y = normal_scale.y, z = has_normal_map (1.0 or 0.0), w = has_ibl (1.0 or 0.0)
    pub normal_map_params: [f32; 4],
    /// x = env_intensity, y/z/w = reserved
    pub ibl_params: [f32; 4],
    /// x = ao_map_intensity, y = has_ao_map (1.0 or 0.0), z/w = reserved
    pub ao_params: [f32; 4],
    pub light_space_matrix: [[f32; 4]; 4],
    /// x = has_shadow, y = bias, z = normal_bias, w = receive_shadow
    pub shadow_params: [f32; 4],
    /// x = shadow light index (as f32), y = 1/map_size, z/w = reserved
    pub shadow_params2: [f32; 4],
    pub lights: [GpuLight; MAX_LIGHTS],
}

pub struct GpuRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    /// Opaque pipelines keyed by `MeshSide` (Front, Back, Double).
    pipelines: [wgpu::RenderPipeline; 3],
    /// Transparent pipelines (no depth write) keyed by `MeshSide`.
    transparent_pipelines: [wgpu::RenderPipeline; 3],
    /// Line / point pipelines: [opaque, transparent] for each.
    line_pipelines: [wgpu::RenderPipeline; 2],
    point_pipelines: [wgpu::RenderPipeline; 2],
    uniform_layout: wgpu::BindGroupLayout,
    texture_layout: wgpu::BindGroupLayout,
    normal_map_layout: wgpu::BindGroupLayout,
    mr_map_layout: wgpu::BindGroupLayout,
    emissive_map_layout: wgpu::BindGroupLayout,
    ibl_layout: wgpu::BindGroupLayout,
    ao_map_layout: wgpu::BindGroupLayout,
    shadow_layout: wgpu::BindGroupLayout,
    /// Depth-only pipeline used to render the shadow map.
    shadow_pipeline: wgpu::RenderPipeline,
    sampler: wgpu::Sampler,
    shadow_sampler: wgpu::Sampler,
    _default_texture: wgpu::Texture,
    default_texture_bind_group: wgpu::BindGroup,
    default_normal_map_bind_group: wgpu::BindGroup,
    default_mr_map_bind_group: wgpu::BindGroup,
    default_emissive_map_bind_group: wgpu::BindGroup,
    default_ibl_bind_group: wgpu::BindGroup,
    default_ao_map_bind_group: wgpu::BindGroup,
    default_shadow_bind_group: wgpu::BindGroup,
    _default_shadow_texture: wgpu::Texture,
}

struct GpuMesh {
    vertex_buffer: wgpu::Buffer,
    index_buffer: Option<wgpu::Buffer>,
    bind_group: wgpu::BindGroup,
    texture_bind_group: wgpu::BindGroup,
    normal_map_bind_group: wgpu::BindGroup,
    mr_map_bind_group: wgpu::BindGroup,
    emissive_map_bind_group: wgpu::BindGroup,
    ao_map_bind_group: wgpu::BindGroup,
    index_count: u32,
    vertex_count: u32,
    side: MeshSide,
    topology: Topology,
    cast_shadow: bool,
    _uniform_buffer: wgpu::Buffer,
    _texture: Option<wgpu::Texture>,
    _normal_map: Option<wgpu::Texture>,
    _mr_map: Option<wgpu::Texture>,
    _emissive_map: Option<wgpu::Texture>,
    _ao_map: Option<wgpu::Texture>,
}

impl GpuRenderer {
    pub fn new() -> Result<Self> {
        pollster::block_on(Self::new_async())
    }

    async fn new_async() -> Result<Self> {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());

        let adapter = match instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
        {
            Ok(adapter) => adapter,
            Err(first_error) => instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::None,
                    force_fallback_adapter: true,
                    compatible_surface: None,
                })
                .await
                .map_err(|fallback_error| {
                    anyhow!(
                        "failed to find a wgpu adapter: {first_error}; fallback adapter also failed: {fallback_error}"
                    )
                })?,
        };

        let required_limits = wgpu::Limits::downlevel_defaults().using_resolution(adapter.limits());
        // Raise the bind group count: we use 8 (uniforms + 5 texture slots +
        // IBL + shadow map), which is the WebGPU spec default.
        let required_limits = wgpu::Limits {
            max_bind_groups: 8,
            ..required_limits
        };
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("headless-three-renderer device"),
                required_features: wgpu::Features::empty(),
                required_limits,
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .context("failed to create wgpu device")?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("headless-three-renderer shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });

        let uniform_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("headless-three-renderer uniform layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<Uniforms>() as u64),
                },
                count: None,
            }],
        });

        let texture_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("headless-three-renderer texture layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let normal_map_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("headless-three-renderer normal map layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let mr_map_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("headless-three-renderer metallic-roughness map layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let emissive_map_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("headless-three-renderer emissive map layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let ao_map_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("headless-three-renderer ao map layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        // Shadow map bind group: depth texture + comparison sampler.
        let shadow_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("headless-three-renderer shadow layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Depth,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
            ],
        });

        // IBL bind group: irradiance cubemap, prefiltered cubemap, BRDF LUT, sampler
        let ibl_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("headless-three-renderer ibl layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::Cube,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::Cube,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("headless-three-renderer sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        let shadow_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("headless-three-renderer shadow sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });

        let default_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("headless-three-renderer default texture"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: COLOR_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &default_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &[255u8, 255, 255, 255],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        let default_texture_view =
            default_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let default_texture_bind_group =
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("headless-three-renderer default texture bind group"),
                layout: &texture_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&default_texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                ],
            });

        // Default normal map: flat normal (0, 0, 1) encoded as (128, 128, 255)
        let default_normal_map = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("headless-three-renderer default normal map"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: COLOR_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &default_normal_map,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &[128u8, 128, 255, 255],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        let default_normal_map_view =
            default_normal_map.create_view(&wgpu::TextureViewDescriptor::default());
        let default_normal_map_bind_group =
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("headless-three-renderer default normal map bind group"),
                layout: &normal_map_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&default_normal_map_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                ],
            });

        // Default metallic-roughness map: white (1,1,1,1) so that
        // metallic = uniform.metallic * 1.0 and roughness = uniform.roughness * 1.0
        let default_mr_map_view =
            default_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let default_mr_map_bind_group =
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("headless-three-renderer default metallic-roughness bind group"),
                layout: &mr_map_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&default_mr_map_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                ],
            });

        // Default emissive map: black (0,0,0,255) so that emissive contribution is zero
        // when no emissive map is provided
        let default_emissive_map = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("headless-three-renderer default emissive map"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: COLOR_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &default_emissive_map,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &[255u8, 255, 255, 255],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        let default_emissive_map_view =
            default_emissive_map.create_view(&wgpu::TextureViewDescriptor::default());
        let default_emissive_map_bind_group =
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("headless-three-renderer default emissive map bind group"),
                layout: &emissive_map_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&default_emissive_map_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                ],
            });

        // Default IBL: 1x1 black cubemaps (no env map contribution)
        let default_ibl_bind_group = create_default_ibl_bind_group(
            &device,
            &queue,
            &ibl_layout,
            &sampler,
        );

        // Default AO map: reuse the 1x1 white default texture; red channel = 1.0
        // means full illumination (no occlusion).
        let default_ao_map_view =
            default_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let default_ao_map_bind_group =
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("headless-three-renderer default ao map bind group"),
                layout: &ao_map_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&default_ao_map_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                ],
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("headless-three-renderer pipeline layout"),
            bind_group_layouts: &[Some(&uniform_layout), Some(&texture_layout), Some(&normal_map_layout), Some(&mr_map_layout), Some(&emissive_map_layout), Some(&ibl_layout), Some(&ao_map_layout), Some(&shadow_layout)],
            immediate_size: 0,
        });

        // 1x1 depth texture used as a "no shadow" default binding.
        let default_shadow_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("headless-three-renderer default shadow texture"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let default_shadow_view = default_shadow_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let default_shadow_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("headless-three-renderer default shadow bind group"),
            layout: &shadow_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&default_shadow_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&shadow_sampler) },
            ],
        });

        // Dedicated pipeline layout for the depth-only shadow pass (uniforms only).
        let shadow_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("headless-three-renderer shadow pipeline layout"),
            bind_group_layouts: &[Some(&uniform_layout)],
            immediate_size: 0,
        });
        let shadow_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("headless-three-renderer shadow pipeline"),
            layout: Some(&shadow_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_shadow"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[Vertex::layout()],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                // No culling: captures shadows from any side, including DoubleSide materials.
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                // Slight slope-scaled bias to reduce acne (in addition to the
                // per-fragment bias we apply during shadow sampling).
                bias: wgpu::DepthBiasState {
                    constant: 2,
                    slope_scale: 2.0,
                    clamp: 0.0,
                },
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: None,
            multiview_mask: None,
            cache: None,
        });

        let color_targets = [Some(wgpu::ColorTargetState {
            format: COLOR_FORMAT,
            blend: Some(wgpu::BlendState::ALPHA_BLENDING),
            write_mask: wgpu::ColorWrites::ALL,
        })];

        let vertex_buffers = [Vertex::layout()];
        let make_pipeline = |topology: Topology, side: MeshSide, transparent: bool| {
            let label = match (topology, side, transparent) {
                (Topology::Triangles, MeshSide::Front, false) => "pipeline (tri front)",
                (Topology::Triangles, MeshSide::Back, false) => "pipeline (tri back)",
                (Topology::Triangles, MeshSide::Double, false) => "pipeline (tri double)",
                (Topology::Triangles, MeshSide::Front, true) => "pipeline (tri front, transparent)",
                (Topology::Triangles, MeshSide::Back, true) => "pipeline (tri back, transparent)",
                (Topology::Triangles, MeshSide::Double, true) => "pipeline (tri double, transparent)",
                (Topology::Lines, _, false) => "pipeline (lines)",
                (Topology::Lines, _, true) => "pipeline (lines, transparent)",
                (Topology::Points, _, false) => "pipeline (points)",
                (Topology::Points, _, true) => "pipeline (points, transparent)",
            };
            let depth_write = !transparent;
            // Lines and points have no faces to cull.
            let cull_mode = match topology {
                Topology::Triangles => side.cull_mode(),
                Topology::Lines | Topology::Points => None,
            };
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &vertex_buffers,
                },
                primitive: wgpu::PrimitiveState {
                    topology: topology.primitive(),
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode,
                    unclipped_depth: false,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    conservative: false,
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: Some(depth_write),
                    depth_compare: Some(wgpu::CompareFunction::Less),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &color_targets,
                }),
                multiview_mask: None,
                cache: None,
            })
        };

        let pipelines = [
            make_pipeline(Topology::Triangles, MeshSide::Front, false),
            make_pipeline(Topology::Triangles, MeshSide::Back, false),
            make_pipeline(Topology::Triangles, MeshSide::Double, false),
        ];
        let transparent_pipelines = [
            make_pipeline(Topology::Triangles, MeshSide::Front, true),
            make_pipeline(Topology::Triangles, MeshSide::Back, true),
            make_pipeline(Topology::Triangles, MeshSide::Double, true),
        ];
        let line_pipelines = [
            make_pipeline(Topology::Lines, MeshSide::Front, false),
            make_pipeline(Topology::Lines, MeshSide::Front, true),
        ];
        let point_pipelines = [
            make_pipeline(Topology::Points, MeshSide::Front, false),
            make_pipeline(Topology::Points, MeshSide::Front, true),
        ];

        Ok(Self {
            device,
            queue,
            pipelines,
            transparent_pipelines,
            line_pipelines,
            point_pipelines,
            uniform_layout,
            texture_layout,
            normal_map_layout,
            mr_map_layout,
            emissive_map_layout,
            ibl_layout,
            ao_map_layout,
            shadow_layout,
            shadow_pipeline,
            sampler,
            shadow_sampler,
            _default_texture: default_texture,
            default_texture_bind_group,
            default_normal_map_bind_group,
            default_mr_map_bind_group,
            default_emissive_map_bind_group,
            default_ibl_bind_group,
            default_ao_map_bind_group,
            default_shadow_bind_group,
            _default_shadow_texture: default_shadow_texture,
        })
    }

    pub fn render(&self, scene: &RenderScene, camera: &Camera) -> Result<Vec<u8>> {
        let settings = RenderSettings::from_scene(scene, camera, self.device.limits())?;
        let meshes = prepare_meshes(scene)?;
        let rgba = self.render_rgba(&settings, &meshes)?;

        match settings.output_format {
            OutputFormat::Png => encode_png(&rgba, settings.width, settings.height),
            OutputFormat::Rgba => Ok(rgba),
        }
    }

    fn render_rgba(&self, settings: &RenderSettings, meshes: &[PreparedMesh]) -> Result<Vec<u8>> {
        let texture_size = wgpu::Extent3d {
            width: settings.width,
            height: settings.height,
            depth_or_array_layers: 1,
        };

        let color_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("headless-three-renderer color texture"),
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: COLOR_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let color_view = color_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let depth_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("headless-three-renderer depth texture"),
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let gpu_meshes = meshes
            .iter()
            .map(|mesh| self.upload_mesh(settings, mesh))
            .collect::<Result<Vec<_>>>()?;

        let ibl_bind_group = match &settings.ibl {
            Some(ibl) => create_ibl_bind_group(&self.device, &self.queue, &self.ibl_layout, &self.sampler, ibl),
            None => self.default_ibl_bind_group.clone(),
        };

        // Optional shadow map pass: render a depth-only view from the
        // directional light's POV into a Depth32Float texture, then bind it
        // to the main pass.
        let (shadow_bind_group, _shadow_texture) = match &settings.shadow {
            Some(_shadow) => {
                let (bg, tex) = self.render_shadow_pass(settings, &gpu_meshes);
                (bg, Some(tex))
            }
            None => (self.default_shadow_bind_group.clone(), None),
        };

        let (opaque_order, transparent_order) = partition_draw_order(meshes, settings.camera_pos);

        let unpadded_bytes_per_row = settings.width * 4;
        let padded_bytes_per_row =
            align_to(unpadded_bytes_per_row, wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let output_buffer_size = padded_bytes_per_row as u64 * settings.height as u64;
        if output_buffer_size > self.device.limits().max_buffer_size {
            bail!(
                "render output requires {} bytes, above device max_buffer_size {}",
                output_buffer_size,
                self.device.limits().max_buffer_size
            );
        }

        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("headless-three-renderer readback buffer"),
            size: output_buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("headless-three-renderer render encoder"),
            });

        {
            let color_attachments = [Some(wgpu::RenderPassColorAttachment {
                view: &color_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: settings.background[0],
                        g: settings.background[1],
                        b: settings.background[2],
                        a: settings.background[3],
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })];

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("headless-three-renderer render pass"),
                color_attachments: &color_attachments,
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            // Opaque meshes first (with depth write)
            pass.set_bind_group(5, &ibl_bind_group, &[]);
            pass.set_bind_group(7, &shadow_bind_group, &[]);
            let mut current_pipeline: Option<PipelineKey> = None;
            for &i in &opaque_order {
                let key = pipeline_key(&gpu_meshes[i]);
                if current_pipeline != Some(key) {
                    pass.set_pipeline(self.pipeline_for(key, false));
                    current_pipeline = Some(key);
                }
                draw_gpu_mesh(&mut pass, &gpu_meshes[i]);
            }

            // Transparent meshes second (back-to-front, no depth write)
            if !transparent_order.is_empty() {
                let mut current_pipeline: Option<PipelineKey> = None;
                for &i in &transparent_order {
                    let key = pipeline_key(&gpu_meshes[i]);
                    if current_pipeline != Some(key) {
                        pass.set_pipeline(self.pipeline_for(key, true));
                        current_pipeline = Some(key);
                    }
                    draw_gpu_mesh(&mut pass, &gpu_meshes[i]);
                }
            }
        }

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &color_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &output_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(settings.height),
                },
            },
            texture_size,
        );

        self.queue.submit([encoder.finish()]);

        let buffer_slice = output_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .context("failed while waiting for GPU readback")?;

        receiver
            .recv()
            .context("GPU readback callback was not delivered")?
            .context("failed to map GPU readback buffer")?;

        let padded_data = buffer_slice.get_mapped_range();
        let mut rgba = vec![0; (settings.width * settings.height * 4) as usize];
        let unpadded = unpadded_bytes_per_row as usize;
        let padded = padded_bytes_per_row as usize;

        for row in 0..settings.height as usize {
            let src_start = row * padded;
            let dst_start = row * unpadded;
            rgba[dst_start..dst_start + unpadded]
                .copy_from_slice(&padded_data[src_start..src_start + unpadded]);
        }

        drop(padded_data);
        output_buffer.unmap();

        Ok(rgba)
    }

    /// Render the scene's shadow casters into a depth-only texture from the
    /// directional light's POV and return a bind group referencing it.
    fn render_shadow_pass(
        &self,
        settings: &RenderSettings,
        gpu_meshes: &[GpuMesh],
    ) -> (wgpu::BindGroup, wgpu::Texture) {
        let shadow = settings
            .shadow
            .as_ref()
            .expect("render_shadow_pass requires a configured shadow caster");
        let size = shadow.map_size;

        let shadow_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("headless-three-renderer shadow map"),
            size: wgpu::Extent3d { width: size, height: size, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let shadow_view = shadow_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("headless-three-renderer shadow encoder"),
            });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("headless-three-renderer shadow pass"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &shadow_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.shadow_pipeline);
            for mesh in gpu_meshes.iter() {
                // Only triangle meshes flagged as shadow casters contribute.
                if !mesh.cast_shadow || mesh.topology != Topology::Triangles {
                    continue;
                }
                pass.set_bind_group(0, &mesh.bind_group, &[]);
                pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                if let Some(index_buffer) = &mesh.index_buffer {
                    pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..mesh.index_count, 0, 0..1);
                } else {
                    pass.draw(0..mesh.vertex_count, 0..1);
                }
            }
        }

        self.queue.submit([encoder.finish()]);

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("headless-three-renderer shadow bind group"),
            layout: &self.shadow_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&shadow_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.shadow_sampler) },
            ],
        });
        let _ = &shadow.light_dir; // silence unused warning; direction baked into light_vp
        (bind_group, shadow_texture)
    }

    fn sampler_for_wrap(&self, wrap_s: WrapMode, wrap_t: WrapMode) -> wgpu::Sampler {
        if wrap_s == WrapMode::ClampToEdge && wrap_t == WrapMode::ClampToEdge {
            return self.sampler.clone();
        }
        self.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("headless-three-renderer per-mesh sampler"),
            address_mode_u: wrap_s.to_address_mode(),
            address_mode_v: wrap_t.to_address_mode(),
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        })
    }

    fn upload_mesh(&self, settings: &RenderSettings, mesh: &PreparedMesh) -> Result<GpuMesh> {
        let vertex_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("headless-three-renderer vertex buffer"),
                contents: bytemuck::cast_slice(&mesh.vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });

        let index_buffer = mesh.indices.as_ref().map(|indices| {
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("headless-three-renderer index buffer"),
                    contents: bytemuck::cast_slice(indices),
                    usage: wgpu::BufferUsages::INDEX,
                })
        });

        let model = mesh.transform;
        let mvp = settings.view_projection * model;
        let normal_matrix = model.inverse().transpose();

        let mut lights = [GpuLight::zeroed(); MAX_LIGHTS];
        for (i, light) in settings.lights.iter().take(MAX_LIGHTS).enumerate() {
            lights[i] = *light;
        }

        let uniforms = Uniforms {
            mvp: mvp.to_cols_array_2d(),
            model: model.to_cols_array_2d(),
            normal_matrix: normal_matrix.to_cols_array_2d(),
            camera_pos: [settings.camera_pos.x, settings.camera_pos.y, settings.camera_pos.z, 0.0],
            base_color: mesh.base_color,
            emissive: [mesh.emissive[0], mesh.emissive[1], mesh.emissive[2], mesh.alpha_test],
            metallic: mesh.metallic,
            roughness: mesh.roughness,
            ambient_intensity: settings.ambient_intensity,
            num_lights: settings.lights.len().min(MAX_LIGHTS) as u32,
            ambient_color: [settings.ambient_color[0], settings.ambient_color[1], settings.ambient_color[2], 0.0],
            normal_map_params: [
                mesh.normal_scale[0],
                mesh.normal_scale[1],
                if mesh.normal_map.is_some() { 1.0 } else { 0.0 },
                if settings.ibl.is_some() { 1.0 } else { 0.0 },
            ],
            ibl_params: [
                settings.env_intensity,
                mesh.shading_model.as_u32() as f32,
                0.0,
                0.0,
            ],
            ao_params: [
                mesh.ao_map_intensity,
                if mesh.ao_map.is_some() { 1.0 } else { 0.0 },
                0.0,
                0.0,
            ],
            light_space_matrix: settings
                .shadow
                .as_ref()
                .map(|s| s.light_vp.to_cols_array_2d())
                .unwrap_or_else(|| Mat4::IDENTITY.to_cols_array_2d()),
            shadow_params: match &settings.shadow {
                Some(s) => [
                    1.0,
                    s.bias,
                    s.normal_bias,
                    if mesh.receive_shadow { 1.0 } else { 0.0 },
                ],
                None => [0.0, 0.0, 0.0, 0.0],
            },
            shadow_params2: match &settings.shadow {
                Some(s) => [s.light_index as f32, 1.0 / s.map_size as f32, 0.0, 0.0],
                None => [0.0, 0.0, 0.0, 0.0],
            },
            lights,
        };
        let uniform_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("headless-three-renderer uniform buffer"),
                contents: bytemuck::bytes_of(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("headless-three-renderer bind group"),
            layout: &self.uniform_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let (texture_bind_group, _mesh_texture) = match &mesh.texture {
            Some(tex) => {
                let tex_size = wgpu::Extent3d {
                    width: tex.width,
                    height: tex.height,
                    depth_or_array_layers: 1,
                };
                let gpu_texture = self.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("headless-three-renderer mesh texture"),
                    size: tex_size,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: COLOR_FORMAT,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                });
                self.queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &gpu_texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    &tex.rgba,
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(4 * tex.width),
                        rows_per_image: Some(tex.height),
                    },
                    tex_size,
                );
                let tex_view =
                    gpu_texture.create_view(&wgpu::TextureViewDescriptor::default());
                let sampler_for_tex = self.sampler_for_wrap(tex.wrap_s, tex.wrap_t);
                let tex_bind_group =
                    self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("headless-three-renderer mesh texture bind group"),
                        layout: &self.texture_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(&tex_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::Sampler(&sampler_for_tex),
                            },
                        ],
                    });
                (tex_bind_group, Some(gpu_texture))
            }
            None => (self.default_texture_bind_group.clone(), None),
        };

        let (normal_map_bind_group, _normal_map_texture) = match &mesh.normal_map {
            Some(tex) => {
                let tex_size = wgpu::Extent3d {
                    width: tex.width,
                    height: tex.height,
                    depth_or_array_layers: 1,
                };
                let gpu_texture = self.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("headless-three-renderer normal map"),
                    size: tex_size,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: COLOR_FORMAT,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                });
                self.queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &gpu_texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    &tex.rgba,
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(4 * tex.width),
                        rows_per_image: Some(tex.height),
                    },
                    tex_size,
                );
                let tex_view =
                    gpu_texture.create_view(&wgpu::TextureViewDescriptor::default());
                let tex_bind_group =
                    self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("headless-three-renderer normal map bind group"),
                        layout: &self.normal_map_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(&tex_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::Sampler(&self.sampler),
                            },
                        ],
                    });
                (tex_bind_group, Some(gpu_texture))
            }
            None => (self.default_normal_map_bind_group.clone(), None),
        };

        let (mr_map_bind_group, _mr_map_texture) = match &mesh.metallic_roughness_texture {
            Some(tex) => {
                let tex_size = wgpu::Extent3d {
                    width: tex.width,
                    height: tex.height,
                    depth_or_array_layers: 1,
                };
                let gpu_texture = self.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("headless-three-renderer metallic-roughness map"),
                    size: tex_size,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: COLOR_FORMAT,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                });
                self.queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &gpu_texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    &tex.rgba,
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(4 * tex.width),
                        rows_per_image: Some(tex.height),
                    },
                    tex_size,
                );
                let tex_view =
                    gpu_texture.create_view(&wgpu::TextureViewDescriptor::default());
                let tex_bind_group =
                    self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("headless-three-renderer metallic-roughness bind group"),
                        layout: &self.mr_map_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(&tex_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::Sampler(&self.sampler),
                            },
                        ],
                    });
                (tex_bind_group, Some(gpu_texture))
            }
            None => (self.default_mr_map_bind_group.clone(), None),
        };

        let (emissive_map_bind_group, _emissive_map_texture) = match &mesh.emissive_map {
            Some(tex) => {
                let tex_size = wgpu::Extent3d {
                    width: tex.width,
                    height: tex.height,
                    depth_or_array_layers: 1,
                };
                let gpu_texture = self.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("headless-three-renderer emissive map"),
                    size: tex_size,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: COLOR_FORMAT,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                });
                self.queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &gpu_texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    &tex.rgba,
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(4 * tex.width),
                        rows_per_image: Some(tex.height),
                    },
                    tex_size,
                );
                let tex_view =
                    gpu_texture.create_view(&wgpu::TextureViewDescriptor::default());
                let tex_bind_group =
                    self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("headless-three-renderer emissive map bind group"),
                        layout: &self.emissive_map_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(&tex_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::Sampler(&self.sampler),
                            },
                        ],
                    });
                (tex_bind_group, Some(gpu_texture))
            }
            None => (self.default_emissive_map_bind_group.clone(), None),
        };

        let (ao_map_bind_group, _ao_map_texture) = match &mesh.ao_map {
            Some(tex) => {
                let tex_size = wgpu::Extent3d {
                    width: tex.width,
                    height: tex.height,
                    depth_or_array_layers: 1,
                };
                let gpu_texture = self.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("headless-three-renderer ao map"),
                    size: tex_size,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: COLOR_FORMAT,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                });
                self.queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &gpu_texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    &tex.rgba,
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(4 * tex.width),
                        rows_per_image: Some(tex.height),
                    },
                    tex_size,
                );
                let tex_view =
                    gpu_texture.create_view(&wgpu::TextureViewDescriptor::default());
                let tex_bind_group =
                    self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("headless-three-renderer ao map bind group"),
                        layout: &self.ao_map_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(&tex_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::Sampler(&self.sampler),
                            },
                        ],
                    });
                (tex_bind_group, Some(gpu_texture))
            }
            None => (self.default_ao_map_bind_group.clone(), None),
        };

        Ok(GpuMesh {
            vertex_buffer,
            index_buffer,
            bind_group,
            texture_bind_group,
            normal_map_bind_group,
            mr_map_bind_group,
            emissive_map_bind_group,
            ao_map_bind_group,
            index_count: mesh
                .indices
                .as_ref()
                .map_or(0, |indices| indices.len() as u32),
            vertex_count: mesh.vertices.len() as u32,
            side: mesh.side,
            topology: mesh.topology,
            cast_shadow: mesh.cast_shadow,
            _uniform_buffer: uniform_buffer,
            _texture: _mesh_texture,
            _normal_map: _normal_map_texture,
            _mr_map: _mr_map_texture,
            _emissive_map: _emissive_map_texture,
            _ao_map: _ao_map_texture,
        })
    }
}

fn side_index(side: MeshSide) -> usize {
    match side {
        MeshSide::Front => 0,
        MeshSide::Back => 1,
        MeshSide::Double => 2,
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum PipelineKey {
    Tri(MeshSide),
    Line,
    Point,
}

fn pipeline_key(mesh: &GpuMesh) -> PipelineKey {
    match mesh.topology {
        Topology::Triangles => PipelineKey::Tri(mesh.side),
        Topology::Lines => PipelineKey::Line,
        Topology::Points => PipelineKey::Point,
    }
}

impl GpuRenderer {
    fn pipeline_for(&self, key: PipelineKey, transparent: bool) -> &wgpu::RenderPipeline {
        match key {
            PipelineKey::Tri(side) => {
                let idx = side_index(side);
                if transparent {
                    &self.transparent_pipelines[idx]
                } else {
                    &self.pipelines[idx]
                }
            }
            PipelineKey::Line => &self.line_pipelines[if transparent { 1 } else { 0 }],
            PipelineKey::Point => &self.point_pipelines[if transparent { 1 } else { 0 }],
        }
    }
}

fn partition_draw_order(meshes: &[PreparedMesh], camera_pos: glam::Vec3) -> (Vec<usize>, Vec<usize>) {
    let mut opaque = Vec::new();
    let mut transparent = Vec::new();

    for (i, mesh) in meshes.iter().enumerate() {
        if mesh.is_transparent {
            transparent.push(i);
        } else {
            opaque.push(i);
        }
    }

    // Sort transparent meshes back-to-front (farthest first)
    transparent.sort_by(|&a, &b| {
        let dist_a = mesh_distance_sq(&meshes[a], camera_pos);
        let dist_b = mesh_distance_sq(&meshes[b], camera_pos);
        dist_b.partial_cmp(&dist_a).unwrap_or(std::cmp::Ordering::Equal)
    });

    (opaque, transparent)
}

fn mesh_distance_sq(mesh: &PreparedMesh, camera_pos: glam::Vec3) -> f32 {
    let pos = mesh.transform.w_axis.truncate();
    (pos - camera_pos).length_squared()
}

fn draw_gpu_mesh(pass: &mut wgpu::RenderPass, mesh: &GpuMesh) {
    pass.set_bind_group(0, &mesh.bind_group, &[]);
    pass.set_bind_group(1, &mesh.texture_bind_group, &[]);
    pass.set_bind_group(2, &mesh.normal_map_bind_group, &[]);
    pass.set_bind_group(3, &mesh.mr_map_bind_group, &[]);
    pass.set_bind_group(4, &mesh.emissive_map_bind_group, &[]);
    // bind group 5 (IBL) is set once per pass, not per mesh
    pass.set_bind_group(6, &mesh.ao_map_bind_group, &[]);
    pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
    if let Some(index_buffer) = &mesh.index_buffer {
        pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..mesh.index_count, 0, 0..1);
    } else {
        pass.draw(0..mesh.vertex_count, 0..1);
    }
}

fn create_default_ibl_bind_group(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    // 1x1 black cubemap for irradiance and prefilter
    let black_cube = create_cubemap(device, queue, 1, 1, &[&[0u8, 0, 0, 255] as &[u8]; 6]);
    let irradiance_view = black_cube.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::Cube),
        ..Default::default()
    });
    let prefilter_view = black_cube.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::Cube),
        ..Default::default()
    });

    // 1x1 BRDF LUT with (0, 0, 0, 255)
    let brdf_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("default brdf lut"),
        size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: COLOR_FORMAT,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo { texture: &brdf_tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        &[0u8, 0, 0, 255],
        wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(4), rows_per_image: Some(1) },
        wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
    );
    let brdf_view = brdf_tex.create_view(&wgpu::TextureViewDescriptor::default());

    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("default ibl bind group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&irradiance_view) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&prefilter_view) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&brdf_view) },
            wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(sampler) },
        ],
    })
}

fn create_ibl_bind_group(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    ibl: &IblMaps,
) -> wgpu::BindGroup {
    // Irradiance cubemap
    let irradiance_tex = create_cubemap(
        device, queue, ibl.irradiance_size, 1,
        &ibl.irradiance_faces.iter().map(|f| f.as_slice()).collect::<Vec<_>>(),
    );
    let irradiance_view = irradiance_tex.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::Cube),
        ..Default::default()
    });

    // Prefiltered specular cubemap with mip levels
    let prefilter_tex = create_cubemap_with_mips(
        device, queue, ibl.prefilter_base_size, ibl.prefilter_mip_levels, &ibl.prefilter_faces,
    );
    let prefilter_view = prefilter_tex.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::Cube),
        ..Default::default()
    });

    // BRDF LUT
    let brdf_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("brdf lut"),
        size: wgpu::Extent3d { width: ibl.brdf_lut_size, height: ibl.brdf_lut_size, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: COLOR_FORMAT,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo { texture: &brdf_tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        &ibl.brdf_lut,
        wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(4 * ibl.brdf_lut_size), rows_per_image: Some(ibl.brdf_lut_size) },
        wgpu::Extent3d { width: ibl.brdf_lut_size, height: ibl.brdf_lut_size, depth_or_array_layers: 1 },
    );
    let brdf_view = brdf_tex.create_view(&wgpu::TextureViewDescriptor::default());

    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("ibl bind group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&irradiance_view) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&prefilter_view) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&brdf_view) },
            wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(sampler) },
        ],
    })
}

fn create_cubemap(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    size: u32,
    mip_levels: u32,
    faces: &[&[u8]],
) -> wgpu::Texture {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("cubemap"),
        size: wgpu::Extent3d { width: size, height: size, depth_or_array_layers: 6 },
        mip_level_count: mip_levels,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: COLOR_FORMAT,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    for (face, data) in faces.iter().enumerate() {
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x: 0, y: 0, z: face as u32 },
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * size),
                rows_per_image: Some(size),
            },
            wgpu::Extent3d { width: size, height: size, depth_or_array_layers: 1 },
        );
    }
    texture
}

fn create_cubemap_with_mips(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    base_size: u32,
    mip_levels: u32,
    faces: &[Vec<u8>],
) -> wgpu::Texture {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("prefiltered cubemap"),
        size: wgpu::Extent3d { width: base_size, height: base_size, depth_or_array_layers: 6 },
        mip_level_count: mip_levels,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: COLOR_FORMAT,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    for mip in 0..mip_levels {
        let mip_size = (base_size >> mip).max(1);
        for face in 0..6u32 {
            let idx = (mip * 6 + face) as usize;
            if idx < faces.len() {
                queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &texture,
                        mip_level: mip,
                        origin: wgpu::Origin3d { x: 0, y: 0, z: face },
                        aspect: wgpu::TextureAspect::All,
                    },
                    &faces[idx],
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(4 * mip_size),
                        rows_per_image: Some(mip_size),
                    },
                    wgpu::Extent3d { width: mip_size, height: mip_size, depth_or_array_layers: 1 },
                );
            }
        }
    }
    texture
}
