use anyhow::{Context, Result, anyhow, bail};
use bytemuck::{Pod, Zeroable};
use glam::Mat4;
use wgpu::util::DeviceExt;

use crate::ibl::IblMaps;
use crate::lights::{GpuLight, MAX_LIGHTS};
use crate::mesh::{
    BlendEquation, BlendFactor, BlendMode, CustomBlendState, MAX_CLIPPING_PLANES, MeshSide,
    PreparedMesh, ShadingModel, StencilCompare, StencilOperation, TextureFilter,
    TextureSamplerSettings, Topology, Vertex, WrapMode, prepare_meshes,
};
use crate::settings::{
    BackgroundTexture, OutputColorSpace, OutputFormat, PostProcessingSettings, RenderSettings,
    ShadowKind,
};
use crate::shader::{BACKGROUND_SHADER, POST_SHADER, SHADER, custom_shader_source};
use crate::types::{Camera, RenderScene};
use crate::util::{align_to, encode_png};
use crate::{COLOR_FORMAT, DEPTH_FORMAT};

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Uniforms {
    pub mvp: [[f32; 4]; 4],
    pub view: [[f32; 4]; 4],
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
    /// xyz = LightProbe SH coefficient, w = reserved.
    pub light_probe: [[f32; 4]; 9],
    /// x = has LightProbe, y = has toon gradient map, z = depth packing, w = has matcap color map.
    pub light_probe_params: [f32; 4],
    // x/y = normalScale or bumpScale, z = normal mode (0=none, 1=normalMap, 2=bumpMap), w = has_ibl
    pub normal_map_params: [f32; 4],
    /// x = env_intensity, y/z/w = reserved
    pub ibl_params: [f32; 4],
    /// x = ao_map_intensity, y = has_ao_map, z = has_alpha_map, w = has_light_map
    pub ao_params: [f32; 4],
    /// x = 1/width, y = 1/height, z = width, w = height
    pub render_params: [f32; 4],
    /// x = 1 for LinearSRGBColorSpace output, 0 for SRGBColorSpace output
    pub output_params: [f32; 4],
    /// x/y/z = base-color texture transform row 0 (`u' = x*u + y*v + z`), w = reserved
    pub texture_transform1: [f32; 4],
    /// x/y/z = base-color texture transform row 1 (`v' = x*u + y*v + z`), w = base texture is sRGB
    pub texture_transform2: [f32; 4],
    /// x/y/z = alpha-map texture transform row 0 (`u' = x*u + y*v + z`), w = reserved
    pub alpha_map_transform1: [f32; 4],
    /// x/y/z = alpha-map texture transform row 1 (`v' = x*u + y*v + z`), w = reserved
    pub alpha_map_transform2: [f32; 4],
    /// Row pairs for normal, metallic-roughness, emissive, AO, light, and specular map transforms.
    /// Emissive/light-map row 0 w lanes flag sRGB decode; row 1 w lanes retain UV selection.
    pub map_transform_rows: [[f32; 4]; 12],
    /// Row pairs for clearcoat, clearcoat roughness, clearcoat normal, sheen color,
    /// sheen roughness, anisotropy, transmission, and thickness map transforms.
    pub physical_map_transform_rows: [[f32; 4]; 20],
    /// World-space clipping planes `[nx, ny, nz, constant]`.
    pub clipping_planes: [[f32; 4]; MAX_CLIPPING_PLANES],
    /// x = union plane count, y = total plane count, z = alpha hash enabled, w = premultiplied alpha.
    pub clipping_params: [f32; 4],
    /// xyz = fog color, w = reserved
    pub fog_color: [f32; 4],
    /// x = mode (0=off, 1=linear, 2=exp2), y = near, z = far, w = density
    pub fog_params: [f32; 4],
    pub light_space_matrices: [[[f32; 4]; 4]; 6],
    /// x = has_shadow, y = bias, z = normal_bias, w = receive_shadow
    pub shadow_params: [f32; 4],
    /// x = shadow light index (as f32), y = 1/map_size, z = shadow kind, w = reserved
    pub shadow_params2: [f32; 4],
    /// x/y/z/w = cascade split distances
    pub shadow_params3: [f32; 4],
    /// x = clearcoat, y = clearcoat roughness, z = transmission, w = ior
    pub physical_params1: [f32; 4],
    /// xyz = sheen color, w = sheen roughness
    pub physical_params2: [f32; 4],
    /// x = anisotropy, y = anisotropy rotation, z/w = thickness/attenuation distance or distance near/far.
    pub physical_params3: [f32; 4],
    /// x/y = clearcoat normal scale, z = light_map_intensity, w = has_specular_map
    pub physical_params4: [f32; 4],
    /// xyz = attenuation color or distance reference position, w = reserved
    pub attenuation_color: [f32; 4],
    /// xyz = MeshPhysicalMaterial specular color factor, w = specular intensity.
    pub physical_specular: [f32; 4],
    pub lights: [GpuLight; MAX_LIGHTS],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct PostUniforms {
    pub params1: [f32; 4],
    pub params2: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct BackgroundUniforms {
    pub transform1: [f32; 4],
    pub transform2: [f32; 4],
}

pub struct GpuRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    shader: wgpu::ShaderModule,
    /// Opaque pipelines keyed by `MeshSide` (Front, Back, Double).
    pipelines: [wgpu::RenderPipeline; 3],
    /// Transparent pipelines (no depth write) keyed by `MeshSide`.
    transparent_pipelines: [wgpu::RenderPipeline; 3],
    /// Line / point pipelines: [opaque, transparent] for each.
    line_pipelines: [wgpu::RenderPipeline; 2],
    point_pipelines: [wgpu::RenderPipeline; 2],
    pipeline_layout: wgpu::PipelineLayout,
    post_layout: wgpu::BindGroupLayout,
    post_pipeline: wgpu::RenderPipeline,
    background_pipeline: wgpu::RenderPipeline,
    uniform_layout: wgpu::BindGroupLayout,
    texture_layout: wgpu::BindGroupLayout,
    normal_map_layout: wgpu::BindGroupLayout,
    mr_map_layout: wgpu::BindGroupLayout,
    emissive_map_layout: wgpu::BindGroupLayout,
    ibl_layout: wgpu::BindGroupLayout,
    ao_map_layout: wgpu::BindGroupLayout,
    shadow_layout: wgpu::BindGroupLayout,
    /// Depth-only pipeline used to render the shadow map.
    shadow_pipelines: [wgpu::RenderPipeline; 6],
    sampler: wgpu::Sampler,
    shadow_sampler: wgpu::Sampler,
    _default_texture: wgpu::Texture,
    _default_normal_map_texture: wgpu::Texture,
    default_texture_bind_group: wgpu::BindGroup,
    default_normal_map_bind_group: wgpu::BindGroup,
    default_mr_map_bind_group: wgpu::BindGroup,
    default_emissive_map_bind_group: wgpu::BindGroup,
    default_ibl_bind_group: wgpu::BindGroup,
    default_ao_map_bind_group: wgpu::BindGroup,
    _default_physical_layers_texture: wgpu::Texture,
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
    pipeline_override: Option<wgpu::RenderPipeline>,
    index_count: u32,
    vertex_count: u32,
    side: MeshSide,
    topology: Topology,
    blend_constant: Option<wgpu::Color>,
    stencil_reference: Option<u32>,
    cast_shadow: bool,
    _uniform_buffer: wgpu::Buffer,
    _texture: Option<wgpu::Texture>,
    _normal_map: Option<wgpu::Texture>,
    _mr_map: Option<wgpu::Texture>,
    _emissive_map: Option<wgpu::Texture>,
    _ao_map: Option<wgpu::Texture>,
    _light_map: Option<wgpu::Texture>,
    _alpha_map: Option<wgpu::Texture>,
    _physical_layers_map: Option<wgpu::Texture>,
    _physical_sheen_map: Option<wgpu::Texture>,
    _physical_specular_map: Option<wgpu::Texture>,
    _clearcoat_normal_map: Option<wgpu::Texture>,
}

struct GpuBackground {
    bind_group: wgpu::BindGroup,
    _texture: wgpu::Texture,
    _uniform_buffer: wgpu::Buffer,
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
        let post_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("headless-three-renderer post shader"),
            source: wgpu::ShaderSource::Wgsl(POST_SHADER.into()),
        });
        let background_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("headless-three-renderer background shader"),
            source: wgpu::ShaderSource::Wgsl(BACKGROUND_SHADER.into()),
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

        let emissive_map_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
            label: Some("headless-three-renderer ao and physical map layout"),
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
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
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
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 10,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 11,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 12,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 13,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 14,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 15,
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
                        view_dimension: wgpu::TextureViewDimension::D2Array,
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

        let post_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("headless-three-renderer post layout"),
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
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            std::mem::size_of::<PostUniforms>() as u64,
                        ),
                    },
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
        let default_texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
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
        let default_normal_map_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
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
        let default_mr_map_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
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
        let default_ibl_bind_group =
            create_default_ibl_bind_group(&device, &queue, &ibl_layout, &sampler);

        // Default physical layers: layer 0 is neutral scalar/specular data, layer 1 is
        // the default +X anisotropy direction with full strength.
        let default_physical_layers_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("headless-three-renderer default physical layers map"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 2,
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
                texture: &default_physical_layers_texture,
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
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &default_physical_layers_texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x: 0, y: 0, z: 1 },
                aspect: wgpu::TextureAspect::All,
            },
            &[255u8, 128, 255, 255],
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
        let default_physical_layers_view =
            default_physical_layers_texture.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                ..Default::default()
            });

        // Default AO map: reuse the 1x1 white default texture; red channel = 1.0
        // means full illumination (no occlusion).
        let default_ao_map_view =
            default_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let default_ao_map_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("headless-three-renderer default ao and physical maps bind group"),
            layout: &ao_map_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&default_ao_map_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&default_physical_layers_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&default_ao_map_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&default_ao_map_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&default_normal_map_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(&default_ao_map_view),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::TextureView(&default_ao_map_view),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 13,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 14,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 15,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("headless-three-renderer pipeline layout"),
            bind_group_layouts: &[
                Some(&uniform_layout),
                Some(&texture_layout),
                Some(&normal_map_layout),
                Some(&mr_map_layout),
                Some(&emissive_map_layout),
                Some(&ibl_layout),
                Some(&ao_map_layout),
                Some(&shadow_layout),
            ],
            immediate_size: 0,
        });

        // 1x1 depth texture used as a "no shadow" default binding.
        let default_shadow_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("headless-three-renderer default shadow texture"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let default_shadow_view =
            default_shadow_texture.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                ..Default::default()
            });
        let default_shadow_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("headless-three-renderer default shadow bind group"),
            layout: &shadow_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&default_shadow_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&shadow_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&default_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        // Dedicated pipeline layout for the depth-only shadow pass (uniforms only).
        let shadow_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("headless-three-renderer shadow pipeline layout"),
                bind_group_layouts: &[Some(&uniform_layout)],
                immediate_size: 0,
            });
        let make_shadow_pipeline = |entry_point: &'static str, label: &'static str| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&shadow_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some(entry_point),
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
            })
        };
        let shadow_pipelines = [
            make_shadow_pipeline("vs_shadow0", "headless-three-renderer shadow pipeline 0"),
            make_shadow_pipeline("vs_shadow1", "headless-three-renderer shadow pipeline 1"),
            make_shadow_pipeline("vs_shadow2", "headless-three-renderer shadow pipeline 2"),
            make_shadow_pipeline("vs_shadow3", "headless-three-renderer shadow pipeline 3"),
            make_shadow_pipeline("vs_shadow4", "headless-three-renderer shadow pipeline 4"),
            make_shadow_pipeline("vs_shadow5", "headless-three-renderer shadow pipeline 5"),
        ];

        let vertex_buffers = [Vertex::layout()];
        let make_pipeline = |topology: Topology, side: MeshSide, transparent: bool| {
            let label = match (topology, side, transparent) {
                (Topology::Triangles, MeshSide::Front, false) => "pipeline (tri front)",
                (Topology::Triangles, MeshSide::Back, false) => "pipeline (tri back)",
                (Topology::Triangles, MeshSide::Double, false) => "pipeline (tri double)",
                (Topology::Triangles, MeshSide::Front, true) => "pipeline (tri front, transparent)",
                (Topology::Triangles, MeshSide::Back, true) => "pipeline (tri back, transparent)",
                (Topology::Triangles, MeshSide::Double, true) => {
                    "pipeline (tri double, transparent)"
                }
                (Topology::Lines, _, false) => "pipeline (lines)",
                (Topology::Lines, _, true) => "pipeline (lines, transparent)",
                (Topology::Points, _, false) => "pipeline (points)",
                (Topology::Points, _, true) => "pipeline (points, transparent)",
            };
            let depth_write = !transparent;
            let color_targets = [Some(color_target_state(
                default_blend_state(transparent),
                true,
            ))];
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
                    depth_compare: Some(wgpu::CompareFunction::LessEqual),
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

        let screen_color_targets = [Some(color_target_state(None, true))];
        let post_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("headless-three-renderer post pipeline layout"),
            bind_group_layouts: &[Some(&post_layout)],
            immediate_size: 0,
        });
        let post_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("headless-three-renderer post pipeline"),
            layout: Some(&post_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &post_shader,
                entry_point: Some("vs_post"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &post_shader,
                entry_point: Some("fs_post"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &screen_color_targets,
            }),
            multiview_mask: None,
            cache: None,
        });

        let background_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("headless-three-renderer background pipeline layout"),
                bind_group_layouts: &[Some(&post_layout)],
                immediate_size: 0,
            });
        let background_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("headless-three-renderer background pipeline"),
            layout: Some(&background_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &background_shader,
                entry_point: Some("vs_background"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &background_shader,
                entry_point: Some("fs_background"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &screen_color_targets,
            }),
            multiview_mask: None,
            cache: None,
        });

        Ok(Self {
            device,
            queue,
            shader,
            pipelines,
            transparent_pipelines,
            line_pipelines,
            point_pipelines,
            pipeline_layout,
            post_layout,
            post_pipeline,
            background_pipeline,
            uniform_layout,
            texture_layout,
            normal_map_layout,
            mr_map_layout,
            emissive_map_layout,
            ibl_layout,
            ao_map_layout,
            shadow_layout,
            shadow_pipelines,
            sampler,
            shadow_sampler,
            _default_texture: default_texture,
            _default_normal_map_texture: default_normal_map,
            default_texture_bind_group,
            default_normal_map_bind_group,
            default_mr_map_bind_group,
            default_emissive_map_bind_group,
            default_ibl_bind_group,
            default_ao_map_bind_group,
            _default_physical_layers_texture: default_physical_layers_texture,
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
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::TEXTURE_BINDING,
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
            Some(ibl) => create_ibl_bind_group(
                &self.device,
                &self.queue,
                &self.ibl_layout,
                &self.sampler,
                ibl,
            ),
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

        let (opaque_order, transparent_order) = partition_draw_order(meshes);

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
        let background_gpu = settings
            .background_texture
            .as_ref()
            .map(|background| self.upload_background(background, settings.output_color_space));
        let background_clear = wgpu::Color {
            r: settings.background[0] * f64::from(settings.background_intensity),
            g: settings.background[1] * f64::from(settings.background_intensity),
            b: settings.background[2] * f64::from(settings.background_intensity),
            a: settings.background[3],
        };

        if let Some(background) = &background_gpu {
            let color_attachments = [Some(wgpu::RenderPassColorAttachment {
                view: &color_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(background_clear),
                    store: wgpu::StoreOp::Store,
                },
            })];

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("headless-three-renderer background render pass"),
                color_attachments: &color_attachments,
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            apply_output_region(&mut pass, settings);
            pass.set_pipeline(&self.background_pipeline);
            pass.set_bind_group(0, &background.bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        {
            let color_load = match background_gpu.as_ref() {
                Some(_) => wgpu::LoadOp::Load,
                None => wgpu::LoadOp::Clear(background_clear),
            };
            let color_attachments = [Some(wgpu::RenderPassColorAttachment {
                view: &color_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: color_load,
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
                    stencil_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0),
                        store: wgpu::StoreOp::Store,
                    }),
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            apply_output_region(&mut pass, settings);

            // Opaque meshes first (with depth write)
            pass.set_bind_group(5, &ibl_bind_group, &[]);
            pass.set_bind_group(7, &shadow_bind_group, &[]);
            let mut current_pipeline: Option<PipelineKey> = None;
            for &i in &opaque_order {
                let mesh = &gpu_meshes[i];
                if let Some(pipeline) = &mesh.pipeline_override {
                    pass.set_pipeline(pipeline);
                    current_pipeline = None;
                } else {
                    let key = pipeline_key(mesh);
                    if current_pipeline != Some(key) {
                        pass.set_pipeline(self.pipeline_for(key, false));
                        current_pipeline = Some(key);
                    }
                }
                if let Some(color) = mesh.blend_constant {
                    pass.set_blend_constant(color);
                }
                if let Some(reference) = mesh.stencil_reference {
                    pass.set_stencil_reference(reference);
                }
                draw_gpu_mesh(&mut pass, mesh);
            }
        }

        if !transparent_order.is_empty() {
            let scene_color_texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("headless-three-renderer scene color texture"),
                size: texture_size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: COLOR_FORMAT,
                usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            encoder.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &color_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyTextureInfo {
                    texture: &scene_color_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                texture_size,
            );
            let scene_color_view =
                scene_color_texture.create_view(&wgpu::TextureViewDescriptor::default());
            let shadow_view = match &_shadow_texture {
                Some(texture) => texture.create_view(&wgpu::TextureViewDescriptor {
                    dimension: Some(wgpu::TextureViewDimension::D2Array),
                    array_layer_count: settings.shadow.as_ref().map(|s| s.layer_count),
                    ..Default::default()
                }),
                None => self
                    ._default_shadow_texture
                    .create_view(&wgpu::TextureViewDescriptor {
                        dimension: Some(wgpu::TextureViewDimension::D2Array),
                        ..Default::default()
                    }),
            };
            let scene_shadow_bind_group =
                self.create_shadow_scene_bind_group(&shadow_view, &scene_color_view);

            let color_attachments = [Some(wgpu::RenderPassColorAttachment {
                view: &color_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })];

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("headless-three-renderer transparent render pass"),
                color_attachments: &color_attachments,
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            apply_output_region(&mut pass, settings);

            pass.set_bind_group(5, &ibl_bind_group, &[]);
            pass.set_bind_group(7, &scene_shadow_bind_group, &[]);
            let mut current_pipeline: Option<PipelineKey> = None;
            for &i in &transparent_order {
                let mesh = &gpu_meshes[i];
                if let Some(pipeline) = &mesh.pipeline_override {
                    pass.set_pipeline(pipeline);
                    current_pipeline = None;
                } else {
                    let key = pipeline_key(mesh);
                    if current_pipeline != Some(key) {
                        pass.set_pipeline(self.pipeline_for(key, true));
                        current_pipeline = Some(key);
                    }
                }
                if let Some(color) = mesh.blend_constant {
                    pass.set_blend_constant(color);
                }
                if let Some(reference) = mesh.stencil_reference {
                    pass.set_stencil_reference(reference);
                }
                draw_gpu_mesh(&mut pass, mesh);
            }
        }

        if settings.post_processing.active {
            let post_texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("headless-three-renderer post color texture"),
                size: texture_size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: COLOR_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let post_view = post_texture.create_view(&wgpu::TextureViewDescriptor::default());
            let post_uniforms = post_uniforms(settings.post_processing);
            let post_uniform_buffer =
                self.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("headless-three-renderer post uniform buffer"),
                        contents: bytemuck::bytes_of(&post_uniforms),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });
            let post_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("headless-three-renderer post bind group"),
                layout: &self.post_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&color_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: post_uniform_buffer.as_entire_binding(),
                    },
                ],
            });
            let color_attachments = [Some(wgpu::RenderPassColorAttachment {
                view: &post_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })];
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("headless-three-renderer post pass"),
                color_attachments: &color_attachments,
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.post_pipeline);
            pass.set_bind_group(0, &post_bind_group, &[]);
            pass.draw(0..3, 0..1);
            drop(pass);

            copy_texture_to_output(
                &mut encoder,
                &post_texture,
                &output_buffer,
                padded_bytes_per_row,
                settings.height,
                texture_size,
            );
        } else {
            copy_texture_to_output(
                &mut encoder,
                &color_texture,
                &output_buffer,
                padded_bytes_per_row,
                settings.height,
                texture_size,
            );
        }

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

    /// Render the scene's shadow casters into a depth-only texture array from
    /// the shadow light's POV and return a bind group referencing it.
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
            size: wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: shadow.layer_count,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let shadow_view = shadow_texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            array_layer_count: Some(shadow.layer_count),
            ..Default::default()
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("headless-three-renderer shadow encoder"),
            });

        for layer in 0..shadow.layer_count {
            let layer_view = shadow_texture.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_array_layer: layer,
                array_layer_count: Some(1),
                ..Default::default()
            });
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("headless-three-renderer shadow pass"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &layer_view,
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
            pass.set_pipeline(&self.shadow_pipelines[layer as usize]);
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

        let default_scene_view = self
            ._default_texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("headless-three-renderer shadow bind group"),
            layout: &self.shadow_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&shadow_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.shadow_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&default_scene_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });
        let _ = &shadow.light_dir; // silence unused warning; direction baked into light_vp
        (bind_group, shadow_texture)
    }

    fn create_shadow_scene_bind_group(
        &self,
        shadow_view: &wgpu::TextureView,
        scene_color_view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("headless-three-renderer shadow scene-color bind group"),
            layout: &self.shadow_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(shadow_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.shadow_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(scene_color_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        })
    }

    fn sampler_for_texture(
        &self,
        wrap_s: WrapMode,
        wrap_t: WrapMode,
        mag_filter: TextureFilter,
        min_filter: TextureFilter,
    ) -> wgpu::Sampler {
        if wrap_s == WrapMode::ClampToEdge
            && wrap_t == WrapMode::ClampToEdge
            && mag_filter == TextureFilter::Linear
            && min_filter == TextureFilter::Linear
        {
            return self.sampler.clone();
        }
        self.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("headless-three-renderer per-mesh sampler"),
            address_mode_u: wrap_s.to_address_mode(),
            address_mode_v: wrap_t.to_address_mode(),
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: mag_filter.to_filter_mode(),
            min_filter: min_filter.to_filter_mode(),
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        })
    }

    fn sampler_for_settings(&self, settings: TextureSamplerSettings) -> wgpu::Sampler {
        self.sampler_for_texture(
            settings.wrap_s,
            settings.wrap_t,
            settings.mag_filter,
            settings.min_filter,
        )
    }

    fn upload_background(
        &self,
        background: &BackgroundTexture,
        output_color_space: OutputColorSpace,
    ) -> GpuBackground {
        let gpu_texture = self.upload_texture(
            "headless-three-renderer scene background texture",
            &background.texture,
        );
        let texture_view = gpu_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = self.sampler_for_texture(
            background.texture.wrap_s,
            background.texture.wrap_t,
            background.texture.mag_filter,
            background.texture.min_filter,
        );
        let background_flags = if background.is_srgb { 1.0 } else { 0.0 }
            + if output_color_space.is_linear() {
                2.0
            } else {
                0.0
            }
            + background.blurriness * 0.25;
        let uniforms = BackgroundUniforms {
            transform1: [
                background.transform[0],
                background.transform[1],
                background.transform[2],
                background.intensity,
            ],
            transform2: [
                background.transform[3],
                background.transform[4],
                background.transform[5],
                background_flags,
            ],
        };
        let uniform_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("headless-three-renderer background uniform buffer"),
                contents: bytemuck::bytes_of(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("headless-three-renderer background bind group"),
            layout: &self.post_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });
        GpuBackground {
            bind_group,
            _texture: gpu_texture,
            _uniform_buffer: uniform_buffer,
        }
    }

    fn upload_texture(
        &self,
        label: &'static str,
        tex: &crate::mesh::PreparedTexture,
    ) -> wgpu::Texture {
        let tex_size = wgpu::Extent3d {
            width: tex.width,
            height: tex.height,
            depth_or_array_layers: 1,
        };
        let gpu_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: tex_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: COLOR_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
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
        gpu_texture
    }

    fn upload_physical_layers_texture(
        &self,
        label: &'static str,
        scalar: &crate::mesh::PreparedTexture,
        anisotropy: Option<&crate::mesh::PreparedTexture>,
    ) -> wgpu::Texture {
        let tex_size = wgpu::Extent3d {
            width: scalar.width,
            height: scalar.height,
            depth_or_array_layers: 2,
        };
        let gpu_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: tex_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: COLOR_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let layer_size = wgpu::Extent3d {
            width: scalar.width,
            height: scalar.height,
            depth_or_array_layers: 1,
        };
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &gpu_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &scalar.rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * scalar.width),
                rows_per_image: Some(scalar.height),
            },
            layer_size,
        );

        let mut default_anisotropy = Vec::new();
        let anisotropy_rgba = match anisotropy {
            Some(tex) if tex.width == scalar.width && tex.height == scalar.height => {
                tex.rgba.as_slice()
            }
            _ => {
                default_anisotropy.reserve_exact((scalar.width * scalar.height * 4) as usize);
                for _ in 0..(scalar.width * scalar.height) {
                    default_anisotropy.extend_from_slice(&[255u8, 128, 255, 255]);
                }
                default_anisotropy.as_slice()
            }
        };
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &gpu_texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x: 0, y: 0, z: 1 },
                aspect: wgpu::TextureAspect::All,
            },
            anisotropy_rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * scalar.width),
                rows_per_image: Some(scalar.height),
            },
            layer_size,
        );
        gpu_texture
    }

    fn create_custom_pipeline(
        &self,
        mesh: &PreparedMesh,
        fragment_body: &str,
    ) -> Result<wgpu::RenderPipeline> {
        let source = custom_shader_source(fragment_body);
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("headless-three-renderer custom material shader"),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
        Ok(self.create_material_pipeline(
            &shader,
            mesh,
            "headless-three-renderer custom material pipeline",
        ))
    }

    fn create_state_override_pipeline(&self, mesh: &PreparedMesh) -> wgpu::RenderPipeline {
        self.create_material_pipeline(
            &self.shader,
            mesh,
            "headless-three-renderer material state override pipeline",
        )
    }

    fn create_material_pipeline(
        &self,
        shader: &wgpu::ShaderModule,
        mesh: &PreparedMesh,
        label: &'static str,
    ) -> wgpu::RenderPipeline {
        let color_targets = [Some(wgpu::ColorTargetState {
            format: COLOR_FORMAT,
            blend: blend_state(
                mesh.blending,
                mesh.custom_blend,
                mesh.is_transparent,
                mesh.premultiplied_alpha,
            ),
            write_mask: if mesh.color_write {
                wgpu::ColorWrites::ALL
            } else {
                wgpu::ColorWrites::empty()
            },
        })];
        self.device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&self.pipeline_layout),
                vertex: wgpu::VertexState {
                    module: shader,
                    entry_point: Some("vs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[Vertex::layout()],
                },
                primitive: wgpu::PrimitiveState {
                    topology: mesh.topology.primitive(),
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: match mesh.topology {
                        Topology::Triangles => mesh.side.cull_mode(),
                        Topology::Lines | Topology::Points => None,
                    },
                    unclipped_depth: false,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    conservative: false,
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: Some(mesh.depth_write),
                    depth_compare: Some(if mesh.depth_test {
                        wgpu::CompareFunction::LessEqual
                    } else {
                        wgpu::CompareFunction::Always
                    }),
                    stencil: stencil_state(mesh),
                    bias: depth_bias_state(mesh),
                }),
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: shader,
                    entry_point: Some("fs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &color_targets,
                }),
                multiview_mask: None,
                cache: None,
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

        let physical_params2 = if mesh.shading_model == ShadingModel::Phong {
            [
                mesh.specular_color[0],
                mesh.specular_color[1],
                mesh.specular_color[2],
                mesh.shininess,
            ]
        } else {
            [
                mesh.sheen_color[0],
                mesh.sheen_color[1],
                mesh.sheen_color[2],
                mesh.sheen_roughness,
            ]
        };
        let distance_reference_position = mesh.distance_reference_position.unwrap_or([
            settings.camera_pos.x,
            settings.camera_pos.y,
            settings.camera_pos.z,
        ]);
        let distance_near = mesh.distance_near.unwrap_or(settings.near);
        let distance_far = mesh.distance_far.unwrap_or(settings.far);

        let uniforms = Uniforms {
            mvp: mvp.to_cols_array_2d(),
            view: settings.view.to_cols_array_2d(),
            model: model.to_cols_array_2d(),
            normal_matrix: normal_matrix.to_cols_array_2d(),
            camera_pos: [
                settings.camera_pos.x,
                settings.camera_pos.y,
                settings.camera_pos.z,
                0.0,
            ],
            base_color: mesh.base_color,
            emissive: [
                mesh.emissive[0],
                mesh.emissive[1],
                mesh.emissive[2],
                mesh.alpha_test,
            ],
            metallic: mesh.metallic,
            roughness: mesh.roughness,
            ambient_intensity: settings.ambient_intensity,
            num_lights: settings.lights.len().min(MAX_LIGHTS) as u32,
            ambient_color: [
                settings.ambient_color[0],
                settings.ambient_color[1],
                settings.ambient_color[2],
                0.0,
            ],
            light_probe: light_probe_rows(settings),
            light_probe_params: [
                if settings.has_light_probe { 1.0 } else { 0.0 },
                if mesh.gradient_map.is_some() {
                    1.0
                } else {
                    0.0
                },
                mesh.depth_packing.as_u32() as f32,
                if mesh.matcap_map.is_some() { 1.0 } else { 0.0 },
            ],
            normal_map_params: [
                if mesh.normal_map.is_some() {
                    mesh.normal_scale[0]
                } else {
                    mesh.bump_scale
                },
                if mesh.normal_map.is_some() {
                    mesh.normal_scale[1]
                } else {
                    0.0
                },
                if mesh.normal_map.is_some() {
                    1.0
                } else if mesh.bump_map.is_some() {
                    2.0
                } else {
                    0.0
                },
                if settings.ibl.is_some() { 1.0 } else { 0.0 },
            ],
            ibl_params: [
                settings.env_intensity,
                mesh.shading_model.as_u32() as f32,
                settings.near,
                settings.far,
            ],
            ao_params: [
                mesh.ao_map_intensity,
                if mesh.ao_map.is_some() { 1.0 } else { 0.0 },
                if mesh.alpha_map.is_some() { 1.0 } else { 0.0 },
                if mesh.light_map.is_some() { 1.0 } else { 0.0 },
            ],
            render_params: [
                1.0 / settings.width as f32,
                1.0 / settings.height as f32,
                settings.width as f32,
                settings.height as f32,
            ],
            output_params: [
                if settings.output_color_space.is_linear() {
                    1.0
                } else {
                    0.0
                },
                0.0,
                0.0,
                0.0,
            ],
            texture_transform1: [
                mesh.texture_transform[0],
                mesh.texture_transform[1],
                mesh.texture_transform[2],
                if mesh.texture_uses_uv2 { 1.0 } else { 0.0 },
            ],
            texture_transform2: [
                mesh.texture_transform[3],
                mesh.texture_transform[4],
                mesh.texture_transform[5],
                if mesh.texture_is_srgb { 1.0 } else { 0.0 },
            ],
            alpha_map_transform1: [
                mesh.alpha_map_transform[0],
                mesh.alpha_map_transform[1],
                mesh.alpha_map_transform[2],
                0.0,
            ],
            alpha_map_transform2: [
                mesh.alpha_map_transform[3],
                mesh.alpha_map_transform[4],
                mesh.alpha_map_transform[5],
                if mesh.alpha_map_uses_uv2 { 1.0 } else { 0.0 },
            ],
            map_transform_rows: map_transform_rows(mesh),
            physical_map_transform_rows: physical_map_transform_rows(mesh),
            clipping_planes: mesh.clipping_planes,
            clipping_params: [
                mesh.clipping_union_count as f32,
                mesh.clipping_plane_count as f32,
                if mesh.alpha_hash { 1.0 } else { 0.0 },
                if mesh.premultiplied_alpha { 1.0 } else { 0.0 },
            ],
            fog_color: [
                settings.fog.color[0],
                settings.fog.color[1],
                settings.fog.color[2],
                0.0,
            ],
            fog_params: [
                if settings.fog.active && mesh.fog {
                    settings.fog.mode
                } else {
                    0.0
                },
                settings.fog.near,
                settings.fog.far,
                settings.fog.density,
            ],
            light_space_matrices: settings
                .shadow
                .as_ref()
                .map(|s| s.light_vps.map(|matrix| matrix.to_cols_array_2d()))
                .unwrap_or_else(|| [Mat4::IDENTITY.to_cols_array_2d(); 6]),
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
                Some(s) => [
                    s.light_index as f32,
                    1.0 / s.map_size as f32,
                    match s.kind {
                        ShadowKind::DirectionalOrSpot => 0.0,
                        ShadowKind::Point => 1.0,
                        ShadowKind::Cascaded => 2.0,
                    },
                    s.layer_count as f32,
                ],
                None => [0.0, 0.0, 0.0, 0.0],
            },
            shadow_params3: settings
                .shadow
                .as_ref()
                .map(|s| s.cascade_splits)
                .unwrap_or([f32::MAX; 4]),
            physical_params1: [
                mesh.clearcoat,
                mesh.clearcoat_roughness,
                mesh.transmission,
                mesh.ior,
            ],
            physical_params2,
            physical_params3: if mesh.shading_model == ShadingModel::Distance {
                [
                    mesh.anisotropy,
                    mesh.anisotropy_rotation,
                    distance_near,
                    distance_far,
                ]
            } else {
                [
                    mesh.anisotropy,
                    mesh.anisotropy_rotation,
                    mesh.thickness,
                    mesh.attenuation_distance,
                ]
            },
            physical_params4: [
                mesh.clearcoat_normal_scale[0],
                mesh.clearcoat_normal_scale[1],
                mesh.light_map_intensity,
                if mesh.shading_model == ShadingModel::Matcap {
                    if mesh.matcap_map_is_srgb { 1.0 } else { 0.0 }
                } else if mesh.specular_map.is_some() {
                    1.0
                } else {
                    0.0
                },
            ],
            attenuation_color: if mesh.shading_model == ShadingModel::Distance {
                [
                    distance_reference_position[0],
                    distance_reference_position[1],
                    distance_reference_position[2],
                    0.0,
                ]
            } else {
                [
                    mesh.attenuation_color[0],
                    mesh.attenuation_color[1],
                    mesh.attenuation_color[2],
                    0.0,
                ]
            },
            physical_specular: [
                mesh.physical_specular_color[0],
                mesh.physical_specular_color[1],
                mesh.physical_specular_color[2],
                mesh.physical_specular_intensity,
            ],
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
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
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
                let tex_view = gpu_texture.create_view(&wgpu::TextureViewDescriptor::default());
                let sampler_for_tex = self.sampler_for_texture(
                    tex.wrap_s,
                    tex.wrap_t,
                    tex.mag_filter,
                    tex.min_filter,
                );
                let tex_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
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

        let (normal_map_bind_group, _normal_map_texture) =
            match mesh.normal_map.as_ref().or(mesh.bump_map.as_ref()) {
                Some(tex) => {
                    let tex_size = wgpu::Extent3d {
                        width: tex.width,
                        height: tex.height,
                        depth_or_array_layers: 1,
                    };
                    let gpu_texture = self.device.create_texture(&wgpu::TextureDescriptor {
                        label: Some("headless-three-renderer normal or bump map"),
                        size: tex_size,
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: COLOR_FORMAT,
                        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
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
                    let tex_view = gpu_texture.create_view(&wgpu::TextureViewDescriptor::default());
                    let sampler_for_tex = self.sampler_for_texture(
                        tex.wrap_s,
                        tex.wrap_t,
                        tex.mag_filter,
                        tex.min_filter,
                    );
                    let tex_bind_group =
                        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("headless-three-renderer normal or bump map bind group"),
                            layout: &self.normal_map_layout,
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
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
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
                let tex_view = gpu_texture.create_view(&wgpu::TextureViewDescriptor::default());
                let sampler_for_tex = self.sampler_for_texture(
                    tex.wrap_s,
                    tex.wrap_t,
                    tex.mag_filter,
                    tex.min_filter,
                );
                let tex_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("headless-three-renderer metallic-roughness bind group"),
                    layout: &self.mr_map_layout,
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
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
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
                let tex_view = gpu_texture.create_view(&wgpu::TextureViewDescriptor::default());
                let sampler_for_tex = self.sampler_for_texture(
                    tex.wrap_s,
                    tex.wrap_t,
                    tex.mag_filter,
                    tex.min_filter,
                );
                let tex_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("headless-three-renderer emissive map bind group"),
                    layout: &self.emissive_map_layout,
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
            None => (self.default_emissive_map_bind_group.clone(), None),
        };

        let (
            ao_map_bind_group,
            _ao_map_texture,
            _light_map_texture,
            _alpha_map_texture,
            _physical_layers_map_texture,
            _physical_sheen_map_texture,
            _physical_specular_map_texture,
            _clearcoat_normal_map_texture,
        ) = if mesh.ao_map.is_some()
            || mesh.light_map.is_some()
            || mesh.specular_map.is_some()
            || mesh.alpha_map.is_some()
            || mesh.matcap_map.is_some()
            || mesh.gradient_map.is_some()
            || mesh.physical_maps.is_some()
            || mesh.clearcoat_normal_map.is_some()
        {
            let ao_texture = mesh
                .ao_map
                .as_ref()
                .map(|tex| self.upload_texture("headless-three-renderer ao map", tex));
            let light_texture = mesh
                .light_map
                .as_ref()
                .map(|tex| self.upload_texture("headless-three-renderer light map", tex));
            let alpha_texture = mesh
                .alpha_map
                .as_ref()
                .map(|tex| self.upload_texture("headless-three-renderer alpha map", tex));
            let physical_layers_texture =
                match (mesh.physical_maps.as_ref(), mesh.specular_map.as_ref()) {
                    (Some(maps), _) => Some(self.upload_physical_layers_texture(
                        "headless-three-renderer physical layers map",
                        &maps.scalar_map,
                        Some(&maps.anisotropy_map),
                    )),
                    (None, Some(tex)) => Some(self.upload_physical_layers_texture(
                        "headless-three-renderer specular and physical layers map",
                        tex,
                        None,
                    )),
                    (None, None) => None,
                };
            let physical_sheen_texture = match (
                mesh.matcap_map.as_ref(),
                mesh.gradient_map.as_ref(),
                mesh.physical_maps.as_ref(),
            ) {
                (Some(tex), _, _) => {
                    Some(self.upload_texture("headless-three-renderer matcap color map", tex))
                }
                (None, Some(tex), _) => {
                    Some(self.upload_texture("headless-three-renderer toon gradient map", tex))
                }
                (None, None, Some(maps)) => Some(self.upload_texture(
                    "headless-three-renderer physical sheen map",
                    &maps.sheen_map,
                )),
                (None, None, None) => None,
            };
            let physical_specular_texture = mesh.physical_maps.as_ref().map(|maps| {
                self.upload_texture(
                    "headless-three-renderer physical specular map",
                    &maps.specular_map,
                )
            });
            let clearcoat_normal_texture = mesh.clearcoat_normal_map.as_ref().map(|tex| {
                self.upload_texture("headless-three-renderer clearcoat normal map", tex)
            });

            let default_white_view = self
                ._default_texture
                .create_view(&wgpu::TextureViewDescriptor::default());
            let default_normal_view = self
                ._default_normal_map_texture
                .create_view(&wgpu::TextureViewDescriptor::default());
            let default_physical_layers_view =
                self._default_physical_layers_texture
                    .create_view(&wgpu::TextureViewDescriptor {
                        dimension: Some(wgpu::TextureViewDimension::D2Array),
                        ..Default::default()
                    });
            let ao_view = ao_texture
                .as_ref()
                .map(|texture| texture.create_view(&wgpu::TextureViewDescriptor::default()));
            let light_view = light_texture
                .as_ref()
                .map(|texture| texture.create_view(&wgpu::TextureViewDescriptor::default()));
            let alpha_view = alpha_texture
                .as_ref()
                .map(|texture| texture.create_view(&wgpu::TextureViewDescriptor::default()));
            let physical_layers_view = physical_layers_texture.as_ref().map(|texture| {
                texture.create_view(&wgpu::TextureViewDescriptor {
                    dimension: Some(wgpu::TextureViewDimension::D2Array),
                    ..Default::default()
                })
            });
            let physical_sheen_view = physical_sheen_texture
                .as_ref()
                .map(|texture| texture.create_view(&wgpu::TextureViewDescriptor::default()));
            let physical_specular_view = physical_specular_texture
                .as_ref()
                .map(|texture| texture.create_view(&wgpu::TextureViewDescriptor::default()));
            let clearcoat_normal_view = clearcoat_normal_texture
                .as_ref()
                .map(|texture| texture.create_view(&wgpu::TextureViewDescriptor::default()));
            let ao_sampler = mesh
                .ao_map
                .as_ref()
                .map(|tex| {
                    self.sampler_for_texture(tex.wrap_s, tex.wrap_t, tex.mag_filter, tex.min_filter)
                })
                .unwrap_or_else(|| self.sampler.clone());
            let alpha_sampler = mesh
                .alpha_map
                .as_ref()
                .map(|tex| {
                    self.sampler_for_texture(tex.wrap_s, tex.wrap_t, tex.mag_filter, tex.min_filter)
                })
                .unwrap_or_else(|| self.sampler.clone());
            let light_sampler = mesh
                .light_map
                .as_ref()
                .map(|tex| {
                    self.sampler_for_texture(tex.wrap_s, tex.wrap_t, tex.mag_filter, tex.min_filter)
                })
                .unwrap_or_else(|| self.sampler.clone());
            let specular_sampler = match (mesh.physical_maps.as_ref(), mesh.specular_map.as_ref()) {
                (None, Some(tex)) => {
                    self.sampler_for_texture(tex.wrap_s, tex.wrap_t, tex.mag_filter, tex.min_filter)
                }
                _ => self.sampler.clone(),
            };
            let physical_layers_sampler = mesh
                .physical_maps
                .as_ref()
                .map(|maps| self.sampler_for_settings(maps.physical_layers_sampler))
                .unwrap_or_else(|| self.sampler.clone());
            let physical_sheen_sampler = mesh
                .physical_maps
                .as_ref()
                .map(|maps| self.sampler_for_settings(maps.sheen_sampler))
                .unwrap_or_else(|| self.sampler.clone());
            let physical_specular_sampler = mesh
                .physical_maps
                .as_ref()
                .map(|maps| self.sampler_for_settings(maps.specular_sampler))
                .unwrap_or_else(|| self.sampler.clone());
            let clearcoat_normal_sampler = mesh
                .clearcoat_normal_map
                .as_ref()
                .map(|tex| {
                    self.sampler_for_texture(tex.wrap_s, tex.wrap_t, tex.mag_filter, tex.min_filter)
                })
                .unwrap_or_else(|| self.sampler.clone());

            let tex_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("headless-three-renderer ao and physical maps bind group"),
                layout: &self.ao_map_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(
                            ao_view.as_ref().unwrap_or(&default_white_view),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(
                            physical_layers_view
                                .as_ref()
                                .unwrap_or(&default_physical_layers_view),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(
                            physical_sheen_view.as_ref().unwrap_or(&default_white_view),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(
                            physical_specular_view
                                .as_ref()
                                .unwrap_or(&default_white_view),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(
                            clearcoat_normal_view
                                .as_ref()
                                .unwrap_or(&default_normal_view),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: wgpu::BindingResource::TextureView(
                            alpha_view.as_ref().unwrap_or(&default_white_view),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: wgpu::BindingResource::TextureView(
                            light_view.as_ref().unwrap_or(&default_white_view),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: wgpu::BindingResource::Sampler(&ao_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 9,
                        resource: wgpu::BindingResource::Sampler(&alpha_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 10,
                        resource: wgpu::BindingResource::Sampler(&light_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 11,
                        resource: wgpu::BindingResource::Sampler(&specular_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 12,
                        resource: wgpu::BindingResource::Sampler(&physical_layers_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 13,
                        resource: wgpu::BindingResource::Sampler(&physical_sheen_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 14,
                        resource: wgpu::BindingResource::Sampler(&physical_specular_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 15,
                        resource: wgpu::BindingResource::Sampler(&clearcoat_normal_sampler),
                    },
                ],
            });
            (
                tex_bind_group,
                ao_texture,
                light_texture,
                alpha_texture,
                physical_layers_texture,
                physical_sheen_texture,
                physical_specular_texture,
                clearcoat_normal_texture,
            )
        } else {
            (
                self.default_ao_map_bind_group.clone(),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )
        };

        let pipeline_override = match mesh.custom_fragment_shader.as_deref() {
            Some(fragment_body) => Some(self.create_custom_pipeline(mesh, fragment_body)?),
            None if requires_pipeline_override(mesh) => {
                Some(self.create_state_override_pipeline(mesh))
            }
            None => None,
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
            pipeline_override,
            index_count: mesh
                .indices
                .as_ref()
                .map_or(0, |indices| indices.len() as u32),
            vertex_count: mesh.vertices.len() as u32,
            side: mesh.side,
            topology: mesh.topology,
            blend_constant: blend_constant(mesh.custom_blend),
            stencil_reference: if mesh.stencil_write {
                Some(mesh.stencil_ref)
            } else {
                None
            },
            cast_shadow: mesh.cast_shadow,
            _uniform_buffer: uniform_buffer,
            _texture: _mesh_texture,
            _normal_map: _normal_map_texture,
            _mr_map: _mr_map_texture,
            _emissive_map: _emissive_map_texture,
            _ao_map: _ao_map_texture,
            _light_map: _light_map_texture,
            _alpha_map: _alpha_map_texture,
            _physical_layers_map: _physical_layers_map_texture,
            _physical_sheen_map: _physical_sheen_map_texture,
            _physical_specular_map: _physical_specular_map_texture,
            _clearcoat_normal_map: _clearcoat_normal_map_texture,
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

fn apply_output_region(pass: &mut wgpu::RenderPass<'_>, settings: &RenderSettings) {
    if let Some(viewport) = settings.viewport {
        pass.set_viewport(
            viewport.x,
            viewport.y,
            viewport.width,
            viewport.height,
            0.0,
            1.0,
        );
    }
    if let Some(scissor) = settings.scissor {
        pass.set_scissor_rect(scissor.x, scissor.y, scissor.width, scissor.height);
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

fn requires_pipeline_override(mesh: &PreparedMesh) -> bool {
    let default_depth_write = !mesh.is_transparent;
    let default_blending = if mesh.is_transparent {
        BlendMode::Normal
    } else {
        BlendMode::None
    };
    !mesh.depth_test
        || mesh.depth_write != default_depth_write
        || !mesh.color_write
        || mesh.polygon_offset
        || mesh.stencil_write
        || (mesh.premultiplied_alpha
            && effective_blend_mode(mesh.blending, mesh.is_transparent) != BlendMode::None)
        || effective_blend_mode(mesh.blending, mesh.is_transparent) != default_blending
}

fn depth_bias_state(mesh: &PreparedMesh) -> wgpu::DepthBiasState {
    if mesh.polygon_offset {
        wgpu::DepthBiasState {
            constant: mesh.polygon_offset_units,
            slope_scale: mesh.polygon_offset_factor,
            clamp: 0.0,
        }
    } else {
        wgpu::DepthBiasState::default()
    }
}

fn stencil_state(mesh: &PreparedMesh) -> wgpu::StencilState {
    if !mesh.stencil_write {
        return wgpu::StencilState::default();
    }
    let face = wgpu::StencilFaceState {
        compare: stencil_compare(mesh.stencil_func),
        fail_op: stencil_operation(mesh.stencil_fail),
        depth_fail_op: stencil_operation(mesh.stencil_z_fail),
        pass_op: stencil_operation(mesh.stencil_z_pass),
    };
    wgpu::StencilState {
        front: face,
        back: face,
        read_mask: mesh.stencil_func_mask,
        write_mask: mesh.stencil_write_mask,
    }
}

fn stencil_compare(compare: StencilCompare) -> wgpu::CompareFunction {
    match compare {
        StencilCompare::Never => wgpu::CompareFunction::Never,
        StencilCompare::Less => wgpu::CompareFunction::Less,
        StencilCompare::Equal => wgpu::CompareFunction::Equal,
        StencilCompare::LessEqual => wgpu::CompareFunction::LessEqual,
        StencilCompare::Greater => wgpu::CompareFunction::Greater,
        StencilCompare::NotEqual => wgpu::CompareFunction::NotEqual,
        StencilCompare::GreaterEqual => wgpu::CompareFunction::GreaterEqual,
        StencilCompare::Always => wgpu::CompareFunction::Always,
    }
}

fn stencil_operation(operation: StencilOperation) -> wgpu::StencilOperation {
    match operation {
        StencilOperation::Zero => wgpu::StencilOperation::Zero,
        StencilOperation::Keep => wgpu::StencilOperation::Keep,
        StencilOperation::Replace => wgpu::StencilOperation::Replace,
        StencilOperation::IncrementClamp => wgpu::StencilOperation::IncrementClamp,
        StencilOperation::DecrementClamp => wgpu::StencilOperation::DecrementClamp,
        StencilOperation::IncrementWrap => wgpu::StencilOperation::IncrementWrap,
        StencilOperation::DecrementWrap => wgpu::StencilOperation::DecrementWrap,
        StencilOperation::Invert => wgpu::StencilOperation::Invert,
    }
}

fn color_target_state(
    blend: Option<wgpu::BlendState>,
    color_write: bool,
) -> wgpu::ColorTargetState {
    wgpu::ColorTargetState {
        format: COLOR_FORMAT,
        blend,
        write_mask: if color_write {
            wgpu::ColorWrites::ALL
        } else {
            wgpu::ColorWrites::empty()
        },
    }
}

fn default_blend_state(transparent: bool) -> Option<wgpu::BlendState> {
    if transparent {
        blend_state(BlendMode::Normal, None, true, false)
    } else {
        None
    }
}

fn effective_blend_mode(mode: BlendMode, is_transparent: bool) -> BlendMode {
    match mode {
        BlendMode::Normal if !is_transparent => BlendMode::None,
        other => other,
    }
}

fn blend_state(
    mode: BlendMode,
    custom: Option<CustomBlendState>,
    is_transparent: bool,
    premultiplied_alpha: bool,
) -> Option<wgpu::BlendState> {
    match effective_blend_mode(mode, is_transparent) {
        BlendMode::None => None,
        BlendMode::Normal => Some(wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: if premultiplied_alpha {
                    wgpu::BlendFactor::One
                } else {
                    wgpu::BlendFactor::SrcAlpha
                },
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
        }),
        BlendMode::Additive => Some(wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: if premultiplied_alpha {
                    wgpu::BlendFactor::One
                } else {
                    wgpu::BlendFactor::SrcAlpha
                },
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
        }),
        BlendMode::Subtractive => Some(wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Zero,
                dst_factor: wgpu::BlendFactor::OneMinusSrc,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Zero,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
        }),
        BlendMode::Multiply => Some(wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Dst,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Zero,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
        }),
        BlendMode::Custom => custom.map(custom_blend_state),
    }
}

fn custom_blend_state(state: CustomBlendState) -> wgpu::BlendState {
    wgpu::BlendState {
        color: wgpu::BlendComponent {
            src_factor: blend_factor(state.color_src_factor),
            dst_factor: blend_factor(state.color_dst_factor),
            operation: blend_operation(state.color_equation),
        },
        alpha: wgpu::BlendComponent {
            src_factor: blend_factor(state.alpha_src_factor),
            dst_factor: blend_factor(state.alpha_dst_factor),
            operation: blend_operation(state.alpha_equation),
        },
    }
}

fn blend_operation(equation: BlendEquation) -> wgpu::BlendOperation {
    match equation {
        BlendEquation::Add => wgpu::BlendOperation::Add,
        BlendEquation::Subtract => wgpu::BlendOperation::Subtract,
        BlendEquation::ReverseSubtract => wgpu::BlendOperation::ReverseSubtract,
        BlendEquation::Min => wgpu::BlendOperation::Min,
        BlendEquation::Max => wgpu::BlendOperation::Max,
    }
}

fn blend_factor(factor: BlendFactor) -> wgpu::BlendFactor {
    match factor {
        BlendFactor::Zero => wgpu::BlendFactor::Zero,
        BlendFactor::One => wgpu::BlendFactor::One,
        BlendFactor::SrcColor => wgpu::BlendFactor::Src,
        BlendFactor::OneMinusSrcColor => wgpu::BlendFactor::OneMinusSrc,
        BlendFactor::SrcAlpha => wgpu::BlendFactor::SrcAlpha,
        BlendFactor::OneMinusSrcAlpha => wgpu::BlendFactor::OneMinusSrcAlpha,
        BlendFactor::DstAlpha => wgpu::BlendFactor::DstAlpha,
        BlendFactor::OneMinusDstAlpha => wgpu::BlendFactor::OneMinusDstAlpha,
        BlendFactor::DstColor => wgpu::BlendFactor::Dst,
        BlendFactor::OneMinusDstColor => wgpu::BlendFactor::OneMinusDst,
        BlendFactor::SrcAlphaSaturate => wgpu::BlendFactor::SrcAlphaSaturated,
        BlendFactor::ConstantColor | BlendFactor::ConstantAlpha => wgpu::BlendFactor::Constant,
        BlendFactor::OneMinusConstantColor | BlendFactor::OneMinusConstantAlpha => {
            wgpu::BlendFactor::OneMinusConstant
        }
    }
}

fn blend_constant(state: Option<CustomBlendState>) -> Option<wgpu::Color> {
    let state = state?;
    if !uses_constant_factor(state) {
        return None;
    }
    let alpha_as_rgb = matches!(
        state.color_src_factor,
        BlendFactor::ConstantAlpha | BlendFactor::OneMinusConstantAlpha
    ) || matches!(
        state.color_dst_factor,
        BlendFactor::ConstantAlpha | BlendFactor::OneMinusConstantAlpha
    );
    let (r, g, b) = if alpha_as_rgb {
        (
            state.constant[3] as f64,
            state.constant[3] as f64,
            state.constant[3] as f64,
        )
    } else {
        (
            state.constant[0] as f64,
            state.constant[1] as f64,
            state.constant[2] as f64,
        )
    };
    Some(wgpu::Color {
        r,
        g,
        b,
        a: state.constant[3] as f64,
    })
}

fn uses_constant_factor(state: CustomBlendState) -> bool {
    [
        state.color_src_factor,
        state.color_dst_factor,
        state.alpha_src_factor,
        state.alpha_dst_factor,
    ]
    .iter()
    .any(|factor| {
        matches!(
            factor,
            BlendFactor::ConstantColor
                | BlendFactor::OneMinusConstantColor
                | BlendFactor::ConstantAlpha
                | BlendFactor::OneMinusConstantAlpha
        )
    })
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

fn partition_draw_order(meshes: &[PreparedMesh]) -> (Vec<usize>, Vec<usize>) {
    let mut opaque = Vec::new();
    let mut transparent = Vec::new();

    for (i, mesh) in meshes.iter().enumerate() {
        if mesh.is_transparent {
            transparent.push(i);
        } else {
            opaque.push(i);
        }
    }

    opaque.sort_by(|&a, &b| compare_opaque_meshes(&meshes[a], &meshes[b]));

    // Sort transparent meshes back-to-front (farthest first)
    transparent.sort_by(|&a, &b| compare_transparent_meshes(&meshes[a], &meshes[b]));

    (opaque, transparent)
}

fn compare_opaque_meshes(a: &PreparedMesh, b: &PreparedMesh) -> std::cmp::Ordering {
    compare_f32(a.group_order, b.group_order)
        .then_with(|| compare_f32(a.render_order, b.render_order))
        .then_with(|| a.material_sort_key.cmp(&b.material_sort_key))
        .then_with(|| compare_f32(a.sort_z, b.sort_z))
        .then_with(|| a.sort_index.cmp(&b.sort_index))
}

fn compare_transparent_meshes(a: &PreparedMesh, b: &PreparedMesh) -> std::cmp::Ordering {
    compare_f32(a.group_order, b.group_order)
        .then_with(|| compare_f32(a.render_order, b.render_order))
        .then_with(|| compare_f32(b.sort_z, a.sort_z))
        .then_with(|| a.sort_index.cmp(&b.sort_index))
}

fn compare_f32(a: f32, b: f32) -> std::cmp::Ordering {
    a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal)
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

fn map_transform_rows(mesh: &PreparedMesh) -> [[f32; 4]; 12] {
    let transforms = [
        if mesh.normal_map.is_some() {
            mesh.normal_map_transform
        } else {
            mesh.bump_map_transform
        },
        mesh.metallic_roughness_texture_transform,
        mesh.emissive_map_transform,
        mesh.ao_map_transform,
        mesh.light_map_transform,
        mesh.specular_map_transform,
    ];
    let mut rows = [[0.0; 4]; 12];
    for (index, transform) in transforms.iter().enumerate() {
        let row = index * 2;
        rows[row] = [transform[0], transform[1], transform[2], 0.0];
        rows[row + 1] = [transform[3], transform[4], transform[5], 0.0];
    }
    rows[1][3] = if mesh.normal_map.is_some() {
        if mesh.normal_map_uses_uv2 { 1.0 } else { 0.0 }
    } else if mesh.bump_map.is_some() {
        if mesh.bump_map_uses_uv2 { 1.0 } else { 0.0 }
    } else {
        0.0
    };
    rows[3][3] = if mesh.metallic_roughness_texture_uses_uv2 {
        1.0
    } else {
        0.0
    };
    rows[4][3] = if mesh.emissive_map_is_srgb { 1.0 } else { 0.0 };
    rows[5][3] = if mesh.emissive_map_uses_uv2 { 1.0 } else { 0.0 };
    rows[8][3] = if mesh.light_map_is_srgb { 1.0 } else { 0.0 };
    rows
}

fn physical_map_transform_rows(mesh: &PreparedMesh) -> [[f32; 4]; 20] {
    let transforms = [
        mesh.clearcoat_map_transform,
        mesh.clearcoat_roughness_map_transform,
        mesh.clearcoat_normal_map_transform,
        if mesh.shading_model == ShadingModel::Matcap {
            mesh.matcap_map_transform
        } else {
            mesh.sheen_color_map_transform
        },
        mesh.sheen_roughness_map_transform,
        mesh.anisotropy_map_transform,
        mesh.transmission_map_transform,
        mesh.thickness_map_transform,
        mesh.specular_color_map_transform,
        mesh.specular_intensity_map_transform,
    ];
    let mut rows = [[0.0; 4]; 20];
    for (index, transform) in transforms.iter().enumerate() {
        let row = index * 2;
        rows[row] = [transform[0], transform[1], transform[2], 0.0];
        rows[row + 1] = [transform[3], transform[4], transform[5], 0.0];
    }
    if mesh.shading_model == ShadingModel::Matcap {
        rows[7][3] = if mesh.matcap_map_uses_uv2 { 1.0 } else { 0.0 };
    } else {
        rows[7][3] = if mesh.sheen_color_map_uses_uv2 {
            1.0
        } else {
            0.0
        };
    }
    rows[1][3] = if mesh.clearcoat_map_uses_uv2 {
        1.0
    } else {
        0.0
    };
    rows[3][3] = if mesh.clearcoat_roughness_map_uses_uv2 {
        1.0
    } else {
        0.0
    };
    rows[5][3] = if mesh.clearcoat_normal_map_uses_uv2 {
        1.0
    } else {
        0.0
    };
    rows[9][3] = if mesh.sheen_roughness_map_uses_uv2 {
        1.0
    } else {
        0.0
    };
    rows[11][3] = if mesh.anisotropy_map_uses_uv2 {
        1.0
    } else {
        0.0
    };
    rows[13][3] = if mesh.transmission_map_uses_uv2 {
        1.0
    } else {
        0.0
    };
    rows[15][3] = if mesh.thickness_map_uses_uv2 {
        1.0
    } else {
        0.0
    };
    rows[17][3] = if mesh.specular_color_map_uses_uv2 {
        1.0
    } else {
        0.0
    };
    rows[19][3] = if mesh.specular_intensity_map_uses_uv2 {
        1.0
    } else {
        0.0
    };
    rows
}

fn light_probe_rows(settings: &RenderSettings) -> [[f32; 4]; 9] {
    let mut rows = [[0.0; 4]; 9];
    for (index, coefficient) in settings.light_probe.iter().enumerate() {
        rows[index] = [coefficient[0], coefficient[1], coefficient[2], 0.0];
    }
    rows
}

fn post_uniforms(settings: PostProcessingSettings) -> PostUniforms {
    PostUniforms {
        params1: [
            settings.exposure,
            settings.contrast,
            settings.saturation,
            settings.vignette,
        ],
        params2: [settings.grayscale, settings.invert, 0.0, 0.0],
    }
}

fn copy_texture_to_output(
    encoder: &mut wgpu::CommandEncoder,
    texture: &wgpu::Texture,
    output_buffer: &wgpu::Buffer,
    padded_bytes_per_row: u32,
    height: u32,
    texture_size: wgpu::Extent3d,
) {
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: output_buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row),
                rows_per_image: Some(height),
            },
        },
        texture_size,
    );
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
            texture: &brdf_tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &[0u8, 0, 0, 255],
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
    let brdf_view = brdf_tex.create_view(&wgpu::TextureViewDescriptor::default());

    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("default ibl bind group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&irradiance_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&prefilter_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&brdf_view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
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
        device,
        queue,
        ibl.irradiance_size,
        1,
        &ibl.irradiance_faces
            .iter()
            .map(|f| f.as_slice())
            .collect::<Vec<_>>(),
    );
    let irradiance_view = irradiance_tex.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::Cube),
        ..Default::default()
    });

    // Prefiltered specular cubemap with mip levels
    let prefilter_tex = create_cubemap_with_mips(
        device,
        queue,
        ibl.prefilter_base_size,
        ibl.prefilter_mip_levels,
        &ibl.prefilter_faces,
    );
    let prefilter_view = prefilter_tex.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::Cube),
        ..Default::default()
    });

    // BRDF LUT
    let brdf_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("brdf lut"),
        size: wgpu::Extent3d {
            width: ibl.brdf_lut_size,
            height: ibl.brdf_lut_size,
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
            texture: &brdf_tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &ibl.brdf_lut,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4 * ibl.brdf_lut_size),
            rows_per_image: Some(ibl.brdf_lut_size),
        },
        wgpu::Extent3d {
            width: ibl.brdf_lut_size,
            height: ibl.brdf_lut_size,
            depth_or_array_layers: 1,
        },
    );
    let brdf_view = brdf_tex.create_view(&wgpu::TextureViewDescriptor::default());

    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("ibl bind group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&irradiance_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&prefilter_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&brdf_view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
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
        size: wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 6,
        },
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
                origin: wgpu::Origin3d {
                    x: 0,
                    y: 0,
                    z: face as u32,
                },
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * size),
                rows_per_image: Some(size),
            },
            wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
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
        size: wgpu::Extent3d {
            width: base_size,
            height: base_size,
            depth_or_array_layers: 6,
        },
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
                        origin: wgpu::Origin3d {
                            x: 0,
                            y: 0,
                            z: face,
                        },
                        aspect: wgpu::TextureAspect::All,
                    },
                    &faces[idx],
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(4 * mip_size),
                        rows_per_image: Some(mip_size),
                    },
                    wgpu::Extent3d {
                        width: mip_size,
                        height: mip_size,
                        depth_or_array_layers: 1,
                    },
                );
            }
        }
    }
    texture
}
