use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::{Mutex, MutexGuard};

use anyhow::{Context, Result, anyhow, bail};
use bytemuck::{Pod, Zeroable};
use glam::Mat4;
use wgpu::util::DeviceExt;

use crate::ibl::IblMaps;
use crate::lights::{GpuLight, MAX_LIGHTS};
use crate::mesh::{
    BlendEquation, BlendFactor, BlendMode, CustomBlendState, MAX_CLIPPING_PLANES, MeshSide,
    MipmapFilter, PreparedMesh, PreparedTexture, ShadingModel, StencilCompare, StencilOperation,
    TextureFilter, TextureSamplerSettings, Topology, Vertex, WrapMode, prepare_meshes,
};
use crate::settings::{
    BackgroundTexture, BackgroundTextureMapping, MAX_SHADOW_LAYERS, OutputColorSpace, OutputFormat,
    PostProcessingSettings, RenderSettings, ShadowKind,
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
    // x/y = normalScale or bumpScale, z = normal mode (0=none, 1=tangent normalMap, 2=bumpMap, 3=object normalMap), w = has_ibl
    pub normal_map_params: [f32; 4],
    /// x = env_intensity, y = shading_model, z = camera near, w = camera far
    pub ibl_params: [f32; 4],
    /// x = legacy env combine, y = reflectivity, z = material env mode (0=off, 1=reflect, 2=refract), w = refraction ratio
    pub env_map_params: [f32; 4],
    /// x = ao_map_intensity, y = has_ao_map, z = has_alpha_map, w = has_light_map
    pub ao_params: [f32; 4],
    /// x = 1/width, y = 1/height, z = width, w = height
    pub render_params: [f32; 4],
    /// x = 1 for LinearSRGBColorSpace output, 0 for SRGBColorSpace output; y = material toneMapped; z = alpha-to-coverage active; w = toneMappingExposure.
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
    /// Row pairs for current physical-extension map transforms.
    pub physical_map_transform_rows: [[f32; 4]; 24],
    /// World-space clipping planes `[nx, ny, nz, constant]`.
    pub clipping_planes: [[f32; 4]; MAX_CLIPPING_PLANES],
    /// x = union plane count, y = total plane count, z = alpha hash enabled, w = premultiplied alpha.
    pub clipping_params: [f32; 4],
    /// xyz = fog color, w = reserved
    pub fog_color: [f32; 4],
    /// x = mode (0=off, 1=linear, 2=exp2), y = near, z = far, w = density
    pub fog_params: [f32; 4],
    pub light_space_matrices: [[[f32; 4]; 4]; MAX_SHADOW_LAYERS],
    /// x = shadow count, y = first bias, z = first normal_bias, w = receive_shadow
    pub shadow_params: [f32; 4],
    /// x = first shadow light index, y = 1/map_width, z = 1/map_height, w = first shadow kind.
    pub shadow_params2: [f32; 4],
    /// x/y/z = first cascade split distances, w = shadow layer count.
    pub shadow_params3: [f32; 4],
    /// x = PCF radius multiplier, y = clip shadow caster fragments by clipping planes, z = explicit shadow side (0 double/no-cull, 1 front, 2 back), w = shadow-only alpha cutoff.
    pub shadow_params4: [f32; 4],
    /// x = light index, y = layer base, z = layer count, w = shadow kind.
    pub shadow_infos: [[f32; 4]; MAX_SHADOW_LAYERS],
    /// x = bias, y = normal_bias, z = PCF radius multiplier, w = reserved.
    pub shadow_biases: [[f32; 4]; MAX_SHADOW_LAYERS],
    /// x/y/z = cascade split distances, w = reserved.
    pub shadow_cascade_splits: [[f32; 4]; MAX_SHADOW_LAYERS],
    /// x = clearcoat, y = clearcoat roughness, z = transmission, w = ior
    pub physical_params1: [f32; 4],
    /// xyz = sheen color, w = sheen roughness
    pub physical_params2: [f32; 4],
    /// x = anisotropy, y = anisotropy rotation, z/w = thickness/attenuation distance or distance near/far.
    pub physical_params3: [f32; 4],
    /// x/y = clearcoat normal scale, z = light_map_intensity, w = has_specular_map, matcap map sRGB flag, or toon gradient map sRGB flag depending on shading model.
    pub physical_params4: [f32; 4],
    /// xyz = attenuation color or distance reference position, w = dispersion for physical materials.
    pub attenuation_color: [f32; 4],
    /// xyz = MeshPhysicalMaterial specular color factor, w = specular intensity.
    pub physical_specular: [f32; 4],
    /// x = iridescence, y = iridescence IOR, z/w = iridescence thickness range in nanometers.
    pub iridescence_params: [f32; 4],
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
    pub inverse_view_projection: [[f32; 4]; 4],
    pub camera_params: [f32; 4],
    pub rotation1: [f32; 4],
    pub rotation2: [f32; 4],
    pub rotation3: [f32; 4],
}

pub struct GpuRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    shader: wgpu::ShaderModule,
    /// Opaque pipelines keyed by `MeshSide` (Front, Back, Double).
    pipelines: [wgpu::RenderPipeline; 3],
    pipelines_msaa4: [wgpu::RenderPipeline; 3],
    /// Transparent pipelines (no depth write) keyed by `MeshSide`.
    transparent_pipelines: [wgpu::RenderPipeline; 3],
    transparent_pipelines_msaa4: [wgpu::RenderPipeline; 3],
    /// Line / point pipelines: [opaque, transparent] for each.
    line_pipelines: [wgpu::RenderPipeline; 2],
    line_pipelines_msaa4: [wgpu::RenderPipeline; 2],
    point_pipelines: [wgpu::RenderPipeline; 2],
    point_pipelines_msaa4: [wgpu::RenderPipeline; 2],
    pipeline_layout: wgpu::PipelineLayout,
    post_layout: wgpu::BindGroupLayout,
    background_layout: wgpu::BindGroupLayout,
    post_pipeline: wgpu::RenderPipeline,
    background_pipeline: wgpu::RenderPipeline,
    background_pipeline_msaa4: wgpu::RenderPipeline,
    uniform_layout: wgpu::BindGroupLayout,
    texture_layout: wgpu::BindGroupLayout,
    normal_map_layout: wgpu::BindGroupLayout,
    mr_map_layout: wgpu::BindGroupLayout,
    emissive_map_layout: wgpu::BindGroupLayout,
    ibl_layout: wgpu::BindGroupLayout,
    ao_map_layout: wgpu::BindGroupLayout,
    shadow_layout: wgpu::BindGroupLayout,
    /// Depth-only pipeline used to render the shadow map.
    shadow_pipelines: [wgpu::RenderPipeline; MAX_SHADOW_LAYERS],
    line_shadow_pipelines: [wgpu::RenderPipeline; MAX_SHADOW_LAYERS],
    sampler: wgpu::Sampler,
    sampler_cache: Mutex<HashMap<SamplerKey, wgpu::Sampler>>,
    texture_cache: Mutex<HashMap<TextureCacheKey, wgpu::Texture>>,
    physical_layers_texture_cache: Mutex<HashMap<PhysicalLayersTextureCacheKey, wgpu::Texture>>,
    texture_bind_group_cache: Mutex<HashMap<TextureBindGroupKey, wgpu::BindGroup>>,
    ao_physical_bind_group_cache: Mutex<HashMap<AoPhysicalBindGroupKey, wgpu::BindGroup>>,
    background_bind_group_cache: Mutex<HashMap<BackgroundBindGroupKey, CachedBackgroundBindGroup>>,
    ibl_bind_group_cache: Mutex<HashMap<IblBindGroupKey, wgpu::BindGroup>>,
    uniform_bind_group_cache: Mutex<HashMap<UniformBindGroupKey, CachedUniformBindGroup>>,
    post_uniform_buffer: Mutex<Option<wgpu::Buffer>>,
    scene_color_texture_cache: Mutex<HashMap<ScratchTextureKey, wgpu::Texture>>,
    post_texture_cache: Mutex<HashMap<ScratchTextureKey, wgpu::Texture>>,
    readback_buffer_cache: Mutex<HashMap<ReadbackBufferKey, wgpu::Buffer>>,
    mesh_buffer_cache: Mutex<HashMap<MeshBufferCacheKey, CachedMeshBuffers>>,
    state_pipeline_cache: Mutex<HashMap<StatePipelineKey, wgpu::RenderPipeline>>,
    custom_pipeline_cache: Mutex<HashMap<CustomPipelineKey, wgpu::RenderPipeline>>,
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

struct GpuBackground<'a> {
    bind_group: wgpu::BindGroup,
    _texture: wgpu::Texture,
    _uniform_buffer: wgpu::Buffer,
    _cache_guard: MutexGuard<'a, HashMap<BackgroundBindGroupKey, CachedBackgroundBindGroup>>,
}

#[derive(Clone)]
struct CachedUniformBindGroup {
    buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

#[derive(Clone)]
struct CachedBackgroundBindGroup {
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
struct ScratchTextureKey {
    width: u32,
    height: u32,
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
struct ReadbackBufferKey {
    size: u64,
}

struct AoPhysicalBindGroupResources {
    bind_group: wgpu::BindGroup,
    ao_texture: Option<wgpu::Texture>,
    light_texture: Option<wgpu::Texture>,
    alpha_texture: Option<wgpu::Texture>,
    physical_layers_texture: Option<wgpu::Texture>,
    physical_sheen_texture: Option<wgpu::Texture>,
    physical_specular_texture: Option<wgpu::Texture>,
    clearcoat_normal_texture: Option<wgpu::Texture>,
}

#[derive(Clone)]
struct CachedMeshBuffers {
    vertex_buffer: wgpu::Buffer,
    index_buffer: Option<wgpu::Buffer>,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct MeshBufferCacheKey {
    vertex: BufferCacheKey,
    index: Option<BufferCacheKey>,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct BufferCacheKey {
    len: usize,
    hash: u64,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct UniformBindGroupKey {
    len: usize,
    hash: u64,
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
struct SamplerKey {
    wrap_s: WrapMode,
    wrap_t: WrapMode,
    mag_filter: TextureFilter,
    min_filter: TextureFilter,
    mipmap_filter: MipmapFilter,
    mip_lod_enabled: bool,
    anisotropy_clamp: u16,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct TextureCacheKey {
    width: u32,
    height: u32,
    mipmap_filter: MipmapFilter,
    rgba_len: usize,
    rgba_hash: u64,
    mipmaps: Vec<TextureMipCacheKey>,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct TextureMipCacheKey {
    width: u32,
    height: u32,
    rgba_len: usize,
    rgba_hash: u64,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct PhysicalLayersTextureCacheKey {
    scalar: TextureCacheKey,
    anisotropy: Option<TextureCacheKey>,
    iridescence: Option<TextureCacheKey>,
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
enum TextureBindGroupKind {
    BaseColor,
    NormalOrBump,
    MetallicRoughness,
    Emissive,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct TextureBindGroupKey {
    kind: TextureBindGroupKind,
    texture: TextureCacheKey,
    sampler: SamplerKey,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct BackgroundBindGroupKey {
    texture: TextureCacheKey,
    sampler: SamplerKey,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct IblBindGroupKey {
    irradiance_size: u32,
    irradiance_faces: Vec<BufferCacheKey>,
    prefilter_base_size: u32,
    prefilter_mip_levels: u32,
    prefilter_faces: Vec<BufferCacheKey>,
    brdf_lut_size: u32,
    brdf_lut: BufferCacheKey,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct AoPhysicalBindGroupKey {
    ao: Option<TextureCacheKey>,
    physical_layers: Option<PhysicalLayersTextureCacheKey>,
    physical_sheen: Option<TextureCacheKey>,
    physical_specular: Option<TextureCacheKey>,
    clearcoat_normal: Option<TextureCacheKey>,
    alpha: Option<TextureCacheKey>,
    light: Option<TextureCacheKey>,
    ao_sampler: SamplerKey,
    alpha_sampler: SamplerKey,
    light_sampler: SamplerKey,
    specular_sampler: SamplerKey,
    physical_layers_sampler: SamplerKey,
    physical_sheen_sampler: SamplerKey,
    physical_specular_sampler: SamplerKey,
    clearcoat_normal_sampler: SamplerKey,
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
struct StatePipelineKey {
    topology: Topology,
    side: MeshSide,
    sample_count: u32,
    blending: BlendMode,
    custom_blend: Option<CustomBlendPipelineKey>,
    is_transparent: bool,
    premultiplied_alpha: bool,
    depth_test: bool,
    depth_func: StencilCompare,
    depth_write: bool,
    color_write: bool,
    polygon_offset: bool,
    polygon_offset_factor: u32,
    polygon_offset_units: i32,
    alpha_to_coverage: bool,
    stencil_write: bool,
    stencil_write_mask: u32,
    stencil_func: StencilCompare,
    stencil_ref: u32,
    stencil_func_mask: u32,
    stencil_fail: StencilOperation,
    stencil_z_fail: StencilOperation,
    stencil_z_pass: StencilOperation,
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
struct CustomBlendPipelineKey {
    color_equation: BlendEquation,
    alpha_equation: BlendEquation,
    color_src_factor: BlendFactor,
    color_dst_factor: BlendFactor,
    alpha_src_factor: BlendFactor,
    alpha_dst_factor: BlendFactor,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct CustomPipelineKey {
    state: StatePipelineKey,
    fragment_body: String,
}

impl MeshBufferCacheKey {
    fn from_mesh(mesh: &PreparedMesh) -> Self {
        let vertex_bytes = bytemuck::cast_slice::<Vertex, u8>(&mesh.vertices);
        let index_bytes = mesh
            .indices
            .as_ref()
            .map(|indices| bytemuck::cast_slice::<u32, u8>(indices));
        Self::from_bytes(vertex_bytes, index_bytes)
    }

    fn from_bytes(vertex_bytes: &[u8], index_bytes: Option<&[u8]>) -> Self {
        Self {
            vertex: BufferCacheKey::from_bytes(vertex_bytes),
            index: index_bytes.map(BufferCacheKey::from_bytes),
        }
    }
}

impl BufferCacheKey {
    fn from_bytes(bytes: &[u8]) -> Self {
        Self {
            len: bytes.len(),
            hash: hash_bytes(bytes),
        }
    }
}

impl UniformBindGroupKey {
    fn from_uniforms(uniforms: &Uniforms) -> Self {
        let bytes = bytemuck::bytes_of(uniforms);
        Self {
            len: bytes.len(),
            hash: hash_bytes(bytes),
        }
    }
}

impl ScratchTextureKey {
    fn from_extent(size: wgpu::Extent3d) -> Self {
        Self {
            width: size.width,
            height: size.height,
        }
    }
}

impl ReadbackBufferKey {
    fn from_size(size: u64) -> Self {
        Self { size }
    }
}

impl TextureCacheKey {
    fn from_texture(texture: &PreparedTexture) -> Self {
        Self {
            width: texture.width,
            height: texture.height,
            mipmap_filter: texture.mipmap_filter,
            rgba_len: texture.rgba.len(),
            rgba_hash: hash_bytes(&texture.rgba),
            mipmaps: texture
                .mipmaps
                .iter()
                .map(|mip| TextureMipCacheKey {
                    width: mip.width,
                    height: mip.height,
                    rgba_len: mip.rgba.len(),
                    rgba_hash: hash_bytes(&mip.rgba),
                })
                .collect(),
        }
    }
}

impl PhysicalLayersTextureCacheKey {
    fn from_layers(
        scalar: &PreparedTexture,
        anisotropy: Option<&PreparedTexture>,
        iridescence: Option<&PreparedTexture>,
    ) -> Self {
        Self {
            scalar: TextureCacheKey::from_texture(scalar),
            anisotropy: matching_layer_key(scalar, anisotropy),
            iridescence: matching_layer_key(scalar, iridescence),
        }
    }
}

fn matching_layer_key(
    scalar: &PreparedTexture,
    layer: Option<&PreparedTexture>,
) -> Option<TextureCacheKey> {
    layer
        .filter(|texture| texture.width == scalar.width && texture.height == scalar.height)
        .map(TextureCacheKey::from_texture)
}

impl TextureBindGroupKey {
    fn new(kind: TextureBindGroupKind, texture: &PreparedTexture) -> Self {
        Self {
            kind,
            texture: TextureCacheKey::from_texture(texture),
            sampler: SamplerKey::from_texture(texture),
        }
    }
}

impl BackgroundBindGroupKey {
    fn new(texture: &PreparedTexture) -> Self {
        Self {
            texture: TextureCacheKey::from_texture(texture),
            sampler: SamplerKey::from_texture(texture),
        }
    }
}

impl IblBindGroupKey {
    fn from_maps(ibl: &IblMaps) -> Self {
        Self {
            irradiance_size: ibl.irradiance_size,
            irradiance_faces: ibl
                .irradiance_faces
                .iter()
                .map(|face| BufferCacheKey::from_bytes(face))
                .collect(),
            prefilter_base_size: ibl.prefilter_base_size,
            prefilter_mip_levels: ibl.prefilter_mip_levels,
            prefilter_faces: ibl
                .prefilter_faces
                .iter()
                .map(|face| BufferCacheKey::from_bytes(face))
                .collect(),
            brdf_lut_size: ibl.brdf_lut_size,
            brdf_lut: BufferCacheKey::from_bytes(&ibl.brdf_lut),
        }
    }
}

impl AoPhysicalBindGroupKey {
    fn new(mesh: &PreparedMesh) -> Self {
        let physical_layers = match (mesh.physical_maps.as_ref(), mesh.specular_map.as_ref()) {
            (Some(maps), _) => Some(PhysicalLayersTextureCacheKey::from_layers(
                &maps.scalar_map,
                Some(&maps.anisotropy_map),
                Some(&maps.iridescence_map),
            )),
            (None, Some(tex)) => Some(PhysicalLayersTextureCacheKey::from_layers(tex, None, None)),
            (None, None) => None,
        };
        let physical_sheen = match (
            mesh.matcap_map.as_ref(),
            mesh.gradient_map.as_ref(),
            mesh.physical_maps.as_ref(),
        ) {
            (Some(tex), _, _) | (None, Some(tex), _) => Some(TextureCacheKey::from_texture(tex)),
            (None, None, Some(maps)) => Some(TextureCacheKey::from_texture(&maps.sheen_map)),
            (None, None, None) => None,
        };

        Self {
            ao: mesh.ao_map.as_ref().map(TextureCacheKey::from_texture),
            physical_layers,
            physical_sheen,
            physical_specular: mesh
                .physical_maps
                .as_ref()
                .map(|maps| TextureCacheKey::from_texture(&maps.specular_map)),
            clearcoat_normal: mesh
                .clearcoat_normal_map
                .as_ref()
                .map(TextureCacheKey::from_texture),
            alpha: mesh.alpha_map.as_ref().map(TextureCacheKey::from_texture),
            light: mesh.light_map.as_ref().map(TextureCacheKey::from_texture),
            ao_sampler: mesh
                .ao_map
                .as_ref()
                .map(SamplerKey::from_texture)
                .unwrap_or_else(SamplerKey::default_texture),
            alpha_sampler: mesh
                .alpha_map
                .as_ref()
                .map(SamplerKey::from_texture)
                .unwrap_or_else(SamplerKey::default_texture),
            light_sampler: mesh
                .light_map
                .as_ref()
                .map(SamplerKey::from_texture)
                .unwrap_or_else(SamplerKey::default_texture),
            specular_sampler: match (mesh.physical_maps.as_ref(), mesh.specular_map.as_ref()) {
                (None, Some(tex)) => SamplerKey::from_texture(tex),
                _ => SamplerKey::default_texture(),
            },
            physical_layers_sampler: mesh
                .physical_maps
                .as_ref()
                .map(|maps| SamplerKey::from_settings(maps.physical_layers_sampler))
                .unwrap_or_else(SamplerKey::default_texture),
            physical_sheen_sampler: match (
                mesh.matcap_map.as_ref(),
                mesh.gradient_map.as_ref(),
                mesh.physical_maps.as_ref(),
            ) {
                (Some(tex), _, _) | (None, Some(tex), _) => SamplerKey::from_texture(tex),
                (None, None, Some(maps)) => SamplerKey::from_settings(maps.sheen_sampler),
                (None, None, None) => SamplerKey::default_texture(),
            },
            physical_specular_sampler: mesh
                .physical_maps
                .as_ref()
                .map(|maps| SamplerKey::from_settings(maps.specular_sampler))
                .unwrap_or_else(SamplerKey::default_texture),
            clearcoat_normal_sampler: mesh
                .clearcoat_normal_map
                .as_ref()
                .map(SamplerKey::from_texture)
                .unwrap_or_else(SamplerKey::default_texture),
        }
    }
}

fn hash_bytes(bytes: &[u8]) -> u64 {
    let mut hasher = DefaultHasher::new();
    bytes.hash(&mut hasher);
    hasher.finish()
}

impl StatePipelineKey {
    fn new(mesh: &PreparedMesh, sample_count: u32) -> Self {
        Self {
            topology: mesh.topology,
            side: mesh.side,
            sample_count,
            blending: mesh.blending,
            custom_blend: if effective_blend_mode(mesh.blending, mesh.is_transparent)
                == BlendMode::Custom
            {
                mesh.custom_blend.map(CustomBlendPipelineKey::from)
            } else {
                None
            },
            is_transparent: mesh.is_transparent,
            premultiplied_alpha: mesh.premultiplied_alpha,
            depth_test: mesh.depth_test,
            depth_func: mesh.depth_func,
            depth_write: mesh.depth_write,
            color_write: mesh.color_write,
            polygon_offset: mesh.polygon_offset,
            polygon_offset_factor: f32_key(mesh.polygon_offset_factor),
            polygon_offset_units: mesh.polygon_offset_units,
            alpha_to_coverage: mesh.alpha_to_coverage,
            stencil_write: mesh.stencil_write,
            stencil_write_mask: mesh.stencil_write_mask,
            stencil_func: mesh.stencil_func,
            stencil_ref: mesh.stencil_ref,
            stencil_func_mask: mesh.stencil_func_mask,
            stencil_fail: mesh.stencil_fail,
            stencil_z_fail: mesh.stencil_z_fail,
            stencil_z_pass: mesh.stencil_z_pass,
        }
    }
}

impl From<CustomBlendState> for CustomBlendPipelineKey {
    fn from(state: CustomBlendState) -> Self {
        Self {
            color_equation: state.color_equation,
            alpha_equation: state.alpha_equation,
            color_src_factor: state.color_src_factor,
            color_dst_factor: state.color_dst_factor,
            alpha_src_factor: state.alpha_src_factor,
            alpha_dst_factor: state.alpha_dst_factor,
        }
    }
}

fn f32_key(value: f32) -> u32 {
    if value == 0.0 { 0 } else { value.to_bits() }
}

impl SamplerKey {
    fn new(
        wrap_s: WrapMode,
        wrap_t: WrapMode,
        mag_filter: TextureFilter,
        min_filter: TextureFilter,
        mipmap_mode: MipmapFilter,
        anisotropy: u16,
    ) -> Self {
        let anisotropy_clamp =
            if mag_filter == TextureFilter::Linear && min_filter == TextureFilter::Linear {
                anisotropy.clamp(1, 16)
            } else {
                1
            };
        let mipmap_filter = if anisotropy_clamp > 1 {
            MipmapFilter::Linear
        } else {
            mipmap_mode
        };
        let mip_lod_enabled = mipmap_mode != MipmapFilter::None || anisotropy_clamp > 1;
        Self {
            wrap_s,
            wrap_t,
            mag_filter,
            min_filter,
            mipmap_filter,
            mip_lod_enabled,
            anisotropy_clamp,
        }
    }

    fn from_texture(texture: &PreparedTexture) -> Self {
        Self::new(
            texture.wrap_s,
            texture.wrap_t,
            texture.mag_filter,
            texture.min_filter,
            texture.mipmap_filter,
            texture.anisotropy,
        )
    }

    fn from_settings(settings: TextureSamplerSettings) -> Self {
        Self::new(
            settings.wrap_s,
            settings.wrap_t,
            settings.mag_filter,
            settings.min_filter,
            settings.mipmap_filter,
            settings.anisotropy,
        )
    }

    fn default_texture() -> Self {
        Self::new(
            WrapMode::ClampToEdge,
            WrapMode::ClampToEdge,
            TextureFilter::Linear,
            TextureFilter::Linear,
            MipmapFilter::None,
            1,
        )
    }

    fn is_default(self) -> bool {
        self.wrap_s == WrapMode::ClampToEdge
            && self.wrap_t == WrapMode::ClampToEdge
            && self.mag_filter == TextureFilter::Linear
            && self.min_filter == TextureFilter::Linear
            && self.mipmap_filter == MipmapFilter::None
            && !self.mip_lod_enabled
            && self.anisotropy_clamp <= 1
    }

    fn lod_max_clamp(self) -> f32 {
        if self.mip_lod_enabled { 32.0 } else { 0.0 }
    }
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
        let background_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("headless-three-renderer background layout"),
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
                        min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<
                            BackgroundUniforms,
                        >() as u64),
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
        // the default +X anisotropy direction with full strength, and layer 2 is
        // neutral iridescence factor/thickness data.
        let default_physical_layers_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("headless-three-renderer default physical layers map"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 3,
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
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &default_physical_layers_texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x: 0, y: 0, z: 2 },
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

        // Dedicated pipeline layout for the shadow pass. The sparse group indices match the main
        // material pipeline so alpha-tested shadow casters can reuse base/alpha texture bindings.
        let shadow_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("headless-three-renderer shadow pipeline layout"),
                bind_group_layouts: &[
                    Some(&uniform_layout),
                    Some(&texture_layout),
                    Some(&normal_map_layout),
                    Some(&mr_map_layout),
                    Some(&emissive_map_layout),
                    Some(&ibl_layout),
                    Some(&ao_map_layout),
                ],
                immediate_size: 0,
            });
        let make_shadow_pipeline =
            |entry_point: &'static str, topology: wgpu::PrimitiveTopology, label: &'static str| {
                let depth_bias = if topology == wgpu::PrimitiveTopology::TriangleList {
                    // Slight slope-scaled bias to reduce acne (in addition to the
                    // per-fragment bias we apply during shadow sampling).
                    wgpu::DepthBiasState {
                        constant: 2,
                        slope_scale: 2.0,
                        clamp: 0.0,
                    }
                } else {
                    wgpu::DepthBiasState::default()
                };
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
                        topology,
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
                        bias: depth_bias,
                    }),
                    multisample: wgpu::MultisampleState::default(),
                    fragment: Some(wgpu::FragmentState {
                        module: &shader,
                        entry_point: Some("fs_shadow"),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        targets: &[],
                    }),
                    multiview_mask: None,
                    cache: None,
                })
            };
        let shadow_pipelines = [
            make_shadow_pipeline(
                "vs_shadow0",
                wgpu::PrimitiveTopology::TriangleList,
                "headless-three-renderer shadow pipeline 0",
            ),
            make_shadow_pipeline(
                "vs_shadow1",
                wgpu::PrimitiveTopology::TriangleList,
                "headless-three-renderer shadow pipeline 1",
            ),
            make_shadow_pipeline(
                "vs_shadow2",
                wgpu::PrimitiveTopology::TriangleList,
                "headless-three-renderer shadow pipeline 2",
            ),
            make_shadow_pipeline(
                "vs_shadow3",
                wgpu::PrimitiveTopology::TriangleList,
                "headless-three-renderer shadow pipeline 3",
            ),
            make_shadow_pipeline(
                "vs_shadow4",
                wgpu::PrimitiveTopology::TriangleList,
                "headless-three-renderer shadow pipeline 4",
            ),
            make_shadow_pipeline(
                "vs_shadow5",
                wgpu::PrimitiveTopology::TriangleList,
                "headless-three-renderer shadow pipeline 5",
            ),
            make_shadow_pipeline(
                "vs_shadow6",
                wgpu::PrimitiveTopology::TriangleList,
                "headless-three-renderer shadow pipeline 6",
            ),
            make_shadow_pipeline(
                "vs_shadow7",
                wgpu::PrimitiveTopology::TriangleList,
                "headless-three-renderer shadow pipeline 7",
            ),
            make_shadow_pipeline(
                "vs_shadow8",
                wgpu::PrimitiveTopology::TriangleList,
                "headless-three-renderer shadow pipeline 8",
            ),
            make_shadow_pipeline(
                "vs_shadow9",
                wgpu::PrimitiveTopology::TriangleList,
                "headless-three-renderer shadow pipeline 9",
            ),
            make_shadow_pipeline(
                "vs_shadow10",
                wgpu::PrimitiveTopology::TriangleList,
                "headless-three-renderer shadow pipeline 10",
            ),
            make_shadow_pipeline(
                "vs_shadow11",
                wgpu::PrimitiveTopology::TriangleList,
                "headless-three-renderer shadow pipeline 11",
            ),
        ];
        let line_shadow_pipelines = [
            make_shadow_pipeline(
                "vs_shadow0",
                wgpu::PrimitiveTopology::LineList,
                "headless-three-renderer line shadow pipeline 0",
            ),
            make_shadow_pipeline(
                "vs_shadow1",
                wgpu::PrimitiveTopology::LineList,
                "headless-three-renderer line shadow pipeline 1",
            ),
            make_shadow_pipeline(
                "vs_shadow2",
                wgpu::PrimitiveTopology::LineList,
                "headless-three-renderer line shadow pipeline 2",
            ),
            make_shadow_pipeline(
                "vs_shadow3",
                wgpu::PrimitiveTopology::LineList,
                "headless-three-renderer line shadow pipeline 3",
            ),
            make_shadow_pipeline(
                "vs_shadow4",
                wgpu::PrimitiveTopology::LineList,
                "headless-three-renderer line shadow pipeline 4",
            ),
            make_shadow_pipeline(
                "vs_shadow5",
                wgpu::PrimitiveTopology::LineList,
                "headless-three-renderer line shadow pipeline 5",
            ),
            make_shadow_pipeline(
                "vs_shadow6",
                wgpu::PrimitiveTopology::LineList,
                "headless-three-renderer line shadow pipeline 6",
            ),
            make_shadow_pipeline(
                "vs_shadow7",
                wgpu::PrimitiveTopology::LineList,
                "headless-three-renderer line shadow pipeline 7",
            ),
            make_shadow_pipeline(
                "vs_shadow8",
                wgpu::PrimitiveTopology::LineList,
                "headless-three-renderer line shadow pipeline 8",
            ),
            make_shadow_pipeline(
                "vs_shadow9",
                wgpu::PrimitiveTopology::LineList,
                "headless-three-renderer line shadow pipeline 9",
            ),
            make_shadow_pipeline(
                "vs_shadow10",
                wgpu::PrimitiveTopology::LineList,
                "headless-three-renderer line shadow pipeline 10",
            ),
            make_shadow_pipeline(
                "vs_shadow11",
                wgpu::PrimitiveTopology::LineList,
                "headless-three-renderer line shadow pipeline 11",
            ),
        ];

        let vertex_buffers = [Vertex::layout()];
        let make_pipeline = |topology: Topology,
                             side: MeshSide,
                             transparent: bool,
                             sample_count: u32| {
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
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::LessEqual),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: multisample_state(sample_count, false),
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
            make_pipeline(Topology::Triangles, MeshSide::Front, false, 1),
            make_pipeline(Topology::Triangles, MeshSide::Back, false, 1),
            make_pipeline(Topology::Triangles, MeshSide::Double, false, 1),
        ];
        let pipelines_msaa4 = [
            make_pipeline(Topology::Triangles, MeshSide::Front, false, 4),
            make_pipeline(Topology::Triangles, MeshSide::Back, false, 4),
            make_pipeline(Topology::Triangles, MeshSide::Double, false, 4),
        ];
        let transparent_pipelines = [
            make_pipeline(Topology::Triangles, MeshSide::Front, true, 1),
            make_pipeline(Topology::Triangles, MeshSide::Back, true, 1),
            make_pipeline(Topology::Triangles, MeshSide::Double, true, 1),
        ];
        let transparent_pipelines_msaa4 = [
            make_pipeline(Topology::Triangles, MeshSide::Front, true, 4),
            make_pipeline(Topology::Triangles, MeshSide::Back, true, 4),
            make_pipeline(Topology::Triangles, MeshSide::Double, true, 4),
        ];
        let line_pipelines = [
            make_pipeline(Topology::Lines, MeshSide::Front, false, 1),
            make_pipeline(Topology::Lines, MeshSide::Front, true, 1),
        ];
        let line_pipelines_msaa4 = [
            make_pipeline(Topology::Lines, MeshSide::Front, false, 4),
            make_pipeline(Topology::Lines, MeshSide::Front, true, 4),
        ];
        let point_pipelines = [
            make_pipeline(Topology::Points, MeshSide::Front, false, 1),
            make_pipeline(Topology::Points, MeshSide::Front, true, 1),
        ];
        let point_pipelines_msaa4 = [
            make_pipeline(Topology::Points, MeshSide::Front, false, 4),
            make_pipeline(Topology::Points, MeshSide::Front, true, 4),
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
                bind_group_layouts: &[Some(&background_layout)],
                immediate_size: 0,
            });
        let make_background_pipeline = |sample_count: u32, label: &'static str| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
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
                multisample: multisample_state(sample_count, false),
                fragment: Some(wgpu::FragmentState {
                    module: &background_shader,
                    entry_point: Some("fs_background"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &screen_color_targets,
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        let background_pipeline =
            make_background_pipeline(1, "headless-three-renderer background pipeline");
        let background_pipeline_msaa4 =
            make_background_pipeline(4, "headless-three-renderer background pipeline msaa4");

        Ok(Self {
            device,
            queue,
            shader,
            pipelines,
            pipelines_msaa4,
            transparent_pipelines,
            transparent_pipelines_msaa4,
            line_pipelines,
            line_pipelines_msaa4,
            point_pipelines,
            point_pipelines_msaa4,
            pipeline_layout,
            post_layout,
            background_layout,
            post_pipeline,
            background_pipeline,
            background_pipeline_msaa4,
            uniform_layout,
            texture_layout,
            normal_map_layout,
            mr_map_layout,
            emissive_map_layout,
            ibl_layout,
            ao_map_layout,
            shadow_layout,
            shadow_pipelines,
            line_shadow_pipelines,
            sampler,
            sampler_cache: Mutex::new(HashMap::new()),
            texture_cache: Mutex::new(HashMap::new()),
            physical_layers_texture_cache: Mutex::new(HashMap::new()),
            texture_bind_group_cache: Mutex::new(HashMap::new()),
            ao_physical_bind_group_cache: Mutex::new(HashMap::new()),
            background_bind_group_cache: Mutex::new(HashMap::new()),
            ibl_bind_group_cache: Mutex::new(HashMap::new()),
            uniform_bind_group_cache: Mutex::new(HashMap::new()),
            post_uniform_buffer: Mutex::new(None),
            scene_color_texture_cache: Mutex::new(HashMap::new()),
            post_texture_cache: Mutex::new(HashMap::new()),
            readback_buffer_cache: Mutex::new(HashMap::new()),
            mesh_buffer_cache: Mutex::new(HashMap::new()),
            state_pipeline_cache: Mutex::new(HashMap::new()),
            custom_pipeline_cache: Mutex::new(HashMap::new()),
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
        let msaa_color_texture = if settings.sample_count > 1 {
            Some(self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("headless-three-renderer msaa color texture"),
                size: texture_size,
                mip_level_count: 1,
                sample_count: settings.sample_count,
                dimension: wgpu::TextureDimension::D2,
                format: COLOR_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            }))
        } else {
            None
        };
        let msaa_color_view = msaa_color_texture
            .as_ref()
            .map(|texture| texture.create_view(&wgpu::TextureViewDescriptor::default()));
        let render_color_view = msaa_color_view.as_ref().unwrap_or(&color_view);

        let depth_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("headless-three-renderer depth texture"),
            size: texture_size,
            mip_level_count: 1,
            sample_count: settings.sample_count,
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
        let mut scene_color_texture_guard = None;
        let mut post_texture_guard = None;

        let ibl_bind_group = match &settings.ibl {
            Some(ibl) => self.ibl_bind_group_for(ibl),
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

        let (opaque_order, transmissive_order, transparent_order) = partition_draw_order(meshes);

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

        let (output_buffer, readback_buffer_guard) =
            self.cached_readback_buffer(output_buffer_size);

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("headless-three-renderer render encoder"),
            });
        let background_gpu = settings.background_texture.as_ref().map(|background| {
            self.upload_background(background, settings, settings.output_color_space)
        });
        let background_clear = wgpu::Color {
            r: settings.background[0] * f64::from(settings.background_intensity),
            g: settings.background[1] * f64::from(settings.background_intensity),
            b: settings.background[2] * f64::from(settings.background_intensity),
            a: settings.background[3],
        };

        if let Some(background) = &background_gpu {
            let color_attachments = [Some(wgpu::RenderPassColorAttachment {
                view: render_color_view,
                depth_slice: None,
                resolve_target: if settings.sample_count > 1 {
                    Some(&color_view)
                } else {
                    None
                },
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
            pass.set_pipeline(self.background_pipeline_for(settings.sample_count));
            pass.set_bind_group(0, &background.bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        {
            let color_load = match background_gpu.as_ref() {
                Some(_) => wgpu::LoadOp::Load,
                None => wgpu::LoadOp::Clear(background_clear),
            };
            let color_attachments = [Some(wgpu::RenderPassColorAttachment {
                view: render_color_view,
                depth_slice: None,
                resolve_target: if settings.sample_count > 1 {
                    Some(&color_view)
                } else {
                    None
                },
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
                        pass.set_pipeline(self.pipeline_for(key, false, settings.sample_count));
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

        if !transmissive_order.is_empty() || !transparent_order.is_empty() {
            let (scene_color_texture, guard) = self.cached_scratch_texture(
                &self.scene_color_texture_cache,
                texture_size,
                "headless-three-renderer scene color texture",
                wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
            );
            scene_color_texture_guard = Some(guard);
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
                view: render_color_view,
                depth_slice: None,
                resolve_target: if settings.sample_count > 1 {
                    Some(&color_view)
                } else {
                    None
                },
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
            for &i in transmissive_order.iter().chain(transparent_order.iter()) {
                let mesh = &gpu_meshes[i];
                if let Some(pipeline) = &mesh.pipeline_override {
                    pass.set_pipeline(pipeline);
                    current_pipeline = None;
                } else {
                    let key = pipeline_key(mesh);
                    if current_pipeline != Some(key) {
                        pass.set_pipeline(self.pipeline_for(key, true, settings.sample_count));
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

        let mut post_uniform_buffer_guard = None;
        if settings.post_processing.active {
            let (post_texture, guard) = self.cached_scratch_texture(
                &self.post_texture_cache,
                texture_size,
                "headless-three-renderer post color texture",
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            );
            post_texture_guard = Some(guard);
            let post_view = post_texture.create_view(&wgpu::TextureViewDescriptor::default());
            let post_uniforms = post_uniforms(settings.post_processing);
            let mut guard = self
                .post_uniform_buffer
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let post_uniform_buffer = self.write_post_uniform_buffer(&mut guard, &post_uniforms);
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
            post_uniform_buffer_guard = Some(guard);
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
        drop(post_uniform_buffer_guard);
        drop(post_texture_guard);
        drop(scene_color_texture_guard);
        drop(readback_buffer_guard);

        Ok(rgba)
    }

    fn cached_scratch_texture<'a>(
        &self,
        cache: &'a Mutex<HashMap<ScratchTextureKey, wgpu::Texture>>,
        size: wgpu::Extent3d,
        label: &'static str,
        usage: wgpu::TextureUsages,
    ) -> (
        wgpu::Texture,
        MutexGuard<'a, HashMap<ScratchTextureKey, wgpu::Texture>>,
    ) {
        let key = ScratchTextureKey::from_extent(size);
        let mut guard = cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let texture = if let Some(texture) = guard.get(&key) {
            texture.clone()
        } else {
            let texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: COLOR_FORMAT,
                usage,
                view_formats: &[],
            });
            guard.insert(key, texture.clone());
            texture
        };

        (texture, guard)
    }

    fn cached_readback_buffer(
        &self,
        size: u64,
    ) -> (
        wgpu::Buffer,
        MutexGuard<'_, HashMap<ReadbackBufferKey, wgpu::Buffer>>,
    ) {
        let key = ReadbackBufferKey::from_size(size);
        let mut guard = self
            .readback_buffer_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let buffer = if let Some(buffer) = guard.get(&key) {
            buffer.clone()
        } else {
            let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("headless-three-renderer readback buffer"),
                size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            guard.insert(key, buffer.clone());
            buffer
        };

        (buffer, guard)
    }

    /// Render the scene's shadow casters into a shared depth-only texture array
    /// and return a bind group referencing it.
    fn render_shadow_pass(
        &self,
        settings: &RenderSettings,
        gpu_meshes: &[GpuMesh],
    ) -> (wgpu::BindGroup, wgpu::Texture) {
        let shadow_maps = settings
            .shadow
            .as_ref()
            .expect("render_shadow_pass requires configured shadow maps");
        let shadow_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("headless-three-renderer shadow map"),
            size: wgpu::Extent3d {
                width: shadow_maps.map_width,
                height: shadow_maps.map_height,
                depth_or_array_layers: shadow_maps.layer_count,
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
            array_layer_count: Some(shadow_maps.layer_count),
            ..Default::default()
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("headless-three-renderer shadow encoder"),
            });

        for caster in &shadow_maps.casters {
            for local_layer in 0..caster.layer_count {
                let layer = caster.layer_base + local_layer;
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
                for mesh in gpu_meshes.iter() {
                    if !mesh.cast_shadow {
                        continue;
                    }
                    match mesh.topology {
                        Topology::Triangles => {
                            pass.set_pipeline(&self.shadow_pipelines[layer as usize]);
                        }
                        Topology::Lines => {
                            pass.set_pipeline(&self.line_shadow_pipelines[layer as usize]);
                        }
                        Topology::Points => continue,
                    }
                    pass.set_bind_group(0, &mesh.bind_group, &[]);
                    pass.set_bind_group(1, &mesh.texture_bind_group, &[]);
                    pass.set_bind_group(2, &mesh.normal_map_bind_group, &[]);
                    pass.set_bind_group(3, &mesh.mr_map_bind_group, &[]);
                    pass.set_bind_group(4, &mesh.emissive_map_bind_group, &[]);
                    pass.set_bind_group(5, &self.default_ibl_bind_group, &[]);
                    pass.set_bind_group(6, &mesh.ao_map_bind_group, &[]);
                    pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                    if let Some(index_buffer) = &mesh.index_buffer {
                        pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                        pass.draw_indexed(0..mesh.index_count, 0, 0..1);
                    } else {
                        pass.draw(0..mesh.vertex_count, 0..1);
                    }
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
        mipmap_mode: MipmapFilter,
        anisotropy: u16,
    ) -> wgpu::Sampler {
        let key = SamplerKey::new(
            wrap_s,
            wrap_t,
            mag_filter,
            min_filter,
            mipmap_mode,
            anisotropy,
        );
        if key.is_default() {
            return self.sampler.clone();
        }

        if let Some(sampler) = self
            .sampler_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&key)
            .cloned()
        {
            return sampler;
        }

        let sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("headless-three-renderer per-mesh sampler"),
            address_mode_u: key.wrap_s.to_address_mode(),
            address_mode_v: key.wrap_t.to_address_mode(),
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: key.mag_filter.to_filter_mode(),
            min_filter: key.min_filter.to_filter_mode(),
            mipmap_filter: key.mipmap_filter.to_mipmap_filter_mode(),
            lod_max_clamp: key.lod_max_clamp(),
            anisotropy_clamp: key.anisotropy_clamp,
            ..Default::default()
        });

        self.sampler_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .entry(key)
            .or_insert_with(|| sampler.clone())
            .clone()
    }

    fn sampler_for_settings(&self, settings: TextureSamplerSettings) -> wgpu::Sampler {
        self.sampler_for_texture(
            settings.wrap_s,
            settings.wrap_t,
            settings.mag_filter,
            settings.min_filter,
            settings.mipmap_filter,
            settings.anisotropy,
        )
    }

    fn upload_background(
        &self,
        background: &BackgroundTexture,
        settings: &RenderSettings,
        output_color_space: OutputColorSpace,
    ) -> GpuBackground<'_> {
        let gpu_texture = self.upload_texture(
            "headless-three-renderer scene background texture",
            &background.texture,
        );
        let background_flags = if background.is_srgb { 1.0 } else { 0.0 }
            + if output_color_space.is_linear() {
                2.0
            } else {
                0.0
            }
            + if background.mapping == BackgroundTextureMapping::Equirectangular {
                4.0
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
            inverse_view_projection: settings.view_projection.inverse().to_cols_array_2d(),
            camera_params: [
                settings.camera_pos.x,
                settings.camera_pos.y,
                settings.camera_pos.z,
                0.0,
            ],
            rotation1: background.rotation[0],
            rotation2: background.rotation[1],
            rotation3: background.rotation[2],
        };
        let key = BackgroundBindGroupKey::new(&background.texture);
        let mut cache_guard = self
            .background_bind_group_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(cached) = cache_guard.get(&key).cloned() {
            self.queue
                .write_buffer(&cached.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
            return GpuBackground {
                bind_group: cached.bind_group,
                _texture: gpu_texture,
                _uniform_buffer: cached.uniform_buffer,
                _cache_guard: cache_guard,
            };
        }

        let texture_view = gpu_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = self.sampler_for_texture(
            background.texture.wrap_s,
            background.texture.wrap_t,
            background.texture.mag_filter,
            background.texture.min_filter,
            background.texture.mipmap_filter,
            background.texture.anisotropy,
        );
        let uniform_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("headless-three-renderer background uniform buffer"),
                contents: bytemuck::bytes_of(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("headless-three-renderer background bind group"),
            layout: &self.background_layout,
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
        let cached = CachedBackgroundBindGroup {
            uniform_buffer,
            bind_group,
        };
        let cached = cache_guard
            .entry(key)
            .or_insert_with(|| cached.clone())
            .clone();
        GpuBackground {
            bind_group: cached.bind_group,
            _texture: gpu_texture,
            _uniform_buffer: cached.uniform_buffer,
            _cache_guard: cache_guard,
        }
    }

    fn upload_texture(&self, label: &'static str, tex: &PreparedTexture) -> wgpu::Texture {
        let key = TextureCacheKey::from_texture(tex);
        if let Some(texture) = self
            .texture_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&key)
            .cloned()
        {
            return texture;
        }

        let texture = self.upload_texture_uncached(label, tex);
        self.texture_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .entry(key)
            .or_insert_with(|| texture.clone())
            .clone()
    }

    fn upload_texture_uncached(&self, label: &'static str, tex: &PreparedTexture) -> wgpu::Texture {
        let tex_size = wgpu::Extent3d {
            width: tex.width,
            height: tex.height,
            depth_or_array_layers: 1,
        };
        let mip_level_count = texture_upload_mip_level_count(tex);
        let gpu_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: tex_size,
            mip_level_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: COLOR_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.write_prepared_texture_mip_chain(&gpu_texture, 0, tex, mip_level_count);
        gpu_texture
    }

    fn texture_bind_group_for(
        &self,
        kind: TextureBindGroupKind,
        layout: &wgpu::BindGroupLayout,
        texture_label: &'static str,
        bind_group_label: &'static str,
        tex: &PreparedTexture,
    ) -> (wgpu::BindGroup, wgpu::Texture) {
        let gpu_texture = self.upload_texture(texture_label, tex);
        let key = TextureBindGroupKey::new(kind, tex);
        if let Some(bind_group) = self
            .texture_bind_group_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&key)
            .cloned()
        {
            return (bind_group, gpu_texture);
        }

        let tex_view = gpu_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler_for_tex = self.sampler_for_texture(
            tex.wrap_s,
            tex.wrap_t,
            tex.mag_filter,
            tex.min_filter,
            tex.mipmap_filter,
            tex.anisotropy,
        );
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(bind_group_label),
            layout,
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

        let bind_group = self
            .texture_bind_group_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .entry(key)
            .or_insert_with(|| bind_group.clone())
            .clone();
        (bind_group, gpu_texture)
    }

    fn ao_physical_bind_group_for(&self, mesh: &PreparedMesh) -> AoPhysicalBindGroupResources {
        let key = AoPhysicalBindGroupKey::new(mesh);
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
                    Some(&maps.iridescence_map),
                )),
                (None, Some(tex)) => Some(self.upload_physical_layers_texture(
                    "headless-three-renderer specular and physical layers map",
                    tex,
                    None,
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
        let clearcoat_normal_texture = mesh
            .clearcoat_normal_map
            .as_ref()
            .map(|tex| self.upload_texture("headless-three-renderer clearcoat normal map", tex));

        if let Some(bind_group) = self
            .ao_physical_bind_group_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&key)
            .cloned()
        {
            return AoPhysicalBindGroupResources {
                bind_group,
                ao_texture,
                light_texture,
                alpha_texture,
                physical_layers_texture,
                physical_sheen_texture,
                physical_specular_texture,
                clearcoat_normal_texture,
            };
        }

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
                self.sampler_for_texture(
                    tex.wrap_s,
                    tex.wrap_t,
                    tex.mag_filter,
                    tex.min_filter,
                    tex.mipmap_filter,
                    tex.anisotropy,
                )
            })
            .unwrap_or_else(|| self.sampler.clone());
        let alpha_sampler = mesh
            .alpha_map
            .as_ref()
            .map(|tex| {
                self.sampler_for_texture(
                    tex.wrap_s,
                    tex.wrap_t,
                    tex.mag_filter,
                    tex.min_filter,
                    tex.mipmap_filter,
                    tex.anisotropy,
                )
            })
            .unwrap_or_else(|| self.sampler.clone());
        let light_sampler = mesh
            .light_map
            .as_ref()
            .map(|tex| {
                self.sampler_for_texture(
                    tex.wrap_s,
                    tex.wrap_t,
                    tex.mag_filter,
                    tex.min_filter,
                    tex.mipmap_filter,
                    tex.anisotropy,
                )
            })
            .unwrap_or_else(|| self.sampler.clone());
        let specular_sampler = match (mesh.physical_maps.as_ref(), mesh.specular_map.as_ref()) {
            (None, Some(tex)) => self.sampler_for_texture(
                tex.wrap_s,
                tex.wrap_t,
                tex.mag_filter,
                tex.min_filter,
                tex.mipmap_filter,
                tex.anisotropy,
            ),
            _ => self.sampler.clone(),
        };
        let physical_layers_sampler = mesh
            .physical_maps
            .as_ref()
            .map(|maps| self.sampler_for_settings(maps.physical_layers_sampler))
            .unwrap_or_else(|| self.sampler.clone());
        let physical_sheen_sampler = match (
            mesh.matcap_map.as_ref(),
            mesh.gradient_map.as_ref(),
            mesh.physical_maps.as_ref(),
        ) {
            (Some(tex), _, _) | (None, Some(tex), _) => self.sampler_for_texture(
                tex.wrap_s,
                tex.wrap_t,
                tex.mag_filter,
                tex.min_filter,
                tex.mipmap_filter,
                tex.anisotropy,
            ),
            (None, None, Some(maps)) => self.sampler_for_settings(maps.sheen_sampler),
            (None, None, None) => self.sampler.clone(),
        };
        let physical_specular_sampler = mesh
            .physical_maps
            .as_ref()
            .map(|maps| self.sampler_for_settings(maps.specular_sampler))
            .unwrap_or_else(|| self.sampler.clone());
        let clearcoat_normal_sampler = mesh
            .clearcoat_normal_map
            .as_ref()
            .map(|tex| {
                self.sampler_for_texture(
                    tex.wrap_s,
                    tex.wrap_t,
                    tex.mag_filter,
                    tex.min_filter,
                    tex.mipmap_filter,
                    tex.anisotropy,
                )
            })
            .unwrap_or_else(|| self.sampler.clone());

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
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
        let bind_group = self
            .ao_physical_bind_group_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .entry(key)
            .or_insert_with(|| bind_group.clone())
            .clone();

        AoPhysicalBindGroupResources {
            bind_group,
            ao_texture,
            light_texture,
            alpha_texture,
            physical_layers_texture,
            physical_sheen_texture,
            physical_specular_texture,
            clearcoat_normal_texture,
        }
    }

    fn upload_physical_layers_texture(
        &self,
        label: &'static str,
        scalar: &PreparedTexture,
        anisotropy: Option<&PreparedTexture>,
        iridescence: Option<&PreparedTexture>,
    ) -> wgpu::Texture {
        let key = PhysicalLayersTextureCacheKey::from_layers(scalar, anisotropy, iridescence);
        if let Some(texture) = self
            .physical_layers_texture_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&key)
            .cloned()
        {
            return texture;
        }

        let texture =
            self.upload_physical_layers_texture_uncached(label, scalar, anisotropy, iridescence);
        self.physical_layers_texture_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .entry(key)
            .or_insert_with(|| texture.clone())
            .clone()
    }

    fn upload_physical_layers_texture_uncached(
        &self,
        label: &'static str,
        scalar: &PreparedTexture,
        anisotropy: Option<&PreparedTexture>,
        iridescence: Option<&PreparedTexture>,
    ) -> wgpu::Texture {
        let tex_size = wgpu::Extent3d {
            width: scalar.width,
            height: scalar.height,
            depth_or_array_layers: 3,
        };
        let mip_level_count =
            texture_mip_level_count(scalar.width, scalar.height, scalar.mipmap_filter);
        let gpu_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: tex_size,
            mip_level_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: COLOR_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.write_texture_mip_chain(
            &gpu_texture,
            0,
            &scalar.rgba,
            scalar.width,
            scalar.height,
            mip_level_count,
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
        self.write_texture_mip_chain(
            &gpu_texture,
            1,
            anisotropy_rgba,
            scalar.width,
            scalar.height,
            mip_level_count,
        );

        let mut default_iridescence = Vec::new();
        let iridescence_rgba = match iridescence {
            Some(tex) if tex.width == scalar.width && tex.height == scalar.height => {
                tex.rgba.as_slice()
            }
            _ => {
                default_iridescence.reserve_exact((scalar.width * scalar.height * 4) as usize);
                for _ in 0..(scalar.width * scalar.height) {
                    default_iridescence.extend_from_slice(&[255u8, 255, 255, 255]);
                }
                default_iridescence.as_slice()
            }
        };
        self.write_texture_mip_chain(
            &gpu_texture,
            2,
            iridescence_rgba,
            scalar.width,
            scalar.height,
            mip_level_count,
        );
        gpu_texture
    }

    fn write_texture_mip_chain(
        &self,
        gpu_texture: &wgpu::Texture,
        array_layer: u32,
        rgba: &[u8],
        width: u32,
        height: u32,
        mip_level_count: u32,
    ) {
        self.write_texture_mip(gpu_texture, 0, array_layer, rgba, width, height);
        if mip_level_count <= 1 {
            return;
        }

        let mut previous = rgba.to_vec();
        let mut previous_width = width;
        let mut previous_height = height;
        for mip_level in 1..mip_level_count {
            let (next, next_width, next_height) =
                downsample_rgba_mip(&previous, previous_width, previous_height);
            self.write_texture_mip(
                gpu_texture,
                mip_level,
                array_layer,
                &next,
                next_width,
                next_height,
            );
            previous = next;
            previous_width = next_width;
            previous_height = next_height;
        }
    }

    fn write_prepared_texture_mip_chain(
        &self,
        gpu_texture: &wgpu::Texture,
        array_layer: u32,
        texture: &PreparedTexture,
        mip_level_count: u32,
    ) {
        self.write_texture_mip(
            gpu_texture,
            0,
            array_layer,
            &texture.rgba,
            texture.width,
            texture.height,
        );
        if texture.mipmaps.is_empty() {
            if mip_level_count <= 1 {
                return;
            }
            let mut previous = texture.rgba.clone();
            let mut previous_width = texture.width;
            let mut previous_height = texture.height;
            for mip_level in 1..mip_level_count {
                let (next, next_width, next_height) =
                    downsample_rgba_mip(&previous, previous_width, previous_height);
                self.write_texture_mip(
                    gpu_texture,
                    mip_level,
                    array_layer,
                    &next,
                    next_width,
                    next_height,
                );
                previous = next;
                previous_width = next_width;
                previous_height = next_height;
            }
            return;
        }

        for (index, mip) in texture.mipmaps.iter().enumerate() {
            self.write_texture_mip(
                gpu_texture,
                (index + 1) as u32,
                array_layer,
                &mip.rgba,
                mip.width,
                mip.height,
            );
        }
    }

    fn write_texture_mip(
        &self,
        gpu_texture: &wgpu::Texture,
        mip_level: u32,
        array_layer: u32,
        rgba: &[u8],
        width: u32,
        height: u32,
    ) {
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: gpu_texture,
                mip_level,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: 0,
                    z: array_layer,
                },
                aspect: wgpu::TextureAspect::All,
            },
            rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * width),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
    }

    fn create_custom_pipeline(
        &self,
        mesh: &PreparedMesh,
        fragment_body: &str,
        sample_count: u32,
    ) -> Result<wgpu::RenderPipeline> {
        let key = CustomPipelineKey {
            state: StatePipelineKey::new(mesh, sample_count),
            fragment_body: fragment_body.to_owned(),
        };
        if let Some(pipeline) = self
            .custom_pipeline_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&key)
            .cloned()
        {
            return Ok(pipeline);
        }

        let source = custom_shader_source(fragment_body);
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("headless-three-renderer custom material shader"),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
        let pipeline = self.create_material_pipeline(
            &shader,
            mesh,
            sample_count,
            "headless-three-renderer custom material pipeline",
        );

        Ok(self
            .custom_pipeline_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .entry(key)
            .or_insert_with(|| pipeline.clone())
            .clone())
    }

    fn create_state_override_pipeline(
        &self,
        mesh: &PreparedMesh,
        sample_count: u32,
    ) -> wgpu::RenderPipeline {
        let key = StatePipelineKey::new(mesh, sample_count);
        if let Some(pipeline) = self
            .state_pipeline_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&key)
            .cloned()
        {
            return pipeline;
        }

        let pipeline = self.create_material_pipeline(
            &self.shader,
            mesh,
            sample_count,
            "headless-three-renderer material state override pipeline",
        );

        self.state_pipeline_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .entry(key)
            .or_insert_with(|| pipeline.clone())
            .clone()
    }

    fn create_material_pipeline(
        &self,
        shader: &wgpu::ShaderModule,
        mesh: &PreparedMesh,
        sample_count: u32,
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
                    depth_compare: Some(depth_compare_function(mesh)),
                    stencil: stencil_state(mesh),
                    bias: depth_bias_state(mesh),
                }),
                multisample: multisample_state(sample_count, mesh.alpha_to_coverage),
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

    fn mesh_buffers_for(&self, mesh: &PreparedMesh) -> CachedMeshBuffers {
        let key = MeshBufferCacheKey::from_mesh(mesh);
        if let Some(buffers) = self
            .mesh_buffer_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&key)
            .cloned()
        {
            return buffers;
        }

        let buffers = CachedMeshBuffers {
            vertex_buffer: self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("headless-three-renderer vertex buffer"),
                    contents: bytemuck::cast_slice(&mesh.vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                }),
            index_buffer: mesh.indices.as_ref().map(|indices| {
                self.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("headless-three-renderer index buffer"),
                        contents: bytemuck::cast_slice(indices),
                        usage: wgpu::BufferUsages::INDEX,
                    })
            }),
        };

        self.mesh_buffer_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .entry(key)
            .or_insert_with(|| buffers.clone())
            .clone()
    }

    fn uniform_bind_group_for(&self, uniforms: &Uniforms) -> CachedUniformBindGroup {
        let key = UniformBindGroupKey::from_uniforms(uniforms);
        if let Some(cached) = self
            .uniform_bind_group_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&key)
            .cloned()
        {
            return cached;
        }

        let buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("headless-three-renderer uniform buffer"),
                contents: bytemuck::bytes_of(uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("headless-three-renderer bind group"),
            layout: &self.uniform_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });
        let cached = CachedUniformBindGroup { buffer, bind_group };

        self.uniform_bind_group_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .entry(key)
            .or_insert_with(|| cached.clone())
            .clone()
    }

    fn write_post_uniform_buffer(
        &self,
        slot: &mut Option<wgpu::Buffer>,
        uniforms: &PostUniforms,
    ) -> wgpu::Buffer {
        let bytes = bytemuck::bytes_of(uniforms);
        if let Some(buffer) = slot.as_ref() {
            self.queue.write_buffer(buffer, 0, bytes);
            return buffer.clone();
        }

        let buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("headless-three-renderer post uniform buffer"),
                contents: bytes,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        *slot = Some(buffer.clone());
        buffer
    }

    fn ibl_bind_group_for(&self, ibl: &IblMaps) -> wgpu::BindGroup {
        let key = IblBindGroupKey::from_maps(ibl);
        if let Some(bind_group) = self
            .ibl_bind_group_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&key)
            .cloned()
        {
            return bind_group;
        }

        let bind_group = create_ibl_bind_group(
            &self.device,
            &self.queue,
            &self.ibl_layout,
            &self.sampler,
            ibl,
        );
        self.ibl_bind_group_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .entry(key)
            .or_insert_with(|| bind_group.clone())
            .clone()
    }

    fn upload_mesh(&self, settings: &RenderSettings, mesh: &PreparedMesh) -> Result<GpuMesh> {
        let CachedMeshBuffers {
            vertex_buffer,
            index_buffer,
        } = self.mesh_buffers_for(mesh);

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
        let mesh_has_ibl = settings.ibl.is_some() && mesh.use_environment_map.unwrap_or(true);
        let mesh_env_intensity = mesh
            .environment_map_intensity
            .unwrap_or(settings.env_intensity);

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
                    mesh.normal_map_type.normal_mode()
                } else if mesh.bump_map.is_some() {
                    2.0
                } else {
                    0.0
                },
                if mesh_has_ibl { 1.0 } else { 0.0 },
            ],
            ibl_params: [
                mesh_env_intensity,
                mesh.shading_model.as_u32() as f32,
                settings.near,
                settings.far,
            ],
            env_map_params: [
                mesh.environment_map_combine as f32,
                mesh.environment_map_reflectivity,
                if mesh.use_environment_map == Some(true)
                    && matches!(
                        mesh.shading_model,
                        ShadingModel::Basic | ShadingModel::Lambert | ShadingModel::Phong
                    )
                {
                    if mesh.environment_map_refraction {
                        2.0
                    } else {
                        1.0
                    }
                } else {
                    0.0
                },
                mesh.environment_map_refraction_ratio,
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
                if mesh.tone_mapped { 1.0 } else { 0.0 },
                if mesh.alpha_to_coverage && settings.sample_count > 1 {
                    1.0
                } else {
                    0.0
                },
                settings.tone_mapping_exposure,
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
            light_space_matrices: shadow_light_space_matrices(settings),
            shadow_params: shadow_params(settings, mesh.receive_shadow),
            shadow_params2: shadow_params2(settings),
            shadow_params3: shadow_params3(settings),
            shadow_params4: settings
                .shadow
                .as_ref()
                .map(|s| {
                    let radius = s.casters.first().map_or(1.0, |caster| caster.radius);
                    [
                        radius,
                        if mesh.clip_shadows { 1.0 } else { 0.0 },
                        shadow_side_mode(mesh.shadow_side),
                        shadow_alpha_cutoff(mesh),
                    ]
                })
                .unwrap_or([1.0, 0.0, 0.0, 0.0]),
            shadow_infos: shadow_infos(settings),
            shadow_biases: shadow_biases(settings),
            shadow_cascade_splits: shadow_cascade_splits(settings),
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
                } else if mesh.shading_model == ShadingModel::Toon {
                    if mesh.gradient_map_is_srgb { 1.0 } else { 0.0 }
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
                    mesh.dispersion,
                ]
            },
            physical_specular: [
                mesh.physical_specular_color[0],
                mesh.physical_specular_color[1],
                mesh.physical_specular_color[2],
                mesh.physical_specular_intensity,
            ],
            iridescence_params: [
                mesh.iridescence,
                mesh.iridescence_ior,
                mesh.iridescence_thickness_min,
                mesh.iridescence_thickness_max,
            ],
            lights,
        };
        let CachedUniformBindGroup {
            buffer: uniform_buffer,
            bind_group,
        } = self.uniform_bind_group_for(&uniforms);

        let (texture_bind_group, _mesh_texture) = match &mesh.texture {
            Some(tex) => {
                let (bind_group, texture) = self.texture_bind_group_for(
                    TextureBindGroupKind::BaseColor,
                    &self.texture_layout,
                    "headless-three-renderer mesh texture",
                    "headless-three-renderer mesh texture bind group",
                    tex,
                );
                (bind_group, Some(texture))
            }
            None => (self.default_texture_bind_group.clone(), None),
        };

        let (normal_map_bind_group, _normal_map_texture) =
            match mesh.normal_map.as_ref().or(mesh.bump_map.as_ref()) {
                Some(tex) => {
                    let (bind_group, texture) = self.texture_bind_group_for(
                        TextureBindGroupKind::NormalOrBump,
                        &self.normal_map_layout,
                        "headless-three-renderer normal or bump map",
                        "headless-three-renderer normal or bump map bind group",
                        tex,
                    );
                    (bind_group, Some(texture))
                }
                None => (self.default_normal_map_bind_group.clone(), None),
            };

        let (mr_map_bind_group, _mr_map_texture) = match &mesh.metallic_roughness_texture {
            Some(tex) => {
                let (bind_group, texture) = self.texture_bind_group_for(
                    TextureBindGroupKind::MetallicRoughness,
                    &self.mr_map_layout,
                    "headless-three-renderer metallic-roughness map",
                    "headless-three-renderer metallic-roughness bind group",
                    tex,
                );
                (bind_group, Some(texture))
            }
            None => (self.default_mr_map_bind_group.clone(), None),
        };

        let (emissive_map_bind_group, _emissive_map_texture) = match &mesh.emissive_map {
            Some(tex) => {
                let (bind_group, texture) = self.texture_bind_group_for(
                    TextureBindGroupKind::Emissive,
                    &self.emissive_map_layout,
                    "headless-three-renderer emissive map",
                    "headless-three-renderer emissive map bind group",
                    tex,
                );
                (bind_group, Some(texture))
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
            let resources = self.ao_physical_bind_group_for(mesh);
            (
                resources.bind_group,
                resources.ao_texture,
                resources.light_texture,
                resources.alpha_texture,
                resources.physical_layers_texture,
                resources.physical_sheen_texture,
                resources.physical_specular_texture,
                resources.clearcoat_normal_texture,
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
            Some(fragment_body) => {
                Some(self.create_custom_pipeline(mesh, fragment_body, settings.sample_count)?)
            }
            None if requires_pipeline_override(mesh, settings.sample_count) => {
                Some(self.create_state_override_pipeline(mesh, settings.sample_count))
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

fn texture_mip_level_count(width: u32, height: u32, mipmap_filter: MipmapFilter) -> u32 {
    if mipmap_filter == MipmapFilter::None {
        return 1;
    }

    let mut levels = 1;
    let mut mip_width = width.max(1);
    let mut mip_height = height.max(1);
    while mip_width > 1 || mip_height > 1 {
        mip_width = (mip_width / 2).max(1);
        mip_height = (mip_height / 2).max(1);
        levels += 1;
    }
    levels
}

fn texture_upload_mip_level_count(texture: &PreparedTexture) -> u32 {
    if texture.mipmaps.is_empty() {
        texture_mip_level_count(texture.width, texture.height, texture.mipmap_filter)
    } else {
        1 + texture.mipmaps.len() as u32
    }
}

fn downsample_rgba_mip(source: &[u8], width: u32, height: u32) -> (Vec<u8>, u32, u32) {
    let next_width = (width / 2).max(1);
    let next_height = (height / 2).max(1);
    let mut output = vec![0u8; (next_width * next_height * 4) as usize];

    for y in 0..next_height {
        let source_y0 = y * height / next_height;
        let source_y1 = ((y + 1) * height / next_height).max(source_y0 + 1);
        for x in 0..next_width {
            let source_x0 = x * width / next_width;
            let source_x1 = ((x + 1) * width / next_width).max(source_x0 + 1);
            let mut sum = [0u32; 4];
            let mut count = 0u32;
            for source_y in source_y0..source_y1.min(height) {
                for source_x in source_x0..source_x1.min(width) {
                    let source_index = ((source_y * width + source_x) * 4) as usize;
                    sum[0] += source[source_index] as u32;
                    sum[1] += source[source_index + 1] as u32;
                    sum[2] += source[source_index + 2] as u32;
                    sum[3] += source[source_index + 3] as u32;
                    count += 1;
                }
            }
            let output_index = ((y * next_width + x) * 4) as usize;
            output[output_index] = (sum[0] / count) as u8;
            output[output_index + 1] = (sum[1] / count) as u8;
            output[output_index + 2] = (sum[2] / count) as u8;
            output[output_index + 3] = (sum[3] / count) as u8;
        }
    }

    (output, next_width, next_height)
}

fn side_index(side: MeshSide) -> usize {
    match side {
        MeshSide::Front => 0,
        MeshSide::Back => 1,
        MeshSide::Double => 2,
    }
}

fn shadow_kind_mode(kind: ShadowKind) -> f32 {
    match kind {
        ShadowKind::DirectionalOrSpot => 0.0,
        ShadowKind::Point => 1.0,
        ShadowKind::Cascaded => 2.0,
    }
}

fn shadow_light_space_matrices(settings: &RenderSettings) -> [[[f32; 4]; 4]; MAX_SHADOW_LAYERS] {
    let mut matrices = [Mat4::IDENTITY.to_cols_array_2d(); MAX_SHADOW_LAYERS];
    if let Some(shadow_maps) = &settings.shadow {
        for caster in &shadow_maps.casters {
            for local_layer in 0..caster.layer_count as usize {
                let global_layer = caster.layer_base as usize + local_layer;
                matrices[global_layer] = caster.light_vps[local_layer].to_cols_array_2d();
            }
        }
    }
    matrices
}

fn shadow_params(settings: &RenderSettings, receive_shadow: bool) -> [f32; 4] {
    let Some(shadow_maps) = &settings.shadow else {
        return [0.0; 4];
    };
    let Some(first) = shadow_maps.casters.first() else {
        return [0.0; 4];
    };
    [
        shadow_maps.casters.len() as f32,
        first.bias,
        first.normal_bias,
        if receive_shadow { 1.0 } else { 0.0 },
    ]
}

fn shadow_params2(settings: &RenderSettings) -> [f32; 4] {
    let Some(shadow_maps) = &settings.shadow else {
        return [0.0; 4];
    };
    let Some(first) = shadow_maps.casters.first() else {
        return [0.0; 4];
    };
    [
        first.light_index as f32,
        1.0 / shadow_maps.map_width as f32,
        1.0 / shadow_maps.map_height as f32,
        shadow_kind_mode(first.kind),
    ]
}

fn shadow_params3(settings: &RenderSettings) -> [f32; 4] {
    let Some(shadow_maps) = &settings.shadow else {
        return [f32::MAX; 4];
    };
    let Some(first) = shadow_maps.casters.first() else {
        return [f32::MAX; 4];
    };
    [
        first.cascade_splits[0],
        first.cascade_splits[1],
        first.cascade_splits[2],
        first.layer_count as f32,
    ]
}

fn shadow_infos(settings: &RenderSettings) -> [[f32; 4]; MAX_SHADOW_LAYERS] {
    let mut infos = [[0.0; 4]; MAX_SHADOW_LAYERS];
    if let Some(shadow_maps) = &settings.shadow {
        for (slot, caster) in shadow_maps.casters.iter().enumerate() {
            infos[slot] = [
                caster.light_index as f32,
                caster.layer_base as f32,
                caster.layer_count as f32,
                shadow_kind_mode(caster.kind),
            ];
        }
    }
    infos
}

fn shadow_biases(settings: &RenderSettings) -> [[f32; 4]; MAX_SHADOW_LAYERS] {
    let mut biases = [[0.0; 4]; MAX_SHADOW_LAYERS];
    if let Some(shadow_maps) = &settings.shadow {
        for (slot, caster) in shadow_maps.casters.iter().enumerate() {
            biases[slot] = [caster.bias, caster.normal_bias, caster.radius, 0.0];
        }
    }
    biases
}

fn shadow_cascade_splits(settings: &RenderSettings) -> [[f32; 4]; MAX_SHADOW_LAYERS] {
    let mut splits = [[f32::MAX; 4]; MAX_SHADOW_LAYERS];
    if let Some(shadow_maps) = &settings.shadow {
        for (slot, caster) in shadow_maps.casters.iter().enumerate() {
            splits[slot] = [
                caster.cascade_splits[0],
                caster.cascade_splits[1],
                caster.cascade_splits[2],
                0.0,
            ];
        }
    }
    splits
}

fn shadow_side_mode(side: MeshSide) -> f32 {
    match side {
        MeshSide::Double => 0.0,
        MeshSide::Front => 1.0,
        MeshSide::Back => 2.0,
    }
}

fn shadow_alpha_cutoff(mesh: &PreparedMesh) -> f32 {
    if mesh.alpha_to_coverage {
        0.5_f32.max(mesh.alpha_test)
    } else {
        0.0
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

fn multisample_state(sample_count: u32, alpha_to_coverage: bool) -> wgpu::MultisampleState {
    wgpu::MultisampleState {
        count: sample_count,
        mask: !0,
        alpha_to_coverage_enabled: alpha_to_coverage && sample_count > 1,
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

fn requires_pipeline_override(mesh: &PreparedMesh, sample_count: u32) -> bool {
    let default_blending = if mesh.is_transparent {
        BlendMode::Normal
    } else {
        BlendMode::None
    };
    !mesh.depth_test
        || mesh.depth_func != StencilCompare::LessEqual
        || !mesh.depth_write
        || !mesh.color_write
        || mesh.polygon_offset
        || mesh.stencil_write
        || (mesh.alpha_to_coverage && sample_count > 1)
        || (mesh.premultiplied_alpha
            && effective_blend_mode(mesh.blending, mesh.is_transparent) != BlendMode::None)
        || effective_blend_mode(mesh.blending, mesh.is_transparent) != default_blending
}

fn depth_compare_function(mesh: &PreparedMesh) -> wgpu::CompareFunction {
    if mesh.depth_test {
        stencil_compare(mesh.depth_func)
    } else {
        wgpu::CompareFunction::Always
    }
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
    fn background_pipeline_for(&self, sample_count: u32) -> &wgpu::RenderPipeline {
        if sample_count == 4 {
            &self.background_pipeline_msaa4
        } else {
            &self.background_pipeline
        }
    }

    fn pipeline_for(
        &self,
        key: PipelineKey,
        transparent: bool,
        sample_count: u32,
    ) -> &wgpu::RenderPipeline {
        let msaa4 = sample_count == 4;
        match key {
            PipelineKey::Tri(side) => {
                let idx = side_index(side);
                if transparent && msaa4 {
                    &self.transparent_pipelines_msaa4[idx]
                } else if transparent {
                    &self.transparent_pipelines[idx]
                } else if msaa4 {
                    &self.pipelines_msaa4[idx]
                } else {
                    &self.pipelines[idx]
                }
            }
            PipelineKey::Line if msaa4 => {
                &self.line_pipelines_msaa4[if transparent { 1 } else { 0 }]
            }
            PipelineKey::Line => &self.line_pipelines[if transparent { 1 } else { 0 }],
            PipelineKey::Point if msaa4 => {
                &self.point_pipelines_msaa4[if transparent { 1 } else { 0 }]
            }
            PipelineKey::Point => &self.point_pipelines[if transparent { 1 } else { 0 }],
        }
    }
}

fn partition_draw_order(meshes: &[PreparedMesh]) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    let mut opaque = Vec::new();
    let mut transmissive = Vec::new();
    let mut transparent = Vec::new();

    for (i, mesh) in meshes.iter().enumerate() {
        if mesh.transmission > 0.0001 {
            transmissive.push(i);
        } else if mesh.is_transparent {
            transparent.push(i);
        } else {
            opaque.push(i);
        }
    }

    opaque.sort_by(|&a, &b| compare_opaque_meshes(&meshes[a], &meshes[b]));

    // Sort transparent meshes back-to-front (farthest first)
    transmissive.sort_by(|&a, &b| compare_transparent_meshes(&meshes[a], &meshes[b]));
    transparent.sort_by(|&a, &b| compare_transparent_meshes(&meshes[a], &meshes[b]));

    (opaque, transmissive, transparent)
}

fn compare_opaque_meshes(a: &PreparedMesh, b: &PreparedMesh) -> std::cmp::Ordering {
    compare_f32(a.group_order, b.group_order)
        .then_with(|| compare_f32(a.render_order, b.render_order))
        .then_with(|| a.material_sort_key.cmp(&b.material_sort_key))
        .then_with(|| a.material_variant.cmp(&b.material_variant))
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
    rows[7][3] = if mesh.ao_map_uses_uv2 { 1.0 } else { 0.0 };
    rows[8][3] = if mesh.light_map_is_srgb { 1.0 } else { 0.0 };
    rows[9][3] = if mesh.light_map_uses_uv2 { 1.0 } else { 0.0 };
    rows[11][3] = if mesh.specular_map_uses_uv2 { 1.0 } else { 0.0 };
    rows
}

fn physical_map_transform_rows(mesh: &PreparedMesh) -> [[f32; 4]; 24] {
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
        mesh.iridescence_map_transform,
        mesh.iridescence_thickness_map_transform,
    ];
    let mut rows = [[0.0; 4]; 24];
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
    rows[21][3] = if mesh.iridescence_map_uses_uv2 {
        1.0
    } else {
        0.0
    };
    rows[23][3] = if mesh.iridescence_thickness_map_uses_uv2 {
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

#[cfg(test)]
mod tests {
    use super::{
        AoPhysicalBindGroupKey, BackgroundBindGroupKey, BackgroundUniforms, CustomBlendPipelineKey,
        IblBindGroupKey, MeshBufferCacheKey, PhysicalLayersTextureCacheKey, PostUniforms,
        SamplerKey, TextureBindGroupKey, TextureBindGroupKind, TextureCacheKey,
        UniformBindGroupKey, Uniforms, downsample_rgba_mip, f32_key, post_uniforms,
        texture_mip_level_count,
    };
    use crate::ibl::IblMaps;
    use crate::mesh::{
        BlendEquation, BlendFactor, CustomBlendState, MipmapFilter, PreparedTexture,
        PreparedTextureMipLevel, TextureFilter, WrapMode,
    };
    use crate::settings::PostProcessingSettings;
    use bytemuck::Zeroable;

    fn single_pixel_texture(rgba: [u8; 4]) -> PreparedTexture {
        solid_texture(1, 1, rgba)
    }

    fn solid_texture(width: u32, height: u32, rgba: [u8; 4]) -> PreparedTexture {
        let mut data = Vec::with_capacity((width * height * 4) as usize);
        for _ in 0..(width * height) {
            data.extend_from_slice(&rgba);
        }
        PreparedTexture {
            rgba: data,
            width,
            height,
            mipmaps: Vec::new(),
            wrap_s: WrapMode::ClampToEdge,
            wrap_t: WrapMode::ClampToEdge,
            mag_filter: TextureFilter::Linear,
            min_filter: TextureFilter::Linear,
            mipmap_filter: MipmapFilter::None,
            anisotropy: 1,
        }
    }

    fn single_pixel_ibl_maps(red: u8) -> IblMaps {
        let face = vec![red, 0, 0, 255];
        IblMaps {
            irradiance_faces: vec![face.clone(); 6],
            irradiance_size: 1,
            prefilter_faces: vec![face; 6],
            prefilter_base_size: 1,
            prefilter_mip_levels: 1,
            brdf_lut: vec![0, red, 0, 255],
            brdf_lut_size: 1,
        }
    }

    #[test]
    fn mip_level_count_tracks_min_filter_mode() {
        assert_eq!(
            texture_mip_level_count(8, 4, MipmapFilter::None),
            1,
            "non-mipmap filters keep a single uploaded level",
        );
        assert_eq!(texture_mip_level_count(8, 4, MipmapFilter::Nearest), 4);
        assert_eq!(texture_mip_level_count(3, 5, MipmapFilter::Linear), 3);
        assert_eq!(texture_mip_level_count(1, 1, MipmapFilter::Linear), 1);
    }

    #[test]
    fn downsample_rgba_mip_averages_source_regions() {
        let rgba = vec![
            0, 0, 0, 255, 10, 20, 30, 255, 200, 0, 0, 255, 210, 20, 30, 255,
        ];
        let (mip, width, height) = downsample_rgba_mip(&rgba, 4, 1);
        assert_eq!((width, height), (2, 1));
        assert_eq!(
            mip,
            vec![5, 10, 15, 255, 205, 10, 15, 255],
            "each output texel averages its covered source span",
        );
    }

    #[test]
    fn downsample_rgba_mip_covers_odd_dimensions() {
        let rgba = vec![
            0, 0, 0, 255, 60, 0, 0, 255, 120, 0, 0, 255, 180, 0, 0, 255, 240, 0, 0, 255,
        ];
        let (mip, width, height) = downsample_rgba_mip(&rgba, 5, 1);
        assert_eq!((width, height), (2, 1));
        assert_eq!(mip, vec![30, 0, 0, 255, 180, 0, 0, 255]);
    }

    #[test]
    fn sampler_key_normalizes_effective_sampler_descriptor() {
        let default = SamplerKey::new(
            WrapMode::ClampToEdge,
            WrapMode::ClampToEdge,
            TextureFilter::Linear,
            TextureFilter::Linear,
            MipmapFilter::None,
            1,
        );
        assert!(default.is_default());
        assert_eq!(default.lod_max_clamp(), 0.0);

        let anisotropic = SamplerKey::new(
            WrapMode::Repeat,
            WrapMode::MirrorRepeat,
            TextureFilter::Linear,
            TextureFilter::Linear,
            MipmapFilter::None,
            32,
        );
        assert_eq!(anisotropic.anisotropy_clamp, 16);
        assert_eq!(anisotropic.mipmap_filter, MipmapFilter::Linear);
        assert_eq!(anisotropic.lod_max_clamp(), 32.0);

        let nearest = SamplerKey::new(
            WrapMode::Repeat,
            WrapMode::ClampToEdge,
            TextureFilter::Nearest,
            TextureFilter::Linear,
            MipmapFilter::Linear,
            8,
        );
        assert_eq!(nearest.anisotropy_clamp, 1);
        assert_eq!(nearest.mipmap_filter, MipmapFilter::Linear);
        assert_eq!(nearest.lod_max_clamp(), 32.0);
    }

    #[test]
    fn texture_cache_keys_track_pixels_and_mip_generation() {
        let base = single_pixel_texture([255, 0, 0, 255]);
        let mut sampler_variant = single_pixel_texture([255, 0, 0, 255]);
        sampler_variant.wrap_s = WrapMode::Repeat;
        sampler_variant.mag_filter = TextureFilter::Nearest;

        assert_eq!(
            TextureCacheKey::from_texture(&base),
            TextureCacheKey::from_texture(&sampler_variant),
            "sampler-only changes reuse the same uploaded texture",
        );

        let mut mipmapped = single_pixel_texture([255, 0, 0, 255]);
        mipmapped.mipmap_filter = MipmapFilter::Linear;
        assert_ne!(
            TextureCacheKey::from_texture(&base),
            TextureCacheKey::from_texture(&mipmapped),
            "generated mip-chain textures need their own cache entry",
        );

        let different_pixels = single_pixel_texture([0, 255, 0, 255]);
        assert_ne!(
            TextureCacheKey::from_texture(&base),
            TextureCacheKey::from_texture(&different_pixels),
        );
    }

    #[test]
    fn texture_cache_keys_include_explicit_mipmaps() {
        let mut first = PreparedTexture {
            rgba: vec![
                255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255,
            ],
            width: 2,
            height: 2,
            mipmaps: vec![PreparedTextureMipLevel {
                rgba: vec![128, 128, 128, 255],
                width: 1,
                height: 1,
            }],
            wrap_s: WrapMode::ClampToEdge,
            wrap_t: WrapMode::ClampToEdge,
            mag_filter: TextureFilter::Linear,
            min_filter: TextureFilter::Linear,
            mipmap_filter: MipmapFilter::Linear,
            anisotropy: 1,
        };
        let second = PreparedTexture {
            rgba: vec![
                255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255,
            ],
            width: 2,
            height: 2,
            mipmaps: vec![PreparedTextureMipLevel {
                rgba: vec![64, 64, 64, 255],
                width: 1,
                height: 1,
            }],
            wrap_s: WrapMode::ClampToEdge,
            wrap_t: WrapMode::ClampToEdge,
            mag_filter: TextureFilter::Linear,
            min_filter: TextureFilter::Linear,
            mipmap_filter: MipmapFilter::Linear,
            anisotropy: 1,
        };
        assert_ne!(
            TextureCacheKey::from_texture(&first),
            TextureCacheKey::from_texture(&second),
        );

        first.mipmaps.clear();
        assert_ne!(
            TextureCacheKey::from_texture(&first),
            TextureCacheKey::from_texture(&second),
            "explicit mipmaps are distinct from generated mip-chain uploads",
        );
    }

    #[test]
    fn mesh_buffer_cache_keys_track_vertex_and_index_bytes() {
        let base = MeshBufferCacheKey::from_bytes(
            &[1, 2, 3, 4, 5, 6, 7, 8],
            Some(&[0, 0, 0, 0, 1, 0, 0, 0]),
        );
        assert_eq!(
            base,
            MeshBufferCacheKey::from_bytes(
                &[1, 2, 3, 4, 5, 6, 7, 8],
                Some(&[0, 0, 0, 0, 1, 0, 0, 0]),
            ),
        );
        assert_ne!(
            base,
            MeshBufferCacheKey::from_bytes(
                &[8, 7, 6, 5, 4, 3, 2, 1],
                Some(&[0, 0, 0, 0, 1, 0, 0, 0]),
            ),
            "vertex data changes need a distinct buffer entry",
        );
        assert_ne!(
            base,
            MeshBufferCacheKey::from_bytes(&[1, 2, 3, 4, 5, 6, 7, 8], None),
            "indexed and non-indexed geometry cannot share a buffer entry",
        );
        assert_ne!(
            base,
            MeshBufferCacheKey::from_bytes(
                &[1, 2, 3, 4, 5, 6, 7, 8],
                Some(&[1, 0, 0, 0, 0, 0, 0, 0]),
            ),
            "index data changes need a distinct buffer entry",
        );
    }

    #[test]
    fn physical_layers_cache_keys_track_effective_layer_uploads() {
        let scalar = solid_texture(2, 2, [255, 0, 0, 255]);
        let anisotropy = solid_texture(2, 2, [0, 128, 255, 255]);
        let iridescence = solid_texture(2, 2, [255, 255, 128, 255]);
        let mismatched_anisotropy = single_pixel_texture([0, 255, 0, 255]);

        assert_eq!(
            PhysicalLayersTextureCacheKey::from_layers(&scalar, None, None),
            PhysicalLayersTextureCacheKey::from_layers(&scalar, Some(&mismatched_anisotropy), None),
            "mismatched optional layers use generated defaults during upload",
        );
        assert_ne!(
            PhysicalLayersTextureCacheKey::from_layers(&scalar, None, None),
            PhysicalLayersTextureCacheKey::from_layers(&scalar, Some(&anisotropy), None),
        );
        assert_ne!(
            PhysicalLayersTextureCacheKey::from_layers(&scalar, Some(&anisotropy), None),
            PhysicalLayersTextureCacheKey::from_layers(
                &scalar,
                Some(&anisotropy),
                Some(&iridescence),
            ),
        );
    }

    #[test]
    fn texture_bind_group_keys_track_slot_texture_and_sampler() {
        let base = single_pixel_texture([255, 0, 0, 255]);
        let same = single_pixel_texture([255, 0, 0, 255]);
        assert_eq!(
            TextureBindGroupKey::new(TextureBindGroupKind::BaseColor, &base),
            TextureBindGroupKey::new(TextureBindGroupKind::BaseColor, &same),
        );
        assert_ne!(
            TextureBindGroupKey::new(TextureBindGroupKind::BaseColor, &base),
            TextureBindGroupKey::new(TextureBindGroupKind::NormalOrBump, &same),
            "different bind group layouts cannot share entries",
        );

        let different_texture = single_pixel_texture([0, 255, 0, 255]);
        assert_ne!(
            TextureBindGroupKey::new(TextureBindGroupKind::BaseColor, &base),
            TextureBindGroupKey::new(TextureBindGroupKind::BaseColor, &different_texture),
        );

        let mut repeat_sampler = single_pixel_texture([255, 0, 0, 255]);
        repeat_sampler.wrap_s = WrapMode::Repeat;
        assert_ne!(
            TextureBindGroupKey::new(TextureBindGroupKind::BaseColor, &base),
            TextureBindGroupKey::new(TextureBindGroupKind::BaseColor, &repeat_sampler),
            "sampler state is part of the bind group resource set",
        );
    }

    #[test]
    fn background_bind_group_keys_track_texture_and_sampler() {
        let base = single_pixel_texture([255, 0, 0, 255]);
        let same = single_pixel_texture([255, 0, 0, 255]);
        let uniforms = BackgroundUniforms::zeroed();
        assert_eq!(
            BackgroundBindGroupKey::new(&base),
            BackgroundBindGroupKey::new(&same)
        );

        let different_texture = single_pixel_texture([0, 255, 0, 255]);
        assert_ne!(
            BackgroundBindGroupKey::new(&base),
            BackgroundBindGroupKey::new(&different_texture),
        );

        let mut repeated_texture = single_pixel_texture([255, 0, 0, 255]);
        repeated_texture.wrap_t = WrapMode::Repeat;
        assert_ne!(
            BackgroundBindGroupKey::new(&base),
            BackgroundBindGroupKey::new(&repeated_texture),
            "background sampler state is part of the cached bind group",
        );

        let mut changed_uniforms = uniforms;
        changed_uniforms.camera_params[0] = 1.0;
        assert_ne!(
            bytemuck::bytes_of(&uniforms),
            bytemuck::bytes_of(&changed_uniforms),
        );
        assert_eq!(
            BackgroundBindGroupKey::new(&base),
            BackgroundBindGroupKey::new(&base),
            "camera and background uniform content update the cached buffer instead of splitting bind groups",
        );
    }

    fn ao_physical_key(ao: Option<&PreparedTexture>) -> AoPhysicalBindGroupKey {
        let default_sampler = SamplerKey::default_texture();
        AoPhysicalBindGroupKey {
            ao: ao.map(TextureCacheKey::from_texture),
            physical_layers: None,
            physical_sheen: None,
            physical_specular: None,
            clearcoat_normal: None,
            alpha: None,
            light: None,
            ao_sampler: ao.map(SamplerKey::from_texture).unwrap_or(default_sampler),
            alpha_sampler: default_sampler,
            light_sampler: default_sampler,
            specular_sampler: default_sampler,
            physical_layers_sampler: default_sampler,
            physical_sheen_sampler: default_sampler,
            physical_specular_sampler: default_sampler,
            clearcoat_normal_sampler: default_sampler,
        }
    }

    #[test]
    fn ao_physical_bind_group_keys_track_resources_and_samplers() {
        let ao = single_pixel_texture([255, 0, 0, 255]);
        let same_ao = single_pixel_texture([255, 0, 0, 255]);
        assert_eq!(ao_physical_key(Some(&ao)), ao_physical_key(Some(&same_ao)));

        let different_ao = single_pixel_texture([0, 255, 0, 255]);
        assert_ne!(
            ao_physical_key(Some(&ao)),
            ao_physical_key(Some(&different_ao)),
        );

        let mut repeated_ao = single_pixel_texture([255, 0, 0, 255]);
        repeated_ao.wrap_s = WrapMode::Repeat;
        assert_ne!(
            ao_physical_key(Some(&ao)),
            ao_physical_key(Some(&repeated_ao)),
            "AO sampler state is part of the combined bind group",
        );
    }

    #[test]
    fn uniform_bind_group_keys_track_uniform_bytes() {
        let first = Uniforms::zeroed();
        let second = Uniforms::zeroed();
        assert_eq!(
            UniformBindGroupKey::from_uniforms(&first),
            UniformBindGroupKey::from_uniforms(&second),
        );

        let mut changed = first;
        changed.base_color[0] = 1.0;
        assert_ne!(
            UniformBindGroupKey::from_uniforms(&first),
            UniformBindGroupKey::from_uniforms(&changed),
        );
    }

    #[test]
    fn post_uniforms_match_reusable_buffer_layout() {
        let uniforms = post_uniforms(PostProcessingSettings {
            active: true,
            exposure: 1.25,
            contrast: 2.0,
            saturation: 0.5,
            vignette: 0.25,
            grayscale: 0.75,
            invert: 1.0,
        });

        assert_eq!(std::mem::size_of::<PostUniforms>(), 32);
        assert_eq!(uniforms.params1, [1.25, 2.0, 0.5, 0.25]);
        assert_eq!(uniforms.params2, [0.75, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn ibl_bind_group_keys_track_uploaded_map_bytes() {
        let base = single_pixel_ibl_maps(32);
        let same = single_pixel_ibl_maps(32);
        assert_eq!(
            IblBindGroupKey::from_maps(&base),
            IblBindGroupKey::from_maps(&same),
        );

        let different_face = single_pixel_ibl_maps(64);
        assert_ne!(
            IblBindGroupKey::from_maps(&base),
            IblBindGroupKey::from_maps(&different_face),
            "IBL face bytes are part of the uploaded resource cache key",
        );

        let mut different_brdf = single_pixel_ibl_maps(32);
        different_brdf.brdf_lut[1] = 96;
        assert_ne!(
            IblBindGroupKey::from_maps(&base),
            IblBindGroupKey::from_maps(&different_brdf),
            "BRDF LUT bytes are part of the uploaded resource cache key",
        );

        let mut different_mips = single_pixel_ibl_maps(32);
        different_mips.prefilter_mip_levels = 2;
        assert_ne!(
            IblBindGroupKey::from_maps(&base),
            IblBindGroupKey::from_maps(&different_mips),
            "prefilter dimensions are part of the uploaded resource cache key",
        );
    }

    #[test]
    fn state_pipeline_keys_ignore_dynamic_blend_constants() {
        let first = CustomBlendState {
            color_equation: BlendEquation::Add,
            alpha_equation: BlendEquation::Max,
            color_src_factor: BlendFactor::ConstantColor,
            color_dst_factor: BlendFactor::OneMinusSrcAlpha,
            alpha_src_factor: BlendFactor::One,
            alpha_dst_factor: BlendFactor::OneMinusConstantAlpha,
            constant: [0.1, 0.2, 0.3, 0.4],
        };
        let second = CustomBlendState {
            constant: [0.9, 0.8, 0.7, 0.6],
            ..first
        };

        assert_eq!(
            CustomBlendPipelineKey::from(first),
            CustomBlendPipelineKey::from(second),
            "blend constants are set on the render pass, not baked into the pipeline",
        );
    }

    #[test]
    fn state_pipeline_float_keys_normalize_signed_zero() {
        assert_eq!(f32_key(0.0), f32_key(-0.0));
        assert_ne!(f32_key(0.0), f32_key(0.25));
    }
}
