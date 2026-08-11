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

const MAX_SAMPLER_CACHE_ENTRIES: usize = 64;
const MAX_TEXTURE_CACHE_ENTRIES: usize = 512;
const MAX_PHYSICAL_LAYERS_TEXTURE_CACHE_ENTRIES: usize = 256;
const MAX_TEXTURE_BIND_GROUP_CACHE_ENTRIES: usize = 512;
const MAX_AO_PHYSICAL_BIND_GROUP_CACHE_ENTRIES: usize = 256;
const MAX_BACKGROUND_BIND_GROUP_CACHE_ENTRIES: usize = 64;
const MAX_IBL_BIND_GROUP_CACHE_ENTRIES: usize = 32;
const MAX_MESH_BUFFER_CACHE_ENTRIES: usize = 2048;
const MAX_STATE_PIPELINE_CACHE_ENTRIES: usize = 128;
const MAX_CUSTOM_PIPELINE_CACHE_ENTRIES: usize = 128;
const MAX_SCRATCH_TEXTURE_CACHE_ENTRIES: usize = 32;
const MAX_POST_BIND_GROUP_CACHE_ENTRIES: usize = 32;
const MAX_READBACK_BUFFER_CACHE_ENTRIES: usize = 32;
const MAX_NATIVE_MESH_BUFFER_CACHE_ENTRIES: usize = 2048;

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
    dynamic_uniform_bind_group_cache: Mutex<Vec<CachedDynamicUniformBindGroup>>,
    post_uniform_buffer: Mutex<Option<wgpu::Buffer>>,
    color_texture_cache: Mutex<HashMap<ScratchTextureKey, wgpu::Texture>>,
    post_bind_group_cache: Mutex<HashMap<ScratchTextureKey, CachedPostBindGroup>>,
    scene_color_texture_cache: Mutex<HashMap<ScratchTextureKey, wgpu::Texture>>,
    post_texture_cache: Mutex<HashMap<ScratchTextureKey, wgpu::Texture>>,
    readback_buffer_cache: Mutex<HashMap<ReadbackBufferKey, wgpu::Buffer>>,
    mesh_buffer_cache: Mutex<HashMap<MeshBufferCacheKey, CachedMeshBuffers>>,
    native_mesh_buffer_cache: Mutex<HashMap<u32, CachedMeshBuffers>>,
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
struct CachedDynamicUniformBindGroup {
    key: UniformBindGroupKey,
    buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

impl CachedDynamicUniformBindGroup {
    fn as_uniform_bind_group(&self) -> CachedUniformBindGroup {
        CachedUniformBindGroup {
            buffer: self.buffer.clone(),
            bind_group: self.bind_group.clone(),
        }
    }
}

#[derive(Clone)]
struct CachedBackgroundBindGroup {
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

#[derive(Clone)]
struct CachedPostBindGroup {
    source_view: wgpu::TextureView,
    bind_group: wgpu::BindGroup,
}

impl CachedPostBindGroup {
    fn bind_group(&self) -> wgpu::BindGroup {
        let _source_view = &self.source_view;
        self.bind_group.clone()
    }
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
    vertex_count: u32,
    index_count: u32,
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

fn insert_bounded_cache<K, V>(cache: &mut HashMap<K, V>, key: K, value: V, max_entries: usize) -> V
where
    K: Clone + Eq + Hash,
    V: Clone,
{
    if let Some(existing) = cache.get(&key) {
        return existing.clone();
    }
    if max_entries == 0 {
        return value;
    }
    while cache.len() >= max_entries {
        let Some(evict_key) = cache.keys().next().cloned() else {
            break;
        };
        cache.remove(&evict_key);
    }
    cache.insert(key, value.clone());
    value
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

mod helpers_01;
mod helpers_02;
mod helpers_03;
mod init_defaults;
mod init_layouts;
mod init_pipelines;
mod methods_01;
mod methods_02;
mod methods_03;
mod methods_04;
mod methods_05;
mod methods_06;
mod methods_07;
mod uniforms;

use helpers_01::*;
use helpers_02::*;
use helpers_03::*;
use init_defaults::*;
use init_layouts::*;
use init_pipelines::*;
use uniforms::*;
