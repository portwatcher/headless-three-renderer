use std::thread;

use anyhow::{Context, Result, bail};
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};

use crate::types::SceneMesh;
use crate::util::{clamp01, color_to_f32, finite_color, finite_f32, parse_color, parse_transform};

pub const MAX_CLIPPING_PLANES: usize = 8;
const PARALLEL_MESH_PREPARE_THRESHOLD: usize = 8;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub tangent: [f32; 4],
    pub color: [f32; 4],
    pub uv: [f32; 2],
    pub uv2: [f32; 2],
}

impl Vertex {
    const ATTRIBUTES: [wgpu::VertexAttribute; 6] = wgpu::vertex_attr_array![
        0 => Float32x3,
        1 => Float32x3,
        2 => Float32x4,
        3 => Float32x4,
        4 => Float32x2,
        5 => Float32x2,
    ];

    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBUTES,
        }
    }
}

pub struct PreparedMesh {
    pub native_mesh_key: Option<u32>,
    pub vertices: Vec<Vertex>,
    pub indices: Option<Vec<u32>>,
    pub transform: Mat4,
    pub texture: Option<PreparedTexture>,
    pub normal_map: Option<PreparedTexture>,
    pub normal_map_type: NormalMapType,
    pub normal_scale: [f32; 2],
    pub bump_map: Option<PreparedTexture>,
    pub bump_scale: f32,
    pub matcap_map: Option<PreparedTexture>,
    pub gradient_map: Option<PreparedTexture>,
    pub metallic_roughness_texture: Option<PreparedTexture>,
    pub specular_map: Option<PreparedTexture>,
    pub emissive_map: Option<PreparedTexture>,
    pub ao_map: Option<PreparedTexture>,
    pub light_map: Option<PreparedTexture>,
    pub alpha_map: Option<PreparedTexture>,
    pub physical_maps: Option<PreparedPhysicalMaps>,
    pub clearcoat_normal_map: Option<PreparedTexture>,
    pub ao_map_intensity: f32,
    pub light_map_intensity: f32,
    pub metallic: f32,
    pub roughness: f32,
    pub clearcoat: f32,
    pub clearcoat_roughness: f32,
    pub clearcoat_normal_scale: [f32; 2],
    pub sheen_color: [f32; 3],
    pub sheen_roughness: f32,
    pub anisotropy: f32,
    pub anisotropy_rotation: f32,
    pub iridescence: f32,
    pub iridescence_ior: f32,
    pub iridescence_thickness_min: f32,
    pub iridescence_thickness_max: f32,
    pub transmission: f32,
    pub dispersion: f32,
    pub ior: f32,
    pub thickness: f32,
    pub attenuation_distance: f32,
    pub attenuation_color: [f32; 3],
    pub physical_specular_color: [f32; 3],
    pub physical_specular_intensity: f32,
    pub distance_reference_position: Option<[f32; 3]>,
    pub distance_near: Option<f32>,
    pub distance_far: Option<f32>,
    pub specular_color: [f32; 3],
    pub shininess: f32,
    pub emissive: [f32; 3],
    pub base_color: [f32; 4],
    pub depth_packing: DepthPacking,
    pub alpha_test: f32,
    pub alpha_hash: bool,
    pub alpha_to_coverage: bool,
    pub premultiplied_alpha: bool,
    pub tone_mapped: bool,
    pub fog: bool,
    pub clipping_planes: [[f32; 4]; MAX_CLIPPING_PLANES],
    pub clipping_plane_count: u32,
    pub clipping_union_count: u32,
    pub clip_shadows: bool,
    pub is_transparent: bool,
    pub blending: BlendMode,
    pub custom_blend: Option<CustomBlendState>,
    pub depth_test: bool,
    pub depth_func: StencilCompare,
    pub depth_write: bool,
    pub color_write: bool,
    pub polygon_offset: bool,
    pub polygon_offset_factor: f32,
    pub polygon_offset_units: i32,
    pub stencil_write: bool,
    pub stencil_ref: u32,
    pub stencil_func_mask: u32,
    pub stencil_write_mask: u32,
    pub stencil_func: StencilCompare,
    pub stencil_fail: StencilOperation,
    pub stencil_z_fail: StencilOperation,
    pub stencil_z_pass: StencilOperation,
    pub side: MeshSide,
    pub shadow_side: MeshSide,
    pub shading_model: ShadingModel,
    pub use_environment_map: Option<bool>,
    pub environment_map_intensity: Option<f32>,
    pub environment_map_combine: u32,
    pub environment_map_reflectivity: f32,
    pub environment_map_refraction: bool,
    pub environment_map_refraction_ratio: f32,
    pub topology: Topology,
    pub custom_fragment_shader: Option<String>,
    pub texture_transform: [f32; 6],
    pub texture_uses_uv2: bool,
    pub alpha_map_transform: [f32; 6],
    pub alpha_map_is_srgb: bool,
    pub alpha_map_uses_uv2: bool,
    pub normal_map_transform: [f32; 6],
    pub normal_map_uses_uv2: bool,
    pub bump_map_transform: [f32; 6],
    pub bump_map_uses_uv2: bool,
    pub matcap_map_transform: [f32; 6],
    pub matcap_map_uses_uv2: bool,
    pub metallic_roughness_texture_transform: [f32; 6],
    pub metallic_roughness_texture_is_srgb: bool,
    pub metallic_roughness_texture_uses_uv2: bool,
    pub emissive_map_transform: [f32; 6],
    pub emissive_map_uses_uv2: bool,
    pub ao_map_transform: [f32; 6],
    pub ao_map_is_srgb: bool,
    pub ao_map_uses_uv2: bool,
    pub light_map_transform: [f32; 6],
    pub light_map_uses_uv2: bool,
    pub specular_map_transform: [f32; 6],
    pub specular_map_is_srgb: bool,
    pub specular_map_uses_uv2: bool,
    pub specular_color_map_transform: [f32; 6],
    pub specular_color_map_uses_uv2: bool,
    pub specular_intensity_map_transform: [f32; 6],
    pub specular_intensity_map_uses_uv2: bool,
    pub clearcoat_map_transform: [f32; 6],
    pub clearcoat_map_uses_uv2: bool,
    pub clearcoat_roughness_map_transform: [f32; 6],
    pub clearcoat_roughness_map_uses_uv2: bool,
    pub clearcoat_normal_map_transform: [f32; 6],
    pub clearcoat_normal_map_uses_uv2: bool,
    pub sheen_color_map_transform: [f32; 6],
    pub sheen_color_map_uses_uv2: bool,
    pub sheen_roughness_map_transform: [f32; 6],
    pub sheen_roughness_map_uses_uv2: bool,
    pub anisotropy_map_transform: [f32; 6],
    pub anisotropy_map_uses_uv2: bool,
    pub iridescence_map_transform: [f32; 6],
    pub iridescence_map_uses_uv2: bool,
    pub iridescence_thickness_map_transform: [f32; 6],
    pub iridescence_thickness_map_uses_uv2: bool,
    pub transmission_map_transform: [f32; 6],
    pub transmission_map_uses_uv2: bool,
    pub thickness_map_transform: [f32; 6],
    pub thickness_map_uses_uv2: bool,
    pub texture_is_srgb: bool,
    pub matcap_map_is_srgb: bool,
    pub gradient_map_is_srgb: bool,
    pub emissive_map_is_srgb: bool,
    pub light_map_is_srgb: bool,
    pub cast_shadow: bool,
    pub receive_shadow: bool,
    pub group_order: f32,
    pub render_order: f32,
    pub sort_z: f32,
    pub sort_index: u32,
    pub material_variant: u32,
    pub material_sort_key: u32,
}

pub struct PreparedPhysicalMaps {
    /// RGBA: clearcoat, clearcoat roughness, transmission, thickness multipliers.
    pub scalar_map: PreparedTexture,
    /// RGBA: sheen color RGB, sheen roughness multiplier in A.
    pub sheen_map: PreparedTexture,
    /// RGBA: anisotropy direction RG, strength B.
    pub anisotropy_map: PreparedTexture,
    /// RGBA: specular color RGB, specular intensity multiplier in A.
    pub specular_map: PreparedTexture,
    /// RG: iridescence multiplier in R, thickness range factor in G.
    pub iridescence_map: PreparedTexture,
    pub physical_layers_sampler: TextureSamplerSettings,
    pub sheen_sampler: TextureSamplerSettings,
    pub specular_sampler: TextureSamplerSettings,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct TextureSamplerSettings {
    pub wrap_s: WrapMode,
    pub wrap_t: WrapMode,
    pub mag_filter: TextureFilter,
    pub min_filter: TextureFilter,
    pub mipmap_filter: MipmapFilter,
    pub anisotropy: u16,
}

impl Default for TextureSamplerSettings {
    fn default() -> Self {
        Self {
            wrap_s: WrapMode::ClampToEdge,
            wrap_t: WrapMode::ClampToEdge,
            mag_filter: TextureFilter::Linear,
            min_filter: TextureFilter::Linear,
            mipmap_filter: MipmapFilter::None,
            anisotropy: 1,
        }
    }
}

impl TextureSamplerSettings {
    fn from_texture(texture: Option<&PreparedTexture>) -> Self {
        texture.map_or_else(Self::default, |tex| Self {
            wrap_s: tex.wrap_s,
            wrap_t: tex.wrap_t,
            mag_filter: tex.mag_filter,
            min_filter: tex.min_filter,
            mipmap_filter: tex.mipmap_filter,
            anisotropy: tex.anisotropy,
        })
    }

    fn first_from_textures(textures: &[Option<&PreparedTexture>]) -> Self {
        for texture in textures {
            if let Some(tex) = *texture {
                return Self::from_texture(Some(tex));
            }
        }
        Self::default()
    }
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum Topology {
    #[default]
    Triangles,
    Lines,
    Points,
}

impl Topology {
    pub fn from_str_opt(value: Option<&str>) -> Self {
        match value {
            Some("lines") => Self::Lines,
            Some("points") => Self::Points,
            _ => Self::Triangles,
        }
    }

    pub fn primitive(self) -> wgpu::PrimitiveTopology {
        match self {
            Self::Triangles => wgpu::PrimitiveTopology::TriangleList,
            Self::Lines => wgpu::PrimitiveTopology::LineList,
            Self::Points => wgpu::PrimitiveTopology::PointList,
        }
    }
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum ShadingModel {
    #[default]
    Standard,
    Basic,
    Lambert,
    Normal,
    Matcap,
    Phong,
    Depth,
    Toon,
    Distance,
    Shadow,
}

impl ShadingModel {
    pub fn from_str_opt(value: Option<&str>) -> Self {
        match value {
            Some("basic") => Self::Basic,
            Some("lambert") => Self::Lambert,
            Some("normal") => Self::Normal,
            Some("matcap") => Self::Matcap,
            Some("phong") => Self::Phong,
            Some("depth") => Self::Depth,
            Some("toon") => Self::Toon,
            Some("distance") => Self::Distance,
            Some("shadow") => Self::Shadow,
            _ => Self::Standard,
        }
    }

    pub fn as_u32(self) -> u32 {
        match self {
            Self::Standard => 0,
            Self::Basic => 1,
            Self::Lambert => 2,
            Self::Normal => 3,
            Self::Matcap => 4,
            Self::Phong => 5,
            Self::Depth => 6,
            Self::Toon => 7,
            Self::Distance => 8,
            Self::Shadow => 9,
        }
    }
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum DepthPacking {
    #[default]
    Basic,
    Rgba,
    Rgb,
    Rg,
}

impl DepthPacking {
    pub fn from_three_constant(value: Option<u32>) -> Self {
        match value {
            Some(3201) => Self::Rgba,
            Some(3202) => Self::Rgb,
            Some(3203) => Self::Rg,
            _ => Self::Basic,
        }
    }

    pub fn as_u32(self) -> u32 {
        match self {
            Self::Basic => 0,
            Self::Rgba => 1,
            Self::Rgb => 2,
            Self::Rg => 3,
        }
    }
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum NormalMapType {
    #[default]
    Tangent,
    Object,
}

impl NormalMapType {
    pub fn from_str_opt(value: Option<&str>, mesh_index: usize) -> Result<Self> {
        match value {
            None | Some("tangent") => Ok(Self::Tangent),
            Some("object") => Ok(Self::Object),
            Some(other) => bail!(
                "mesh {mesh_index} normalMapType must be \"tangent\" or \"object\", got {other:?}"
            ),
        }
    }

    pub fn normal_mode(self) -> f32 {
        match self {
            Self::Tangent => 1.0,
            Self::Object => 3.0,
        }
    }
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum MeshSide {
    #[default]
    Front,
    Back,
    Double,
}

impl MeshSide {
    pub fn from_str_opt(value: Option<&str>) -> Self {
        match value {
            Some("back") => Self::Back,
            Some("double") => Self::Double,
            _ => Self::Front,
        }
    }

    pub fn cull_mode(self) -> Option<wgpu::Face> {
        match self {
            Self::Front => Some(wgpu::Face::Back),
            Self::Back => Some(wgpu::Face::Front),
            Self::Double => None,
        }
    }
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum BlendMode {
    None,
    #[default]
    Normal,
    Additive,
    Subtractive,
    Multiply,
    Custom,
}

impl BlendMode {
    pub fn from_str_opt(value: Option<&str>) -> Self {
        match value {
            Some("none") => Self::None,
            Some("additive") => Self::Additive,
            Some("subtractive") => Self::Subtractive,
            Some("multiply") => Self::Multiply,
            Some("custom") => Self::Custom,
            _ => Self::Normal,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum BlendEquation {
    Add,
    Subtract,
    ReverseSubtract,
    Min,
    Max,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum BlendFactor {
    Zero,
    One,
    SrcColor,
    OneMinusSrcColor,
    SrcAlpha,
    OneMinusSrcAlpha,
    DstAlpha,
    OneMinusDstAlpha,
    DstColor,
    OneMinusDstColor,
    SrcAlphaSaturate,
    ConstantColor,
    OneMinusConstantColor,
    ConstantAlpha,
    OneMinusConstantAlpha,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct CustomBlendState {
    pub color_equation: BlendEquation,
    pub alpha_equation: BlendEquation,
    pub color_src_factor: BlendFactor,
    pub color_dst_factor: BlendFactor,
    pub alpha_src_factor: BlendFactor,
    pub alpha_dst_factor: BlendFactor,
    pub constant: [f32; 4],
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum StencilCompare {
    Never,
    Less,
    Equal,
    LessEqual,
    Greater,
    NotEqual,
    GreaterEqual,
    Always,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum StencilOperation {
    Zero,
    Keep,
    Replace,
    IncrementClamp,
    DecrementClamp,
    IncrementWrap,
    DecrementWrap,
    Invert,
}

pub struct PreparedTextureMipLevel {
    pub rgba: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

pub struct PreparedTexture {
    pub rgba: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub mipmaps: Vec<PreparedTextureMipLevel>,
    pub wrap_s: WrapMode,
    pub wrap_t: WrapMode,
    pub mag_filter: TextureFilter,
    pub min_filter: TextureFilter,
    pub mipmap_filter: MipmapFilter,
    pub anisotropy: u16,
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum WrapMode {
    #[default]
    ClampToEdge,
    Repeat,
    MirrorRepeat,
}

impl WrapMode {
    pub fn from_str_opt(value: Option<&str>) -> Self {
        match value {
            Some("repeat") => Self::Repeat,
            Some("mirror") => Self::MirrorRepeat,
            _ => Self::ClampToEdge,
        }
    }

    pub fn to_address_mode(self) -> wgpu::AddressMode {
        match self {
            Self::ClampToEdge => wgpu::AddressMode::ClampToEdge,
            Self::Repeat => wgpu::AddressMode::Repeat,
            Self::MirrorRepeat => wgpu::AddressMode::MirrorRepeat,
        }
    }
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum TextureFilter {
    Nearest,
    #[default]
    Linear,
}

impl TextureFilter {
    pub fn from_str_opt(value: Option<&str>) -> Self {
        match value {
            Some("nearest") => Self::Nearest,
            _ => Self::Linear,
        }
    }

    pub fn from_min_filter_str(value: Option<&str>) -> Self {
        match value {
            Some("nearest") | Some("nearest-mipmap-nearest") | Some("nearest-mipmap-linear") => {
                Self::Nearest
            }
            _ => Self::Linear,
        }
    }

    pub fn to_filter_mode(self) -> wgpu::FilterMode {
        match self {
            Self::Nearest => wgpu::FilterMode::Nearest,
            Self::Linear => wgpu::FilterMode::Linear,
        }
    }
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum MipmapFilter {
    #[default]
    None,
    Nearest,
    Linear,
}

impl MipmapFilter {
    pub fn from_min_filter_str(value: Option<&str>) -> Self {
        match value {
            Some("nearest-mipmap-nearest") | Some("linear-mipmap-nearest") => Self::Nearest,
            Some("nearest-mipmap-linear") | Some("linear-mipmap-linear") => Self::Linear,
            _ => Self::None,
        }
    }

    pub fn to_mipmap_filter_mode(self) -> wgpu::MipmapFilterMode {
        match self {
            Self::None | Self::Nearest => wgpu::MipmapFilterMode::Nearest,
            Self::Linear => wgpu::MipmapFilterMode::Linear,
        }
    }
}

pub fn prepare_meshes(scene: &crate::types::RenderScene) -> Result<Vec<PreparedMesh>> {
    if let Some(meshes) = scene.meshes.as_deref() {
        prepare_mesh_slice(meshes)
    } else {
        Ok(Vec::new())
    }
}

fn prepare_mesh_slice(meshes: &[SceneMesh]) -> Result<Vec<PreparedMesh>> {
    if meshes.len() < PARALLEL_MESH_PREPARE_THRESHOLD {
        return meshes.iter().enumerate().map(prepare_mesh).collect();
    }

    let worker_count = thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1)
        .min(meshes.len());
    if worker_count <= 1 {
        return meshes.iter().enumerate().map(prepare_mesh).collect();
    }

    let chunk_size = meshes.len().div_ceil(worker_count);
    let mut chunks = thread::scope(|scope| {
        let handles = (0..worker_count)
            .filter_map(|worker_index| {
                let start = worker_index * chunk_size;
                if start >= meshes.len() {
                    return None;
                }
                let end = (start + chunk_size).min(meshes.len());
                Some(scope.spawn(move || {
                    let results = meshes[start..end]
                        .iter()
                        .enumerate()
                        .map(|(offset, mesh)| prepare_mesh((start + offset, mesh)))
                        .collect::<Vec<_>>();
                    (start, results)
                }))
            })
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .map(|handle| handle.join().expect("mesh preparation worker panicked"))
            .collect::<Vec<_>>()
    });
    chunks.sort_by_key(|(start, _)| *start);

    let mut prepared = Vec::with_capacity(meshes.len());
    for (_, results) in chunks {
        for result in results {
            prepared.push(result?);
        }
    }
    Ok(prepared)
}

pub fn texture_anisotropy(value: Option<f64>, field: &str) -> Result<u16> {
    match value {
        Some(value) => {
            let value = finite_f32(value, field)?;
            Ok(if value > 1.0 {
                value.floor().clamp(1.0, 16.0) as u16
            } else {
                1
            })
        }
        None => Ok(1),
    }
}

fn prepare_mesh((mesh_index, mesh): (usize, &SceneMesh)) -> Result<PreparedMesh> {
    let topology = Topology::from_str_opt(mesh.topology.as_deref());
    let cached_native_mesh = mesh.positions.is_empty() && mesh.native_mesh_key.is_some();

    let min_positions = match topology {
        Topology::Triangles => 9, // at least 3 xyz
        Topology::Lines => 6,     // at least 2 xyz
        Topology::Points => 3,    // at least 1 xyz
    };
    if cached_native_mesh {
        if mesh.native_vertex_count.unwrap_or(0) == 0 {
            bail!(
                "scene.meshes[{mesh_index}].nativeVertexCount must be positive when using nativeMeshKey without positions"
            );
        }
    } else if mesh.positions.len() < min_positions || mesh.positions.len() % 3 != 0 {
        bail!(
            "scene.meshes[{mesh_index}].positions must contain at least {} xyz vertices",
            min_positions / 3
        );
    }

    let vertex_count = if cached_native_mesh {
        mesh.native_vertex_count.unwrap() as usize
    } else {
        mesh.positions.len() / 3
    };
    let material_color = parse_color(
        mesh.color.as_deref(),
        [0.82, 0.82, 0.82, 1.0],
        &format!("scene.meshes[{mesh_index}].color"),
    )?;

    let color_mode = if cached_native_mesh {
        ColorMode::new(None, vertex_count, material_color, mesh_index)?
    } else {
        ColorMode::new(
            mesh.colors.as_deref(),
            vertex_count,
            material_color,
            mesh_index,
        )?
    };

    let uvs = if cached_native_mesh {
        None
    } else {
        mesh.uvs.as_deref()
    };
    let has_uvs = uvs.map_or(false, |u| u.len() == vertex_count * 2);
    if let Some(u) = uvs {
        if u.len() != vertex_count * 2 {
            bail!(
                "scene.meshes[{mesh_index}].uvs has length {}; expected {}",
                u.len(),
                vertex_count * 2,
            );
        }
    }

    let uvs2 = if cached_native_mesh {
        None
    } else {
        mesh.uvs2.as_deref()
    };
    let has_uvs2 = uvs2.map_or(false, |u| u.len() == vertex_count * 2);
    if let Some(u) = uvs2 {
        if u.len() != vertex_count * 2 {
            bail!(
                "scene.meshes[{mesh_index}].uvs2 has length {}; expected {}",
                u.len(),
                vertex_count * 2,
            );
        }
    }

    let normals = if cached_native_mesh {
        None
    } else {
        mesh.normals.as_deref()
    };
    let has_normals = normals.map_or(false, |n| n.len() == vertex_count * 3);
    if let Some(n) = normals {
        if n.len() != vertex_count * 3 {
            bail!(
                "scene.meshes[{mesh_index}].normals has length {}; expected {}",
                n.len(),
                vertex_count * 3,
            );
        }
    }

    let mut vertices = Vec::with_capacity(vertex_count);
    let normals_field = format!("scene.meshes[{mesh_index}].normals");
    let uvs_field = format!("scene.meshes[{mesh_index}].uvs");
    let uvs2_field = format!("scene.meshes[{mesh_index}].uvs2");
    if !cached_native_mesh {
        for vertex_index in 0..vertex_count {
            let base = vertex_index * 3;
            let uv_base = vertex_index * 2;
            vertices.push(Vertex {
                position: [
                    finite_f32(mesh.positions[base], "mesh position")?,
                    finite_f32(mesh.positions[base + 1], "mesh position")?,
                    finite_f32(mesh.positions[base + 2], "mesh position")?,
                ],
                normal: if has_normals {
                    let n = normals.unwrap();
                    [
                        finite_f32(n[base], &normals_field)?,
                        finite_f32(n[base + 1], &normals_field)?,
                        finite_f32(n[base + 2], &normals_field)?,
                    ]
                } else {
                    [0.0, 0.0, 0.0]
                },
                tangent: [0.0, 0.0, 0.0, 0.0],
                color: color_mode.color(vertex_index),
                uv: if has_uvs {
                    let u = uvs.unwrap();
                    [
                        finite_f32(u[uv_base], &uvs_field)?,
                        finite_f32(u[uv_base + 1], &uvs_field)?,
                    ]
                } else {
                    [0.0, 0.0]
                },
                uv2: if has_uvs2 {
                    let u = uvs2.unwrap();
                    [
                        finite_f32(u[uv_base], &uvs2_field)?,
                        finite_f32(u[uv_base + 1], &uvs2_field)?,
                    ]
                } else if has_uvs {
                    let u = uvs.unwrap();
                    [
                        finite_f32(u[uv_base], &uvs_field)?,
                        finite_f32(u[uv_base + 1], &uvs_field)?,
                    ]
                } else {
                    [0.0, 0.0]
                },
            });
        }
    }

    let mut indices = match &mesh.indices {
        Some(indices) => {
            let stride = match topology {
                Topology::Triangles => 3,
                Topology::Lines => 2,
                Topology::Points => 1,
            };
            if indices.len() < stride || indices.len() % stride != 0 {
                bail!(
                    "scene.meshes[{mesh_index}].indices must contain {} index/indices per primitive",
                    stride
                );
            }
            for &index in indices {
                if index as usize >= vertex_count {
                    bail!("scene.meshes[{mesh_index}].indices contains out-of-range index {index}");
                }
            }
            Some(indices.clone())
        }
        None if cached_native_mesh => None,
        None => {
            let stride = match topology {
                Topology::Triangles => 3,
                Topology::Lines => 2,
                Topology::Points => 1,
            };
            if vertex_count % stride != 0 {
                bail!(
                    "scene.meshes[{mesh_index}] has no indices, so positions must define complete primitives ({stride} vertices per primitive)"
                );
            }
            None
        }
    };

    let has_normal_map_slot = mesh
        .normal_map
        .as_ref()
        .is_some_and(|data| !data.is_empty());
    let use_flat_normals = mesh.flat_shading.unwrap_or(false)
        && !cached_native_mesh
        && !has_normal_map_slot
        && topology == Topology::Triangles;

    // If no normals were provided, compute vertex normals from triangle faces.
    // `flatShading` needs duplicated vertices for indexed geometry so each face
    // can carry its own normal. Lines and points stay unlit.
    if use_flat_normals {
        if let Some(flat_indices) = indices.take() {
            let mut expanded = Vec::with_capacity(flat_indices.len());
            for index in flat_indices {
                expanded.push(vertices[index as usize]);
            }
            vertices = expanded;
        }
        compute_flat_normals(&mut vertices, None);
    } else if !cached_native_mesh && !has_normals && topology == Topology::Triangles {
        compute_flat_normals(&mut vertices, indices.as_deref());
    }

    let (surface_texture_inputs, physical_texture_inputs) = thread::scope(|scope| -> Result<_> {
        let surface_handle = scope.spawn(|| prepare_surface_texture_inputs(mesh, mesh_index));
        let physical_handle = scope.spawn(|| prepare_physical_texture_inputs(mesh, mesh_index));

        let surface_texture_inputs = surface_handle
            .join()
            .expect("surface texture preparation worker panicked")?;
        let physical_texture_inputs = physical_handle
            .join()
            .expect("physical texture preparation worker panicked")?;

        Ok((surface_texture_inputs, physical_texture_inputs))
    })?;

    let SurfaceTextureInputs {
        texture,
        texture_transform,
        alpha_map_transform,
        normal_map_transform,
        bump_map_transform,
        matcap_map_transform,
        displacement_map_transform,
        displacement_map_is_srgb,
        metallic_roughness_texture_transform,
        metallic_roughness_texture_is_srgb,
        emissive_map_transform,
        ao_map_transform,
        light_map_transform,
        specular_map_transform,
        specular_color_map_transform,
        specular_intensity_map_transform,
        clearcoat_map_transform,
        clearcoat_roughness_map_transform,
        clearcoat_normal_map_transform,
        sheen_color_map_transform,
        sheen_roughness_map_transform,
        anisotropy_map_transform,
        iridescence_map_transform,
        iridescence_thickness_map_transform,
        transmission_map_transform,
        thickness_map_transform,
        texture_is_srgb,
        alpha_map_is_srgb,
        matcap_map_is_srgb,
        gradient_map_is_srgb,
        emissive_map_is_srgb,
        ao_map_is_srgb,
        light_map_is_srgb,
        specular_map_is_srgb,
        common,
    } = surface_texture_inputs;

    let CommonTextureInputs {
        normal_map,
        normal_map_type,
        normal_scale,
        bump_map,
        bump_scale,
        displacement_map,
        displacement_scale,
        displacement_bias,
        gradient_map,
        matcap_map,
        metallic_roughness_texture,
        specular_map,
        emissive_map,
        ao_map,
        ao_map_intensity,
        light_map,
        light_map_intensity,
        alpha_map,
    } = common;

    let PhysicalTextureInputs {
        clearcoat_map,
        clearcoat_roughness_map,
        clearcoat_normal_map,
        sheen_color_map,
        sheen_roughness_map,
        anisotropy_map,
        iridescence_map,
        iridescence_thickness_map,
        transmission_map,
        thickness_map,
        specular_color_map,
        specular_intensity_map,
    } = physical_texture_inputs;

    let side = MeshSide::from_str_opt(mesh.side.as_deref());
    let normal_scale = if side == MeshSide::Back {
        [-normal_scale[0], -normal_scale[1]]
    } else {
        normal_scale
    };
    let bump_scale = if side == MeshSide::Back {
        -bump_scale
    } else {
        bump_scale
    };

    let physical_maps = pack_physical_maps(PhysicalMapInputs {
        clearcoat: clearcoat_map.as_ref(),
        clearcoat_roughness: clearcoat_roughness_map.as_ref(),
        sheen_color: sheen_color_map.as_ref(),
        sheen_roughness: sheen_roughness_map.as_ref(),
        anisotropy: anisotropy_map.as_ref(),
        iridescence: iridescence_map.as_ref(),
        iridescence_thickness: iridescence_thickness_map.as_ref(),
        transmission: transmission_map.as_ref(),
        thickness: thickness_map.as_ref(),
        specular_color: specular_color_map.as_ref(),
        specular_intensity: specular_intensity_map.as_ref(),
        clearcoat_is_srgb: matches!(mesh.clearcoat_map_color_space.as_deref(), Some("srgb")),
        clearcoat_roughness_is_srgb: matches!(
            mesh.clearcoat_roughness_map_color_space.as_deref(),
            Some("srgb")
        ),
        anisotropy_is_srgb: matches!(mesh.anisotropy_map_color_space.as_deref(), Some("srgb")),
        iridescence_is_srgb: matches!(mesh.iridescence_map_color_space.as_deref(), Some("srgb")),
        iridescence_thickness_is_srgb: matches!(
            mesh.iridescence_thickness_map_color_space.as_deref(),
            Some("srgb")
        ),
        transmission_is_srgb: matches!(mesh.transmission_map_color_space.as_deref(), Some("srgb")),
        thickness_is_srgb: matches!(mesh.thickness_map_color_space.as_deref(), Some("srgb")),
        sheen_color_is_srgb: matches!(mesh.sheen_color_map_color_space.as_deref(), Some("srgb")),
        specular_color_is_srgb: matches!(
            mesh.specular_color_map_color_space.as_deref(),
            Some("srgb")
        ),
    });

    let metallic = finite_clamp01(mesh.metallic.unwrap_or(0.0), "mesh metallic")?;
    let roughness = finite_clamp01(mesh.roughness.unwrap_or(1.0), "mesh roughness")?;
    let clearcoat = finite_clamp01(mesh.clearcoat.unwrap_or(0.0), "mesh clearcoat")?;
    let clearcoat_roughness = finite_clamp01(
        mesh.clearcoat_roughness.unwrap_or(0.0),
        "mesh clearcoatRoughness",
    )?;
    let clearcoat_normal_scale = parse_optional_vec2(
        mesh.clearcoat_normal_scale.as_deref(),
        [1.0, 1.0],
        "mesh clearcoatNormalScale",
    )?;
    let clearcoat_normal_scale = if side == MeshSide::Back {
        [-clearcoat_normal_scale[0], -clearcoat_normal_scale[1]]
    } else {
        clearcoat_normal_scale
    };
    let sheen_color = parse_optional_clamped_color3(
        mesh.sheen_color.as_deref(),
        [0.0, 0.0, 0.0],
        "mesh sheenColor",
    )?;
    let sheen_roughness =
        finite_clamp01(mesh.sheen_roughness.unwrap_or(1.0), "mesh sheenRoughness")?.max(0.0001);
    let anisotropy = finite_clamp01(mesh.anisotropy.unwrap_or(0.0), "mesh anisotropy")?;
    let anisotropy_rotation = finite_f32(
        mesh.anisotropy_rotation.unwrap_or(0.0),
        "mesh anisotropyRotation",
    )?;
    let iridescence = finite_clamp01(mesh.iridescence.unwrap_or(0.0), "mesh iridescence")?;
    let iridescence_ior =
        finite_f32(mesh.iridescence_ior.unwrap_or(1.3), "mesh iridescenceIor")?.clamp(1.0, 2.333);
    let iridescence_thickness_min = mesh
        .iridescence_thickness_min
        .map(|value| finite_f32(value, "mesh iridescenceThicknessMin"))
        .transpose()?
        .unwrap_or(100.0)
        .max(0.0);
    let iridescence_thickness_max = mesh
        .iridescence_thickness_max
        .map(|value| finite_f32(value, "mesh iridescenceThicknessMax"))
        .transpose()?
        .unwrap_or(400.0)
        .max(iridescence_thickness_min);
    let transmission = finite_clamp01(mesh.transmission.unwrap_or(0.0), "mesh transmission")?;
    let dispersion = finite_f32(mesh.dispersion.unwrap_or(0.0), "mesh dispersion")?.max(0.0);
    let ior = finite_f32(mesh.ior.unwrap_or(1.5), "mesh ior")?.clamp(1.0, 2.333);
    let thickness = finite_f32(mesh.thickness.unwrap_or(0.0), "mesh thickness")?.max(0.0);
    let attenuation_distance = mesh
        .attenuation_distance
        .map(|value| finite_f32(value, "mesh attenuationDistance"))
        .transpose()?
        .unwrap_or(1.0e20)
        .max(0.0)
        .min(1.0e20);
    let attenuation_color = parse_optional_clamped_color3(
        mesh.attenuation_color.as_deref(),
        [1.0, 1.0, 1.0],
        "mesh attenuationColor",
    )?;
    let physical_specular_color = parse_optional_clamped_color3(
        mesh.physical_specular_color.as_deref(),
        [1.0, 1.0, 1.0],
        "mesh specularColor",
    )?;
    let physical_specular_intensity = finite_clamp01(
        mesh.physical_specular_intensity.unwrap_or(1.0),
        "mesh specularIntensity",
    )?;
    let distance_reference_position = parse_optional_vec3(
        mesh.distance_reference_position.as_deref(),
        &format!("scene.meshes[{mesh_index}].distanceReferencePosition"),
    )?;
    let distance_near = mesh
        .distance_near
        .map(|value| finite_f32(value, "mesh distanceNear"))
        .transpose()?;
    let distance_far = mesh
        .distance_far
        .map(|value| finite_f32(value, "mesh distanceFar"))
        .transpose()?;
    let specular_color = parse_optional_clamped_color3(
        mesh.specular_color.as_deref(),
        [17.0 / 255.0, 17.0 / 255.0, 17.0 / 255.0],
        "mesh specular",
    )?;
    let shininess = finite_f32(mesh.shininess.unwrap_or(30.0), "mesh shininess")?.max(0.0001);

    if !cached_native_mesh && let Some(displacement_map) = displacement_map.as_ref() {
        if has_uvs && topology == Topology::Triangles {
            apply_displacement_map(
                &mut vertices,
                displacement_map,
                displacement_map_transform,
                displacement_map_is_srgb,
                mesh.displacement_map_uses_uv2.unwrap_or(false),
                displacement_scale,
                displacement_bias,
            );
        }
    }

    // Compute tangents when normal/bump mapping or anisotropic shading needs a frame.
    if !cached_native_mesh
        && (normal_map.is_some()
        || bump_map.is_some()
        || clearcoat_normal_map.is_some()
        || anisotropy > 0.0)
        && has_uvs
        && topology == Topology::Triangles
    {
        compute_tangents(&mut vertices, mesh.indices.as_deref());
    }

    let emissive_intensity = finite_f32(
        mesh.emissive_intensity.unwrap_or(1.0),
        "mesh emissiveIntensity",
    )?;
    let emissive_color =
        parse_optional_clamped_color3(mesh.emissive.as_deref(), [0.0, 0.0, 0.0], "mesh emissive")?;
    let emissive = [
        emissive_color[0] * emissive_intensity,
        emissive_color[1] * emissive_intensity,
        emissive_color[2] * emissive_intensity,
    ];

    let alpha_test = finite_clamp01(mesh.alpha_test.unwrap_or(0.0), "mesh alphaTest")?;
    let alpha_hash = mesh.alpha_hash.unwrap_or(false);
    let alpha_to_coverage = mesh.alpha_to_coverage.unwrap_or(false);
    let premultiplied_alpha = mesh.premultiplied_alpha.unwrap_or(false);
    let tone_mapped = mesh.tone_mapped.unwrap_or(true);
    let fog = mesh.fog.unwrap_or(true);
    let (clipping_planes, clipping_plane_count, clipping_union_count) = parse_clipping_planes(
        mesh.clipping_planes.as_deref(),
        mesh.clipping_union_count,
        mesh_index,
    )?;
    let is_transparent = (!alpha_hash && mesh.transparent.unwrap_or(material_color[3] < 0.999))
        || transmission > 0.0001;
    let blending = BlendMode::from_str_opt(mesh.blending.as_deref());
    let custom_blend = parse_custom_blend_state(mesh, blending, mesh_index)?;
    let depth_test = mesh.depth_test.unwrap_or(true);
    let depth_func = parse_depth_func(mesh.depth_func.as_deref(), mesh_index)?;
    let depth_write = depth_test && mesh.depth_write.unwrap_or(true);
    let color_write = mesh.color_write.unwrap_or(true);
    let polygon_offset = mesh.polygon_offset.unwrap_or(false);
    let polygon_offset_factor = finite_f32(
        mesh.polygon_offset_factor.unwrap_or(0.0),
        "mesh polygonOffsetFactor",
    )?;
    let polygon_offset_units = finite_f32(
        mesh.polygon_offset_units.unwrap_or(0.0),
        "mesh polygonOffsetUnits",
    )?
    .round()
    .clamp(i32::MIN as f32, i32::MAX as f32) as i32;
    let stencil_write = mesh.stencil_write.unwrap_or(false);
    let stencil_ref = mesh.stencil_ref.unwrap_or(0).min(0xff);
    let stencil_func_mask = mesh.stencil_func_mask.unwrap_or(0xff).min(0xff);
    let stencil_write_mask = mesh.stencil_write_mask.unwrap_or(0xff).min(0xff);
    let stencil_func =
        parse_stencil_compare(mesh.stencil_func.unwrap_or(519), mesh_index, "stencilFunc")?;
    let stencil_fail =
        parse_stencil_operation(mesh.stencil_fail.unwrap_or(7680), mesh_index, "stencilFail")?;
    let stencil_z_fail = parse_stencil_operation(
        mesh.stencil_z_fail.unwrap_or(7680),
        mesh_index,
        "stencilZFail",
    )?;
    let stencil_z_pass = parse_stencil_operation(
        mesh.stencil_z_pass.unwrap_or(7680),
        mesh_index,
        "stencilZPass",
    )?;
    let shadow_side = parse_shadow_side(mesh.shadow_side.as_deref(), mesh_index)?;
    let requested_shading_model = ShadingModel::from_str_opt(mesh.shading_model.as_deref());
    // Lines and points are normally unlit. Depth/distance wireframe meshes are
    // emitted as lines but still use position-only material outputs.
    let shading_model = match topology {
        Topology::Triangles => requested_shading_model,
        Topology::Lines
            if matches!(
                requested_shading_model,
                ShadingModel::Depth | ShadingModel::Distance
            ) =>
        {
            requested_shading_model
        }
        Topology::Lines | Topology::Points => ShadingModel::Basic,
    };
    let environment_map_intensity = match mesh.environment_map_intensity {
        Some(value) => Some(finite_f32(value, "mesh environmentMapIntensity")?),
        None => None,
    };
    let environment_map_combine = mesh.environment_map_combine.unwrap_or(0);
    if environment_map_combine > 2 {
        bail!("mesh {mesh_index} environmentMapCombine must be 0, 1, or 2");
    }
    let environment_map_reflectivity = match mesh.environment_map_reflectivity {
        Some(value) => finite_f32(value, "mesh environmentMapReflectivity")?,
        None => 1.0,
    };
    let environment_map_refraction = mesh.environment_map_refraction.unwrap_or(false);
    let environment_map_refraction_ratio = match mesh.environment_map_refraction_ratio {
        Some(value) => finite_f32(value, "mesh environmentMapRefractionRatio")?,
        None => 0.98,
    };

    Ok(PreparedMesh {
        native_mesh_key: mesh.native_mesh_key,
        vertices,
        indices,
        transform: parse_transform(mesh.transform.as_deref(), mesh_index)?,
        texture,
        normal_map,
        normal_map_type,
        normal_scale,
        bump_map,
        bump_scale,
        matcap_map,
        gradient_map,
        metallic_roughness_texture,
        specular_map,
        emissive_map,
        ao_map,
        light_map,
        alpha_map,
        physical_maps,
        clearcoat_normal_map,
        ao_map_intensity,
        light_map_intensity,
        metallic,
        roughness,
        clearcoat,
        clearcoat_roughness,
        clearcoat_normal_scale,
        sheen_color,
        sheen_roughness,
        anisotropy,
        anisotropy_rotation,
        iridescence,
        iridescence_ior,
        iridescence_thickness_min,
        iridescence_thickness_max,
        transmission,
        dispersion,
        ior,
        thickness,
        attenuation_distance,
        attenuation_color,
        physical_specular_color,
        physical_specular_intensity,
        distance_reference_position,
        distance_near,
        distance_far,
        specular_color,
        shininess,
        emissive,
        base_color: color_to_f32(material_color),
        depth_packing: DepthPacking::from_three_constant(mesh.depth_packing),
        alpha_test,
        alpha_hash,
        alpha_to_coverage,
        premultiplied_alpha,
        tone_mapped,
        fog,
        clipping_planes,
        clipping_plane_count,
        clipping_union_count,
        clip_shadows: mesh.clip_shadows.unwrap_or(false),
        is_transparent,
        blending,
        custom_blend,
        depth_test,
        depth_func,
        depth_write,
        color_write,
        polygon_offset,
        polygon_offset_factor,
        polygon_offset_units,
        stencil_write,
        stencil_ref,
        stencil_func_mask,
        stencil_write_mask,
        stencil_func,
        stencil_fail,
        stencil_z_fail,
        stencil_z_pass,
        side,
        shadow_side,
        shading_model,
        use_environment_map: mesh.use_environment_map,
        environment_map_intensity,
        environment_map_combine,
        environment_map_reflectivity,
        environment_map_refraction,
        environment_map_refraction_ratio,
        topology,
        custom_fragment_shader: mesh
            .custom_fragment_shader
            .as_deref()
            .map(str::trim)
            .filter(|source| !source.is_empty())
            .map(str::to_owned),
        texture_transform,
        texture_uses_uv2: mesh.texture_uses_uv2.unwrap_or(false),
        alpha_map_transform,
        alpha_map_is_srgb,
        alpha_map_uses_uv2: mesh.alpha_map_uses_uv2.unwrap_or(false),
        normal_map_transform,
        normal_map_uses_uv2: mesh.normal_map_uses_uv2.unwrap_or(false),
        bump_map_transform,
        bump_map_uses_uv2: mesh.bump_map_uses_uv2.unwrap_or(false),
        matcap_map_transform,
        matcap_map_uses_uv2: mesh.matcap_map_uses_uv2.unwrap_or(false),
        metallic_roughness_texture_transform,
        metallic_roughness_texture_is_srgb,
        metallic_roughness_texture_uses_uv2: mesh
            .metallic_roughness_texture_uses_uv2
            .unwrap_or(false),
        emissive_map_transform,
        emissive_map_uses_uv2: mesh.emissive_map_uses_uv2.unwrap_or(false),
        ao_map_transform,
        ao_map_is_srgb,
        ao_map_uses_uv2: mesh.ao_map_uses_uv2.unwrap_or(false),
        light_map_transform,
        light_map_uses_uv2: mesh.light_map_uses_uv2.unwrap_or(false),
        specular_map_transform,
        specular_map_is_srgb,
        specular_map_uses_uv2: mesh.specular_map_uses_uv2.unwrap_or(false),
        specular_color_map_transform,
        specular_color_map_uses_uv2: mesh.specular_color_map_uses_uv2.unwrap_or(false),
        specular_intensity_map_transform,
        specular_intensity_map_uses_uv2: mesh.specular_intensity_map_uses_uv2.unwrap_or(false),
        clearcoat_map_transform,
        clearcoat_map_uses_uv2: mesh.clearcoat_map_uses_uv2.unwrap_or(false),
        clearcoat_roughness_map_transform,
        clearcoat_roughness_map_uses_uv2: mesh.clearcoat_roughness_map_uses_uv2.unwrap_or(false),
        clearcoat_normal_map_transform,
        clearcoat_normal_map_uses_uv2: mesh.clearcoat_normal_map_uses_uv2.unwrap_or(false),
        sheen_color_map_transform,
        sheen_color_map_uses_uv2: mesh.sheen_color_map_uses_uv2.unwrap_or(false),
        sheen_roughness_map_transform,
        sheen_roughness_map_uses_uv2: mesh.sheen_roughness_map_uses_uv2.unwrap_or(false),
        anisotropy_map_transform,
        anisotropy_map_uses_uv2: mesh.anisotropy_map_uses_uv2.unwrap_or(false),
        iridescence_map_transform,
        iridescence_map_uses_uv2: mesh.iridescence_map_uses_uv2.unwrap_or(false),
        iridescence_thickness_map_transform,
        iridescence_thickness_map_uses_uv2: mesh
            .iridescence_thickness_map_uses_uv2
            .unwrap_or(false),
        transmission_map_transform,
        transmission_map_uses_uv2: mesh.transmission_map_uses_uv2.unwrap_or(false),
        thickness_map_transform,
        thickness_map_uses_uv2: mesh.thickness_map_uses_uv2.unwrap_or(false),
        texture_is_srgb,
        matcap_map_is_srgb,
        gradient_map_is_srgb,
        emissive_map_is_srgb,
        light_map_is_srgb,
        cast_shadow: mesh.cast_shadow.unwrap_or(false),
        receive_shadow: mesh.receive_shadow.unwrap_or(false),
        group_order: finite_f32(mesh.group_order.unwrap_or(0.0), "mesh groupOrder")?,
        render_order: finite_f32(mesh.render_order.unwrap_or(0.0), "mesh renderOrder")?,
        sort_z: finite_f32(mesh.sort_z.unwrap_or(0.0), "mesh sortZ")?,
        sort_index: mesh.sort_index.unwrap_or(mesh_index as u32),
        material_variant: mesh.material_variant.unwrap_or(0),
        material_sort_key: mesh.material_sort_key.unwrap_or(0),
    })
}

fn parse_texture_transform(values: Option<&[f64]>, mesh_index: usize) -> Result<[f32; 6]> {
    let Some(values) = values else {
        return Ok([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    };
    if values.len() != 6 {
        bail!("scene.meshes[{mesh_index}].textureTransform must be an array of 6 numbers");
    }
    Ok([
        finite_f32(values[0], "mesh textureTransform")?,
        finite_f32(values[1], "mesh textureTransform")?,
        finite_f32(values[2], "mesh textureTransform")?,
        finite_f32(values[3], "mesh textureTransform")?,
        finite_f32(values[4], "mesh textureTransform")?,
        finite_f32(values[5], "mesh textureTransform")?,
    ])
}

fn parse_optional_vec3(values: Option<&[f64]>, field: &str) -> Result<Option<[f32; 3]>> {
    let Some(values) = values else {
        return Ok(None);
    };
    if values.len() != 3 {
        bail!("{field} must be an array of 3 numbers");
    }
    Ok(Some([
        finite_f32(values[0], field)?,
        finite_f32(values[1], field)?,
        finite_f32(values[2], field)?,
    ]))
}

fn parse_optional_vec2(values: Option<&[f64]>, default: [f32; 2], field: &str) -> Result<[f32; 2]> {
    let Some(values) = values else {
        return Ok(default);
    };
    if values.len() != 2 {
        bail!("{field} must be an array of 2 numbers");
    }
    Ok([finite_f32(values[0], field)?, finite_f32(values[1], field)?])
}

fn parse_optional_clamped_color3(
    values: Option<&[f64]>,
    default: [f32; 3],
    field: &str,
) -> Result<[f32; 3]> {
    let Some(values) = values else {
        return Ok(default);
    };
    if values.len() != 3 {
        bail!("{field} must be an array of 3 numbers");
    }
    Ok([
        finite_clamp01(values[0], field)?,
        finite_clamp01(values[1], field)?,
        finite_clamp01(values[2], field)?,
    ])
}

fn finite_clamp01(value: f64, field: &str) -> Result<f32> {
    Ok(finite_f32(value, field)?.clamp(0.0, 1.0))
}

fn parse_clipping_planes(
    values: Option<&[f64]>,
    union_count: Option<u32>,
    mesh_index: usize,
) -> Result<([[f32; 4]; MAX_CLIPPING_PLANES], u32, u32)> {
    let Some(values) = values else {
        return Ok(([[0.0; 4]; MAX_CLIPPING_PLANES], 0, 0));
    };
    if values.len() % 4 != 0 {
        bail!(
            "scene.meshes[{mesh_index}].clippingPlanes must be an array of [nx, ny, nz, constant] planes"
        );
    }

    let count = (values.len() / 4).min(MAX_CLIPPING_PLANES);
    let mut planes = [[0.0; 4]; MAX_CLIPPING_PLANES];
    for plane_index in 0..count {
        let base = plane_index * 4;
        planes[plane_index] = [
            finite_f32(values[base], "mesh clippingPlanes")?,
            finite_f32(values[base + 1], "mesh clippingPlanes")?,
            finite_f32(values[base + 2], "mesh clippingPlanes")?,
            finite_f32(values[base + 3], "mesh clippingPlanes")?,
        ];
    }

    let count = count as u32;
    let union_count = union_count.unwrap_or(count).min(count);
    Ok((planes, count, union_count))
}

fn parse_custom_blend_state(
    mesh: &SceneMesh,
    blending: BlendMode,
    mesh_index: usize,
) -> Result<Option<CustomBlendState>> {
    if blending != BlendMode::Custom {
        return Ok(None);
    }

    const ADD_EQUATION: u32 = 100;
    const SRC_ALPHA_FACTOR: u32 = 204;
    const ONE_MINUS_SRC_ALPHA_FACTOR: u32 = 205;

    let color_equation = parse_blend_equation(
        mesh.blend_equation.unwrap_or(ADD_EQUATION),
        mesh_index,
        "blendEquation",
    )?;
    let alpha_equation = parse_blend_equation(
        mesh.blend_equation_alpha
            .unwrap_or(mesh.blend_equation.unwrap_or(ADD_EQUATION)),
        mesh_index,
        "blendEquationAlpha",
    )?;
    let color_src_factor = parse_blend_factor(
        mesh.blend_src.unwrap_or(SRC_ALPHA_FACTOR),
        mesh_index,
        "blendSrc",
    )?;
    let color_dst_factor = parse_blend_factor(
        mesh.blend_dst.unwrap_or(ONE_MINUS_SRC_ALPHA_FACTOR),
        mesh_index,
        "blendDst",
    )?;
    let alpha_src_factor = parse_blend_factor(
        mesh.blend_src_alpha
            .unwrap_or(mesh.blend_src.unwrap_or(SRC_ALPHA_FACTOR)),
        mesh_index,
        "blendSrcAlpha",
    )?;
    let alpha_dst_factor = parse_blend_factor(
        mesh.blend_dst_alpha
            .unwrap_or(mesh.blend_dst.unwrap_or(ONE_MINUS_SRC_ALPHA_FACTOR)),
        mesh_index,
        "blendDstAlpha",
    )?;
    let blend_color = parse_color(
        mesh.blend_color.as_deref(),
        [0.0, 0.0, 0.0, 1.0],
        &format!("scene.meshes[{mesh_index}].blendColor"),
    )?;
    let blend_alpha = finite_clamp01(mesh.blend_alpha.unwrap_or(0.0), "mesh blendAlpha")?;

    Ok(Some(CustomBlendState {
        color_equation,
        alpha_equation,
        color_src_factor,
        color_dst_factor,
        alpha_src_factor,
        alpha_dst_factor,
        constant: [
            clamp01(blend_color[0]) as f32,
            clamp01(blend_color[1]) as f32,
            clamp01(blend_color[2]) as f32,
            blend_alpha,
        ],
    }))
}

fn parse_blend_equation(value: u32, mesh_index: usize, field: &str) -> Result<BlendEquation> {
    match value {
        100 => Ok(BlendEquation::Add),
        101 => Ok(BlendEquation::Subtract),
        102 => Ok(BlendEquation::ReverseSubtract),
        103 => Ok(BlendEquation::Min),
        104 => Ok(BlendEquation::Max),
        _ => bail!("scene.meshes[{mesh_index}].{field} has unsupported blend equation {value}"),
    }
}

fn parse_blend_factor(value: u32, mesh_index: usize, field: &str) -> Result<BlendFactor> {
    match value {
        200 => Ok(BlendFactor::Zero),
        201 => Ok(BlendFactor::One),
        202 => Ok(BlendFactor::SrcColor),
        203 => Ok(BlendFactor::OneMinusSrcColor),
        204 => Ok(BlendFactor::SrcAlpha),
        205 => Ok(BlendFactor::OneMinusSrcAlpha),
        206 => Ok(BlendFactor::DstAlpha),
        207 => Ok(BlendFactor::OneMinusDstAlpha),
        208 => Ok(BlendFactor::DstColor),
        209 => Ok(BlendFactor::OneMinusDstColor),
        210 => Ok(BlendFactor::SrcAlphaSaturate),
        211 => Ok(BlendFactor::ConstantColor),
        212 => Ok(BlendFactor::OneMinusConstantColor),
        213 => Ok(BlendFactor::ConstantAlpha),
        214 => Ok(BlendFactor::OneMinusConstantAlpha),
        _ => bail!("scene.meshes[{mesh_index}].{field} has unsupported blend factor {value}"),
    }
}

fn parse_depth_func(value: Option<&str>, mesh_index: usize) -> Result<StencilCompare> {
    match value.unwrap_or("less-equal") {
        "never" => Ok(StencilCompare::Never),
        "less" => Ok(StencilCompare::Less),
        "equal" => Ok(StencilCompare::Equal),
        "less-equal" => Ok(StencilCompare::LessEqual),
        "greater" => Ok(StencilCompare::Greater),
        "not-equal" => Ok(StencilCompare::NotEqual),
        "greater-equal" => Ok(StencilCompare::GreaterEqual),
        "always" => Ok(StencilCompare::Always),
        other => {
            bail!("scene.meshes[{mesh_index}].depthFunc has unsupported compare function `{other}`")
        }
    }
}

fn parse_shadow_side(value: Option<&str>, mesh_index: usize) -> Result<MeshSide> {
    match value {
        None => Ok(MeshSide::Double),
        Some("front") => Ok(MeshSide::Front),
        Some("back") => Ok(MeshSide::Back),
        Some("double") => Ok(MeshSide::Double),
        Some(other) => {
            bail!("scene.meshes[{mesh_index}].shadowSide has unsupported side `{other}`")
        }
    }
}

fn parse_stencil_compare(value: u32, mesh_index: usize, field: &str) -> Result<StencilCompare> {
    match value {
        512 => Ok(StencilCompare::Never),
        513 => Ok(StencilCompare::Less),
        514 => Ok(StencilCompare::Equal),
        515 => Ok(StencilCompare::LessEqual),
        516 => Ok(StencilCompare::Greater),
        517 => Ok(StencilCompare::NotEqual),
        518 => Ok(StencilCompare::GreaterEqual),
        519 => Ok(StencilCompare::Always),
        _ => bail!("scene.meshes[{mesh_index}].{field} has unsupported stencil function {value}"),
    }
}

fn parse_stencil_operation(value: u32, mesh_index: usize, field: &str) -> Result<StencilOperation> {
    match value {
        0 => Ok(StencilOperation::Zero),
        7680 => Ok(StencilOperation::Keep),
        7681 => Ok(StencilOperation::Replace),
        7682 => Ok(StencilOperation::IncrementClamp),
        7683 => Ok(StencilOperation::DecrementClamp),
        34055 => Ok(StencilOperation::IncrementWrap),
        34056 => Ok(StencilOperation::DecrementWrap),
        5386 => Ok(StencilOperation::Invert),
        _ => bail!("scene.meshes[{mesh_index}].{field} has unsupported stencil operation {value}"),
    }
}

pub fn decode_texture(
    data: &[u8],
    width_hint: Option<u32>,
    height_hint: Option<u32>,
    mesh_index: usize,
) -> Result<PreparedTexture> {
    decode_texture_with_label(
        data,
        width_hint,
        height_hint,
        &format!("scene.meshes[{mesh_index}].texture"),
    )
}

pub fn decode_texture_with_label(
    data: &[u8],
    width_hint: Option<u32>,
    height_hint: Option<u32>,
    label: &str,
) -> Result<PreparedTexture> {
    let w = width_hint.unwrap_or(0);
    let h = height_hint.unwrap_or(0);

    if w > 0 && h > 0 && data.len() >= (w as usize) * (h as usize) * 4 {
        let base_len = (w as usize) * (h as usize) * 4;
        let mut mipmaps = Vec::new();
        let mut offset = base_len;
        let mut mip_width = w;
        let mut mip_height = h;
        let mut level = 1u32;

        while offset < data.len() {
            mip_width = (mip_width / 2).max(1);
            mip_height = (mip_height / 2).max(1);
            let level_len = (mip_width as usize) * (mip_height as usize) * 4;
            if data.len() - offset < level_len {
                bail!(
                    "{label}: explicit mipmap level {level} expected {mip_width}x{mip_height} RGBA8 bytes"
                );
            }
            let level_end = offset + level_len;
            mipmaps.push(PreparedTextureMipLevel {
                rgba: data[offset..level_end].to_vec(),
                width: mip_width,
                height: mip_height,
            });
            offset = level_end;
            if mip_width == 1 && mip_height == 1 {
                if offset != data.len() {
                    bail!("{label}: explicit mipmap data continues after the 1x1 level");
                }
                break;
            }
            level += 1;
        }
        if !mipmaps.is_empty() && (mip_width != 1 || mip_height != 1) {
            bail!("{label}: explicit mipmap data must include the complete chain down to 1x1");
        }

        return Ok(PreparedTexture {
            rgba: data[..base_len].to_vec(),
            width: w,
            height: h,
            mipmaps,
            wrap_s: WrapMode::ClampToEdge,
            wrap_t: WrapMode::ClampToEdge,
            mag_filter: TextureFilter::Linear,
            min_filter: TextureFilter::Linear,
            mipmap_filter: MipmapFilter::None,
            anisotropy: 1,
        });
    }

    let img = image::load_from_memory(data)
        .with_context(|| format!("{label}: failed to decode image ({} bytes)", data.len()))?;
    let rgba = img.to_rgba8();
    Ok(PreparedTexture {
        width: rgba.width(),
        height: rgba.height(),
        rgba: rgba.into_raw(),
        mipmaps: Vec::new(),
        wrap_s: WrapMode::ClampToEdge,
        wrap_t: WrapMode::ClampToEdge,
        mag_filter: TextureFilter::Linear,
        min_filter: TextureFilter::Linear,
        mipmap_filter: MipmapFilter::None,
        anisotropy: 1,
    })
}

fn decode_optional_texture(
    data: Option<&[u8]>,
    width_hint: Option<u32>,
    height_hint: Option<u32>,
    mesh_index: usize,
) -> Result<Option<PreparedTexture>> {
    match data {
        Some(tex_data) if !tex_data.is_empty() => Ok(Some(decode_texture(
            tex_data,
            width_hint,
            height_hint,
            mesh_index,
        )?)),
        _ => Ok(None),
    }
}

fn decode_optional_texture_with_sampling(
    data: Option<&[u8]>,
    width_hint: Option<u32>,
    height_hint: Option<u32>,
    mesh_index: usize,
    wrap_s: Option<&str>,
    wrap_t: Option<&str>,
    mag_filter: Option<&str>,
    min_filter: Option<&str>,
    anisotropy: Option<f64>,
    anisotropy_field: &str,
) -> Result<Option<PreparedTexture>> {
    let mut texture = decode_optional_texture(data, width_hint, height_hint, mesh_index)?;
    if let Some(tex) = texture.as_mut() {
        apply_texture_sampling(
            tex,
            wrap_s,
            wrap_t,
            mag_filter,
            min_filter,
            anisotropy,
            anisotropy_field,
        )?;
    }
    Ok(texture)
}

fn apply_texture_sampling(
    texture: &mut PreparedTexture,
    wrap_s: Option<&str>,
    wrap_t: Option<&str>,
    mag_filter: Option<&str>,
    min_filter: Option<&str>,
    anisotropy: Option<f64>,
    anisotropy_field: &str,
) -> Result<()> {
    texture.wrap_s = WrapMode::from_str_opt(wrap_s);
    texture.wrap_t = WrapMode::from_str_opt(wrap_t);
    texture.mag_filter = TextureFilter::from_str_opt(mag_filter);
    texture.min_filter = TextureFilter::from_min_filter_str(min_filter);
    texture.mipmap_filter = MipmapFilter::from_min_filter_str(min_filter);
    texture.anisotropy = texture_anisotropy(anisotropy, anisotropy_field)?;
    Ok(())
}

struct SurfaceTextureInputs {
    texture: Option<PreparedTexture>,
    texture_transform: [f32; 6],
    alpha_map_transform: [f32; 6],
    normal_map_transform: [f32; 6],
    bump_map_transform: [f32; 6],
    matcap_map_transform: [f32; 6],
    displacement_map_transform: [f32; 6],
    displacement_map_is_srgb: bool,
    metallic_roughness_texture_transform: [f32; 6],
    emissive_map_transform: [f32; 6],
    ao_map_transform: [f32; 6],
    light_map_transform: [f32; 6],
    specular_map_transform: [f32; 6],
    specular_color_map_transform: [f32; 6],
    specular_intensity_map_transform: [f32; 6],
    clearcoat_map_transform: [f32; 6],
    clearcoat_roughness_map_transform: [f32; 6],
    clearcoat_normal_map_transform: [f32; 6],
    sheen_color_map_transform: [f32; 6],
    sheen_roughness_map_transform: [f32; 6],
    anisotropy_map_transform: [f32; 6],
    iridescence_map_transform: [f32; 6],
    iridescence_thickness_map_transform: [f32; 6],
    transmission_map_transform: [f32; 6],
    thickness_map_transform: [f32; 6],
    texture_is_srgb: bool,
    alpha_map_is_srgb: bool,
    matcap_map_is_srgb: bool,
    gradient_map_is_srgb: bool,
    metallic_roughness_texture_is_srgb: bool,
    emissive_map_is_srgb: bool,
    ao_map_is_srgb: bool,
    light_map_is_srgb: bool,
    specular_map_is_srgb: bool,
    common: CommonTextureInputs,
}

fn prepare_surface_texture_inputs(
    mesh: &SceneMesh,
    mesh_index: usize,
) -> Result<SurfaceTextureInputs> {
    thread::scope(|scope| {
        let texture = has_texture_data(mesh.texture.as_deref()).then(|| {
            scope.spawn(|| {
                decode_optional_texture_with_sampling(
                    mesh.texture.as_deref(),
                    mesh.texture_width,
                    mesh.texture_height,
                    mesh_index,
                    mesh.texture_wrap_s.as_deref(),
                    mesh.texture_wrap_t.as_deref(),
                    mesh.texture_mag_filter.as_deref(),
                    mesh.texture_min_filter.as_deref(),
                    mesh.texture_anisotropy,
                    &format!("scene.meshes[{mesh_index}].textureAnisotropy"),
                )
            })
        });
        let alpha_map = has_texture_data(mesh.alpha_map.as_deref()).then(|| {
            scope.spawn(|| {
                decode_optional_texture_with_sampling(
                    mesh.alpha_map.as_deref(),
                    mesh.alpha_map_width,
                    mesh.alpha_map_height,
                    mesh_index,
                    mesh.alpha_map_wrap_s.as_deref(),
                    mesh.alpha_map_wrap_t.as_deref(),
                    mesh.alpha_map_mag_filter.as_deref(),
                    mesh.alpha_map_min_filter.as_deref(),
                    mesh.alpha_map_anisotropy,
                    &format!("scene.meshes[{mesh_index}].alphaMapAnisotropy"),
                )
            })
        });

        let texture = match texture {
            Some(texture) => join_texture_worker(texture, "base texture worker")?,
            None => None,
        };

        let texture_transform =
            parse_texture_transform(mesh.texture_transform.as_deref(), mesh_index)?;
        let alpha_map_transform =
            parse_texture_transform(mesh.alpha_map_transform.as_deref(), mesh_index)?;
        let normal_map_transform =
            parse_texture_transform(mesh.normal_map_transform.as_deref(), mesh_index)?;
        let bump_map_transform =
            parse_texture_transform(mesh.bump_map_transform.as_deref(), mesh_index)?;
        let matcap_map_transform =
            parse_texture_transform(mesh.matcap_map_transform.as_deref(), mesh_index)?;
        let displacement_map_transform =
            parse_texture_transform(mesh.displacement_map_transform.as_deref(), mesh_index)?;
        let metallic_roughness_texture_transform = parse_texture_transform(
            mesh.metallic_roughness_texture_transform.as_deref(),
            mesh_index,
        )?;
        let emissive_map_transform =
            parse_texture_transform(mesh.emissive_map_transform.as_deref(), mesh_index)?;
        let ao_map_transform =
            parse_texture_transform(mesh.ao_map_transform.as_deref(), mesh_index)?;
        let light_map_transform =
            parse_texture_transform(mesh.light_map_transform.as_deref(), mesh_index)?;
        let specular_map_transform =
            parse_texture_transform(mesh.specular_map_transform.as_deref(), mesh_index)?;
        let specular_color_map_transform =
            parse_texture_transform(mesh.specular_color_map_transform.as_deref(), mesh_index)?;
        let specular_intensity_map_transform =
            parse_texture_transform(mesh.specular_intensity_map_transform.as_deref(), mesh_index)?;
        let clearcoat_map_transform =
            parse_texture_transform(mesh.clearcoat_map_transform.as_deref(), mesh_index)?;
        let clearcoat_roughness_map_transform = parse_texture_transform(
            mesh.clearcoat_roughness_map_transform.as_deref(),
            mesh_index,
        )?;
        let clearcoat_normal_map_transform =
            parse_texture_transform(mesh.clearcoat_normal_map_transform.as_deref(), mesh_index)?;
        let sheen_color_map_transform =
            parse_texture_transform(mesh.sheen_color_map_transform.as_deref(), mesh_index)?;
        let sheen_roughness_map_transform =
            parse_texture_transform(mesh.sheen_roughness_map_transform.as_deref(), mesh_index)?;
        let anisotropy_map_transform =
            parse_texture_transform(mesh.anisotropy_map_transform.as_deref(), mesh_index)?;
        let iridescence_map_transform =
            parse_texture_transform(mesh.iridescence_map_transform.as_deref(), mesh_index)?;
        let iridescence_thickness_map_transform = parse_texture_transform(
            mesh.iridescence_thickness_map_transform.as_deref(),
            mesh_index,
        )?;
        let transmission_map_transform =
            parse_texture_transform(mesh.transmission_map_transform.as_deref(), mesh_index)?;
        let thickness_map_transform =
            parse_texture_transform(mesh.thickness_map_transform.as_deref(), mesh_index)?;
        let texture_is_srgb = matches!(mesh.texture_color_space.as_deref(), Some("srgb"));
        let alpha_map_is_srgb = matches!(mesh.alpha_map_color_space.as_deref(), Some("srgb"));
        let matcap_map_is_srgb = matches!(mesh.matcap_map_color_space.as_deref(), Some("srgb"));
        let gradient_map_is_srgb = matches!(mesh.gradient_map_color_space.as_deref(), Some("srgb"));
        let metallic_roughness_texture_is_srgb = matches!(
            mesh.metallic_roughness_texture_color_space.as_deref(),
            Some("srgb")
        );
        let emissive_map_is_srgb = matches!(mesh.emissive_map_color_space.as_deref(), Some("srgb"));
        let ao_map_is_srgb = matches!(mesh.ao_map_color_space.as_deref(), Some("srgb"));
        let light_map_is_srgb = matches!(mesh.light_map_color_space.as_deref(), Some("srgb"));
        let specular_map_is_srgb = matches!(mesh.specular_map_color_space.as_deref(), Some("srgb"));
        let displacement_map_is_srgb =
            matches!(mesh.displacement_map_color_space.as_deref(), Some("srgb"));

        let mut common = prepare_common_texture_inputs(mesh, mesh_index)?;
        common.alpha_map = match alpha_map {
            Some(alpha_map) => join_texture_worker(alpha_map, "alpha map worker")?,
            None => None,
        };

        Ok(SurfaceTextureInputs {
            texture,
            texture_transform,
            alpha_map_transform,
            normal_map_transform,
            bump_map_transform,
            matcap_map_transform,
            displacement_map_transform,
            displacement_map_is_srgb,
            metallic_roughness_texture_transform,
            emissive_map_transform,
            ao_map_transform,
            light_map_transform,
            specular_map_transform,
            specular_color_map_transform,
            specular_intensity_map_transform,
            clearcoat_map_transform,
            clearcoat_roughness_map_transform,
            clearcoat_normal_map_transform,
            sheen_color_map_transform,
            sheen_roughness_map_transform,
            anisotropy_map_transform,
            iridescence_map_transform,
            iridescence_thickness_map_transform,
            transmission_map_transform,
            thickness_map_transform,
            texture_is_srgb,
            alpha_map_is_srgb,
            matcap_map_is_srgb,
            gradient_map_is_srgb,
            metallic_roughness_texture_is_srgb,
            emissive_map_is_srgb,
            ao_map_is_srgb,
            light_map_is_srgb,
            specular_map_is_srgb,
            common,
        })
    })
}

fn has_texture_data(data: Option<&[u8]>) -> bool {
    data.is_some_and(|data| !data.is_empty())
}

#[derive(Default)]
struct CommonTextureInputs {
    normal_map: Option<PreparedTexture>,
    normal_map_type: NormalMapType,
    normal_scale: [f32; 2],
    bump_map: Option<PreparedTexture>,
    bump_scale: f32,
    displacement_map: Option<PreparedTexture>,
    displacement_scale: f32,
    displacement_bias: f32,
    gradient_map: Option<PreparedTexture>,
    matcap_map: Option<PreparedTexture>,
    metallic_roughness_texture: Option<PreparedTexture>,
    specular_map: Option<PreparedTexture>,
    emissive_map: Option<PreparedTexture>,
    ao_map: Option<PreparedTexture>,
    ao_map_intensity: f32,
    light_map: Option<PreparedTexture>,
    light_map_intensity: f32,
    alpha_map: Option<PreparedTexture>,
}

fn prepare_common_texture_inputs(
    mesh: &SceneMesh,
    mesh_index: usize,
) -> Result<CommonTextureInputs> {
    if !has_common_texture_input(mesh) {
        return Ok(CommonTextureInputs {
            normal_map_type: NormalMapType::from_str_opt(
                mesh.normal_map_type.as_deref(),
                mesh_index,
            )?,
            normal_scale: parse_optional_vec2(
                mesh.normal_scale.as_deref(),
                [1.0, 1.0],
                "mesh normalScale",
            )?,
            bump_scale: finite_f32(mesh.bump_scale.unwrap_or(1.0), "mesh bumpScale")?,
            displacement_scale: finite_f32(
                mesh.displacement_scale.unwrap_or(1.0),
                "mesh displacementScale",
            )?,
            displacement_bias: finite_f32(
                mesh.displacement_bias.unwrap_or(0.0),
                "mesh displacementBias",
            )?,
            ao_map_intensity: finite_clamp01(
                mesh.ao_map_intensity.unwrap_or(1.0),
                "mesh aoMapIntensity",
            )?,
            light_map_intensity: finite_f32(
                mesh.light_map_intensity.unwrap_or(1.0),
                "mesh lightMapIntensity",
            )?
            .max(0.0),
            ..CommonTextureInputs::default()
        });
    }

    thread::scope(|scope| {
        let normal_map = scope.spawn(|| {
            decode_optional_texture_with_sampling(
                mesh.normal_map.as_deref(),
                mesh.normal_map_width,
                mesh.normal_map_height,
                mesh_index,
                mesh.normal_map_wrap_s.as_deref(),
                mesh.normal_map_wrap_t.as_deref(),
                mesh.normal_map_mag_filter.as_deref(),
                mesh.normal_map_min_filter.as_deref(),
                mesh.normal_map_anisotropy,
                &format!("scene.meshes[{mesh_index}].normalMapAnisotropy"),
            )
        });
        let bump_map = scope.spawn(|| {
            decode_optional_texture_with_sampling(
                mesh.bump_map.as_deref(),
                mesh.bump_map_width,
                mesh.bump_map_height,
                mesh_index,
                mesh.bump_map_wrap_s.as_deref(),
                mesh.bump_map_wrap_t.as_deref(),
                mesh.bump_map_mag_filter.as_deref(),
                mesh.bump_map_min_filter.as_deref(),
                mesh.bump_map_anisotropy,
                &format!("scene.meshes[{mesh_index}].bumpMapAnisotropy"),
            )
        });
        let displacement_map = scope.spawn(|| {
            decode_optional_texture_with_sampling(
                mesh.displacement_map.as_deref(),
                mesh.displacement_map_width,
                mesh.displacement_map_height,
                mesh_index,
                mesh.displacement_map_wrap_s.as_deref(),
                mesh.displacement_map_wrap_t.as_deref(),
                mesh.displacement_map_mag_filter.as_deref(),
                mesh.displacement_map_min_filter.as_deref(),
                mesh.displacement_map_anisotropy,
                &format!("scene.meshes[{mesh_index}].displacementMapAnisotropy"),
            )
        });
        let gradient_map = scope.spawn(|| {
            decode_optional_texture_with_sampling(
                mesh.gradient_map.as_deref(),
                mesh.gradient_map_width,
                mesh.gradient_map_height,
                mesh_index,
                mesh.gradient_map_wrap_s.as_deref(),
                mesh.gradient_map_wrap_t.as_deref(),
                mesh.gradient_map_mag_filter.as_deref(),
                mesh.gradient_map_min_filter.as_deref(),
                mesh.gradient_map_anisotropy,
                &format!("scene.meshes[{mesh_index}].gradientMapAnisotropy"),
            )
        });
        let matcap_map = scope.spawn(|| {
            decode_optional_texture_with_sampling(
                mesh.matcap_map.as_deref(),
                mesh.matcap_map_width,
                mesh.matcap_map_height,
                mesh_index,
                mesh.matcap_map_wrap_s.as_deref(),
                mesh.matcap_map_wrap_t.as_deref(),
                mesh.matcap_map_mag_filter.as_deref(),
                mesh.matcap_map_min_filter.as_deref(),
                mesh.matcap_map_anisotropy,
                &format!("scene.meshes[{mesh_index}].matcapMapAnisotropy"),
            )
        });
        let metallic_roughness_texture = scope.spawn(|| {
            decode_optional_texture_with_sampling(
                mesh.metallic_roughness_texture.as_deref(),
                mesh.metallic_roughness_texture_width,
                mesh.metallic_roughness_texture_height,
                mesh_index,
                mesh.metallic_roughness_texture_wrap_s.as_deref(),
                mesh.metallic_roughness_texture_wrap_t.as_deref(),
                mesh.metallic_roughness_texture_mag_filter.as_deref(),
                mesh.metallic_roughness_texture_min_filter.as_deref(),
                mesh.metallic_roughness_texture_anisotropy,
                &format!("scene.meshes[{mesh_index}].metallicRoughnessTextureAnisotropy"),
            )
        });
        let specular_map = scope.spawn(|| {
            decode_optional_texture_with_sampling(
                mesh.specular_map.as_deref(),
                mesh.specular_map_width,
                mesh.specular_map_height,
                mesh_index,
                mesh.specular_map_wrap_s.as_deref(),
                mesh.specular_map_wrap_t.as_deref(),
                mesh.specular_map_mag_filter.as_deref(),
                mesh.specular_map_min_filter.as_deref(),
                mesh.specular_map_anisotropy,
                &format!("scene.meshes[{mesh_index}].specularMapAnisotropy"),
            )
        });
        let emissive_map = scope.spawn(|| {
            decode_optional_texture_with_sampling(
                mesh.emissive_map.as_deref(),
                mesh.emissive_map_width,
                mesh.emissive_map_height,
                mesh_index,
                mesh.emissive_map_wrap_s.as_deref(),
                mesh.emissive_map_wrap_t.as_deref(),
                mesh.emissive_map_mag_filter.as_deref(),
                mesh.emissive_map_min_filter.as_deref(),
                mesh.emissive_map_anisotropy,
                &format!("scene.meshes[{mesh_index}].emissiveMapAnisotropy"),
            )
        });
        let ao_map = scope.spawn(|| {
            decode_optional_texture_with_sampling(
                mesh.ao_map.as_deref(),
                mesh.ao_map_width,
                mesh.ao_map_height,
                mesh_index,
                mesh.ao_map_wrap_s.as_deref(),
                mesh.ao_map_wrap_t.as_deref(),
                mesh.ao_map_mag_filter.as_deref(),
                mesh.ao_map_min_filter.as_deref(),
                mesh.ao_map_anisotropy,
                &format!("scene.meshes[{mesh_index}].aoMapAnisotropy"),
            )
        });
        let light_map = scope.spawn(|| {
            decode_optional_texture_with_sampling(
                mesh.light_map.as_deref(),
                mesh.light_map_width,
                mesh.light_map_height,
                mesh_index,
                mesh.light_map_wrap_s.as_deref(),
                mesh.light_map_wrap_t.as_deref(),
                mesh.light_map_mag_filter.as_deref(),
                mesh.light_map_min_filter.as_deref(),
                mesh.light_map_anisotropy,
                &format!("scene.meshes[{mesh_index}].lightMapAnisotropy"),
            )
        });
        let mut normal_map = join_texture_worker(normal_map, "normal map worker")?;
        if matches!(mesh.normal_map_color_space.as_deref(), Some("srgb")) {
            if let Some(texture) = normal_map.as_mut() {
                decode_texture_rgb_srgb_to_linear(texture);
            }
        }
        let normal_map_type =
            NormalMapType::from_str_opt(mesh.normal_map_type.as_deref(), mesh_index)?;
        let normal_scale =
            parse_optional_vec2(mesh.normal_scale.as_deref(), [1.0, 1.0], "mesh normalScale")?;
        let mut bump_map = join_texture_worker(bump_map, "bump map worker")?;
        if matches!(mesh.bump_map_color_space.as_deref(), Some("srgb")) {
            if let Some(texture) = bump_map.as_mut() {
                decode_texture_rgb_srgb_to_linear(texture);
            }
        }
        let bump_scale = finite_f32(mesh.bump_scale.unwrap_or(1.0), "mesh bumpScale")?;
        let displacement_map = join_texture_worker(displacement_map, "displacement map worker")?;
        let displacement_scale = finite_f32(
            mesh.displacement_scale.unwrap_or(1.0),
            "mesh displacementScale",
        )?;
        let displacement_bias = finite_f32(
            mesh.displacement_bias.unwrap_or(0.0),
            "mesh displacementBias",
        )?;
        let gradient_map = join_texture_worker(gradient_map, "gradient map worker")?;
        let matcap_map = join_texture_worker(matcap_map, "matcap map worker")?;
        let metallic_roughness_texture = join_texture_worker(
            metallic_roughness_texture,
            "metallic roughness texture worker",
        )?;
        let specular_map = join_texture_worker(specular_map, "specular map worker")?;
        let emissive_map = join_texture_worker(emissive_map, "emissive map worker")?;
        let ao_map = join_texture_worker(ao_map, "ao map worker")?;
        let ao_map_intensity =
            finite_clamp01(mesh.ao_map_intensity.unwrap_or(1.0), "mesh aoMapIntensity")?;
        let light_map = join_texture_worker(light_map, "light map worker")?;
        let light_map_intensity = finite_f32(
            mesh.light_map_intensity.unwrap_or(1.0),
            "mesh lightMapIntensity",
        )?
        .max(0.0);

        Ok(CommonTextureInputs {
            normal_map,
            normal_map_type,
            normal_scale,
            bump_map,
            bump_scale,
            displacement_map,
            displacement_scale,
            displacement_bias,
            gradient_map,
            matcap_map,
            metallic_roughness_texture,
            specular_map,
            emissive_map,
            ao_map,
            ao_map_intensity,
            light_map,
            light_map_intensity,
            alpha_map: None,
        })
    })
}

fn has_common_texture_input(mesh: &SceneMesh) -> bool {
    [
        mesh.normal_map.as_deref(),
        mesh.bump_map.as_deref(),
        mesh.displacement_map.as_deref(),
        mesh.gradient_map.as_deref(),
        mesh.matcap_map.as_deref(),
        mesh.metallic_roughness_texture.as_deref(),
        mesh.specular_map.as_deref(),
        mesh.emissive_map.as_deref(),
        mesh.ao_map.as_deref(),
        mesh.light_map.as_deref(),
    ]
    .into_iter()
    .any(has_texture_data)
}

#[derive(Default)]
struct PhysicalTextureInputs {
    clearcoat_map: Option<PreparedTexture>,
    clearcoat_roughness_map: Option<PreparedTexture>,
    clearcoat_normal_map: Option<PreparedTexture>,
    sheen_color_map: Option<PreparedTexture>,
    sheen_roughness_map: Option<PreparedTexture>,
    anisotropy_map: Option<PreparedTexture>,
    iridescence_map: Option<PreparedTexture>,
    iridescence_thickness_map: Option<PreparedTexture>,
    transmission_map: Option<PreparedTexture>,
    thickness_map: Option<PreparedTexture>,
    specular_color_map: Option<PreparedTexture>,
    specular_intensity_map: Option<PreparedTexture>,
}

fn prepare_physical_texture_inputs(
    mesh: &SceneMesh,
    mesh_index: usize,
) -> Result<PhysicalTextureInputs> {
    if !has_physical_texture_input(mesh) {
        return Ok(PhysicalTextureInputs::default());
    }

    thread::scope(|scope| {
        let clearcoat_map = scope.spawn(|| {
            decode_optional_texture_with_sampling(
                mesh.clearcoat_map.as_deref(),
                mesh.clearcoat_map_width,
                mesh.clearcoat_map_height,
                mesh_index,
                mesh.clearcoat_map_wrap_s.as_deref(),
                mesh.clearcoat_map_wrap_t.as_deref(),
                mesh.clearcoat_map_mag_filter.as_deref(),
                mesh.clearcoat_map_min_filter.as_deref(),
                mesh.clearcoat_map_anisotropy,
                &format!("scene.meshes[{mesh_index}].clearcoatMapAnisotropy"),
            )
        });
        let clearcoat_roughness_map = scope.spawn(|| {
            decode_optional_texture_with_sampling(
                mesh.clearcoat_roughness_map.as_deref(),
                mesh.clearcoat_roughness_map_width,
                mesh.clearcoat_roughness_map_height,
                mesh_index,
                mesh.clearcoat_roughness_map_wrap_s.as_deref(),
                mesh.clearcoat_roughness_map_wrap_t.as_deref(),
                mesh.clearcoat_roughness_map_mag_filter.as_deref(),
                mesh.clearcoat_roughness_map_min_filter.as_deref(),
                mesh.clearcoat_roughness_map_anisotropy,
                &format!("scene.meshes[{mesh_index}].clearcoatRoughnessMapAnisotropy"),
            )
        });
        let clearcoat_normal_map = scope.spawn(|| {
            decode_optional_texture_with_sampling(
                mesh.clearcoat_normal_map.as_deref(),
                mesh.clearcoat_normal_map_width,
                mesh.clearcoat_normal_map_height,
                mesh_index,
                mesh.clearcoat_normal_map_wrap_s.as_deref(),
                mesh.clearcoat_normal_map_wrap_t.as_deref(),
                mesh.clearcoat_normal_map_mag_filter.as_deref(),
                mesh.clearcoat_normal_map_min_filter.as_deref(),
                mesh.clearcoat_normal_map_anisotropy,
                &format!("scene.meshes[{mesh_index}].clearcoatNormalMapAnisotropy"),
            )
        });
        let sheen_color_map = scope.spawn(|| {
            decode_optional_texture_with_sampling(
                mesh.sheen_color_map.as_deref(),
                mesh.sheen_color_map_width,
                mesh.sheen_color_map_height,
                mesh_index,
                mesh.sheen_color_map_wrap_s.as_deref(),
                mesh.sheen_color_map_wrap_t.as_deref(),
                mesh.sheen_color_map_mag_filter.as_deref(),
                mesh.sheen_color_map_min_filter.as_deref(),
                mesh.sheen_color_map_anisotropy,
                &format!("scene.meshes[{mesh_index}].sheenColorMapAnisotropy"),
            )
        });
        let sheen_roughness_map = scope.spawn(|| {
            decode_optional_texture_with_sampling(
                mesh.sheen_roughness_map.as_deref(),
                mesh.sheen_roughness_map_width,
                mesh.sheen_roughness_map_height,
                mesh_index,
                mesh.sheen_roughness_map_wrap_s.as_deref(),
                mesh.sheen_roughness_map_wrap_t.as_deref(),
                mesh.sheen_roughness_map_mag_filter.as_deref(),
                mesh.sheen_roughness_map_min_filter.as_deref(),
                mesh.sheen_roughness_map_anisotropy,
                &format!("scene.meshes[{mesh_index}].sheenRoughnessMapAnisotropy"),
            )
        });
        let anisotropy_map = scope.spawn(|| {
            decode_optional_texture_with_sampling(
                mesh.anisotropy_map.as_deref(),
                mesh.anisotropy_map_width,
                mesh.anisotropy_map_height,
                mesh_index,
                mesh.anisotropy_map_wrap_s.as_deref(),
                mesh.anisotropy_map_wrap_t.as_deref(),
                mesh.anisotropy_map_mag_filter.as_deref(),
                mesh.anisotropy_map_min_filter.as_deref(),
                mesh.anisotropy_map_anisotropy,
                &format!("scene.meshes[{mesh_index}].anisotropyMapAnisotropy"),
            )
        });
        let iridescence_map = scope.spawn(|| {
            decode_optional_texture_with_sampling(
                mesh.iridescence_map.as_deref(),
                mesh.iridescence_map_width,
                mesh.iridescence_map_height,
                mesh_index,
                mesh.iridescence_map_wrap_s.as_deref(),
                mesh.iridescence_map_wrap_t.as_deref(),
                mesh.iridescence_map_mag_filter.as_deref(),
                mesh.iridescence_map_min_filter.as_deref(),
                mesh.iridescence_map_anisotropy,
                &format!("scene.meshes[{mesh_index}].iridescenceMapAnisotropy"),
            )
        });
        let iridescence_thickness_map = scope.spawn(|| {
            decode_optional_texture_with_sampling(
                mesh.iridescence_thickness_map.as_deref(),
                mesh.iridescence_thickness_map_width,
                mesh.iridescence_thickness_map_height,
                mesh_index,
                mesh.iridescence_thickness_map_wrap_s.as_deref(),
                mesh.iridescence_thickness_map_wrap_t.as_deref(),
                mesh.iridescence_thickness_map_mag_filter.as_deref(),
                mesh.iridescence_thickness_map_min_filter.as_deref(),
                mesh.iridescence_thickness_map_anisotropy,
                &format!("scene.meshes[{mesh_index}].iridescenceThicknessMapAnisotropy"),
            )
        });
        let transmission_map = scope.spawn(|| {
            decode_optional_texture_with_sampling(
                mesh.transmission_map.as_deref(),
                mesh.transmission_map_width,
                mesh.transmission_map_height,
                mesh_index,
                mesh.transmission_map_wrap_s.as_deref(),
                mesh.transmission_map_wrap_t.as_deref(),
                mesh.transmission_map_mag_filter.as_deref(),
                mesh.transmission_map_min_filter.as_deref(),
                mesh.transmission_map_anisotropy,
                &format!("scene.meshes[{mesh_index}].transmissionMapAnisotropy"),
            )
        });
        let thickness_map = scope.spawn(|| {
            decode_optional_texture_with_sampling(
                mesh.thickness_map.as_deref(),
                mesh.thickness_map_width,
                mesh.thickness_map_height,
                mesh_index,
                mesh.thickness_map_wrap_s.as_deref(),
                mesh.thickness_map_wrap_t.as_deref(),
                mesh.thickness_map_mag_filter.as_deref(),
                mesh.thickness_map_min_filter.as_deref(),
                mesh.thickness_map_anisotropy,
                &format!("scene.meshes[{mesh_index}].thicknessMapAnisotropy"),
            )
        });
        let specular_color_map = scope.spawn(|| {
            decode_optional_texture_with_sampling(
                mesh.specular_color_map.as_deref(),
                mesh.specular_color_map_width,
                mesh.specular_color_map_height,
                mesh_index,
                mesh.specular_color_map_wrap_s.as_deref(),
                mesh.specular_color_map_wrap_t.as_deref(),
                mesh.specular_color_map_mag_filter.as_deref(),
                mesh.specular_color_map_min_filter.as_deref(),
                mesh.specular_color_map_anisotropy,
                &format!("scene.meshes[{mesh_index}].specularColorMapAnisotropy"),
            )
        });
        let specular_intensity_map = scope.spawn(|| {
            decode_optional_texture_with_sampling(
                mesh.specular_intensity_map.as_deref(),
                mesh.specular_intensity_map_width,
                mesh.specular_intensity_map_height,
                mesh_index,
                mesh.specular_intensity_map_wrap_s.as_deref(),
                mesh.specular_intensity_map_wrap_t.as_deref(),
                mesh.specular_intensity_map_mag_filter.as_deref(),
                mesh.specular_intensity_map_min_filter.as_deref(),
                mesh.specular_intensity_map_anisotropy,
                &format!("scene.meshes[{mesh_index}].specularIntensityMapAnisotropy"),
            )
        });

        Ok(PhysicalTextureInputs {
            clearcoat_map: join_texture_worker(clearcoat_map, "clearcoat map worker")?,
            clearcoat_roughness_map: join_texture_worker(
                clearcoat_roughness_map,
                "clearcoat roughness map worker",
            )?,
            clearcoat_normal_map: {
                let mut texture =
                    join_texture_worker(clearcoat_normal_map, "clearcoat normal map worker")?;
                if matches!(
                    mesh.clearcoat_normal_map_color_space.as_deref(),
                    Some("srgb")
                ) {
                    if let Some(texture) = texture.as_mut() {
                        decode_texture_rgb_srgb_to_linear(texture);
                    }
                }
                texture
            },
            sheen_color_map: join_texture_worker(sheen_color_map, "sheen color map worker")?,
            sheen_roughness_map: join_texture_worker(
                sheen_roughness_map,
                "sheen roughness map worker",
            )?,
            anisotropy_map: join_texture_worker(anisotropy_map, "anisotropy map worker")?,
            iridescence_map: join_texture_worker(iridescence_map, "iridescence map worker")?,
            iridescence_thickness_map: join_texture_worker(
                iridescence_thickness_map,
                "iridescence thickness map worker",
            )?,
            transmission_map: join_texture_worker(transmission_map, "transmission map worker")?,
            thickness_map: join_texture_worker(thickness_map, "thickness map worker")?,
            specular_color_map: join_texture_worker(
                specular_color_map,
                "specular color map worker",
            )?,
            specular_intensity_map: join_texture_worker(
                specular_intensity_map,
                "specular intensity map worker",
            )?,
        })
    })
}

fn has_physical_texture_input(mesh: &SceneMesh) -> bool {
    [
        mesh.clearcoat_map.as_deref(),
        mesh.clearcoat_roughness_map.as_deref(),
        mesh.clearcoat_normal_map.as_deref(),
        mesh.sheen_color_map.as_deref(),
        mesh.sheen_roughness_map.as_deref(),
        mesh.anisotropy_map.as_deref(),
        mesh.iridescence_map.as_deref(),
        mesh.iridescence_thickness_map.as_deref(),
        mesh.transmission_map.as_deref(),
        mesh.thickness_map.as_deref(),
        mesh.specular_color_map.as_deref(),
        mesh.specular_intensity_map.as_deref(),
    ]
    .into_iter()
    .any(|data| data.is_some_and(|data| !data.is_empty()))
}

fn join_texture_worker<T>(
    handle: thread::ScopedJoinHandle<'_, Result<T>>,
    label: &str,
) -> Result<T> {
    handle.join().unwrap_or_else(|_| panic!("{label} panicked"))
}

struct PhysicalMapInputs<'a> {
    clearcoat: Option<&'a PreparedTexture>,
    clearcoat_roughness: Option<&'a PreparedTexture>,
    sheen_color: Option<&'a PreparedTexture>,
    sheen_roughness: Option<&'a PreparedTexture>,
    anisotropy: Option<&'a PreparedTexture>,
    iridescence: Option<&'a PreparedTexture>,
    iridescence_thickness: Option<&'a PreparedTexture>,
    transmission: Option<&'a PreparedTexture>,
    thickness: Option<&'a PreparedTexture>,
    specular_color: Option<&'a PreparedTexture>,
    specular_intensity: Option<&'a PreparedTexture>,
    clearcoat_is_srgb: bool,
    clearcoat_roughness_is_srgb: bool,
    anisotropy_is_srgb: bool,
    iridescence_is_srgb: bool,
    iridescence_thickness_is_srgb: bool,
    transmission_is_srgb: bool,
    thickness_is_srgb: bool,
    sheen_color_is_srgb: bool,
    specular_color_is_srgb: bool,
}

fn pack_physical_maps(inputs: PhysicalMapInputs<'_>) -> Option<PreparedPhysicalMaps> {
    let maps = [
        inputs.clearcoat,
        inputs.clearcoat_roughness,
        inputs.sheen_color,
        inputs.sheen_roughness,
        inputs.anisotropy,
        inputs.iridescence,
        inputs.iridescence_thickness,
        inputs.transmission,
        inputs.thickness,
        inputs.specular_color,
        inputs.specular_intensity,
    ];
    if maps.iter().all(|map| map.is_none()) {
        return None;
    }

    let width = maps
        .iter()
        .flatten()
        .map(|map| map.width)
        .max()
        .unwrap_or(1);
    let height = maps
        .iter()
        .flatten()
        .map(|map| map.height)
        .max()
        .unwrap_or(1);
    let pixel_count = (width * height) as usize;
    let mut scalar = vec![255u8; pixel_count * 4];
    let mut sheen = vec![255u8; pixel_count * 4];
    let mut specular = vec![255u8; pixel_count * 4];
    let mut iridescence = vec![255u8; pixel_count * 4];
    // Default anisotropy map is direction +X, full strength.
    let mut anisotropy = vec![0u8; pixel_count * 4];
    for px in 0..pixel_count {
        anisotropy[px * 4] = 255;
        anisotropy[px * 4 + 1] = 128;
        anisotropy[px * 4 + 2] = 255;
        anisotropy[px * 4 + 3] = 255;
    }

    for y in 0..height {
        for x in 0..width {
            let out = ((y * width + x) * 4) as usize;
            if let Some(map) = inputs.clearcoat {
                scalar[out] = sample_texture_color_channel(
                    map,
                    x,
                    y,
                    width,
                    height,
                    0,
                    inputs.clearcoat_is_srgb,
                );
            }
            if let Some(map) = inputs.clearcoat_roughness {
                scalar[out + 1] = sample_texture_color_channel(
                    map,
                    x,
                    y,
                    width,
                    height,
                    1,
                    inputs.clearcoat_roughness_is_srgb,
                );
            }
            if let Some(map) = inputs.transmission {
                scalar[out + 2] = sample_texture_color_channel(
                    map,
                    x,
                    y,
                    width,
                    height,
                    0,
                    inputs.transmission_is_srgb,
                );
            }
            if let Some(map) = inputs.thickness {
                scalar[out + 3] = sample_texture_color_channel(
                    map,
                    x,
                    y,
                    width,
                    height,
                    1,
                    inputs.thickness_is_srgb,
                );
            }
            if let Some(map) = inputs.sheen_color {
                sheen[out] = sample_texture_color_channel(
                    map,
                    x,
                    y,
                    width,
                    height,
                    0,
                    inputs.sheen_color_is_srgb,
                );
                sheen[out + 1] = sample_texture_color_channel(
                    map,
                    x,
                    y,
                    width,
                    height,
                    1,
                    inputs.sheen_color_is_srgb,
                );
                sheen[out + 2] = sample_texture_color_channel(
                    map,
                    x,
                    y,
                    width,
                    height,
                    2,
                    inputs.sheen_color_is_srgb,
                );
            }
            if let Some(map) = inputs.sheen_roughness {
                sheen[out + 3] = sample_texture_channel(map, x, y, width, height, 3);
            }
            if let Some(map) = inputs.anisotropy {
                anisotropy[out] = sample_texture_color_channel(
                    map,
                    x,
                    y,
                    width,
                    height,
                    0,
                    inputs.anisotropy_is_srgb,
                );
                anisotropy[out + 1] = sample_texture_color_channel(
                    map,
                    x,
                    y,
                    width,
                    height,
                    1,
                    inputs.anisotropy_is_srgb,
                );
                anisotropy[out + 2] = sample_texture_color_channel(
                    map,
                    x,
                    y,
                    width,
                    height,
                    2,
                    inputs.anisotropy_is_srgb,
                );
            }
            if let Some(map) = inputs.iridescence {
                iridescence[out] = sample_texture_color_channel(
                    map,
                    x,
                    y,
                    width,
                    height,
                    0,
                    inputs.iridescence_is_srgb,
                );
            }
            if let Some(map) = inputs.iridescence_thickness {
                iridescence[out + 1] = sample_texture_color_channel(
                    map,
                    x,
                    y,
                    width,
                    height,
                    1,
                    inputs.iridescence_thickness_is_srgb,
                );
            }
            if let Some(map) = inputs.specular_color {
                specular[out] = sample_texture_color_channel(
                    map,
                    x,
                    y,
                    width,
                    height,
                    0,
                    inputs.specular_color_is_srgb,
                );
                specular[out + 1] = sample_texture_color_channel(
                    map,
                    x,
                    y,
                    width,
                    height,
                    1,
                    inputs.specular_color_is_srgb,
                );
                specular[out + 2] = sample_texture_color_channel(
                    map,
                    x,
                    y,
                    width,
                    height,
                    2,
                    inputs.specular_color_is_srgb,
                );
            }
            if let Some(map) = inputs.specular_intensity {
                specular[out + 3] = sample_texture_channel(map, x, y, width, height, 3);
            }
        }
    }

    let physical_layers_sampler = TextureSamplerSettings::first_from_textures(&[
        inputs.clearcoat,
        inputs.clearcoat_roughness,
        inputs.transmission,
        inputs.thickness,
        inputs.anisotropy,
        inputs.iridescence,
        inputs.iridescence_thickness,
    ]);
    let sheen_sampler =
        TextureSamplerSettings::first_from_textures(&[inputs.sheen_color, inputs.sheen_roughness]);
    let specular_sampler = TextureSamplerSettings::first_from_textures(&[
        inputs.specular_color,
        inputs.specular_intensity,
    ]);
    Some(PreparedPhysicalMaps {
        scalar_map: packed_texture(scalar, width, height, physical_layers_sampler),
        sheen_map: packed_texture(sheen, width, height, sheen_sampler),
        anisotropy_map: packed_texture(anisotropy, width, height, physical_layers_sampler),
        specular_map: packed_texture(specular, width, height, specular_sampler),
        iridescence_map: packed_texture(iridescence, width, height, physical_layers_sampler),
        physical_layers_sampler,
        sheen_sampler,
        specular_sampler,
    })
}

fn packed_texture(
    rgba: Vec<u8>,
    width: u32,
    height: u32,
    sampler: TextureSamplerSettings,
) -> PreparedTexture {
    PreparedTexture {
        rgba,
        width,
        height,
        mipmaps: Vec::new(),
        wrap_s: sampler.wrap_s,
        wrap_t: sampler.wrap_t,
        mag_filter: sampler.mag_filter,
        min_filter: sampler.min_filter,
        mipmap_filter: sampler.mipmap_filter,
        anisotropy: sampler.anisotropy,
    }
}

fn sample_texture_channel(
    texture: &PreparedTexture,
    x: u32,
    y: u32,
    out_width: u32,
    out_height: u32,
    channel: usize,
) -> u8 {
    let sx = (((x as f32 + 0.5) / out_width as f32) * texture.width as f32)
        .floor()
        .clamp(0.0, (texture.width - 1) as f32) as u32;
    let sy = (((y as f32 + 0.5) / out_height as f32) * texture.height as f32)
        .floor()
        .clamp(0.0, (texture.height - 1) as f32) as u32;
    texture.rgba[((sy * texture.width + sx) * 4) as usize + channel]
}

fn sample_texture_color_channel(
    texture: &PreparedTexture,
    x: u32,
    y: u32,
    out_width: u32,
    out_height: u32,
    channel: usize,
    is_srgb: bool,
) -> u8 {
    let value = sample_texture_channel(texture, x, y, out_width, out_height, channel);
    if is_srgb && channel < 3 {
        srgb_u8_to_linear_u8(value)
    } else {
        value
    }
}

fn srgb_u8_to_linear_u8(value: u8) -> u8 {
    (srgb_u8_to_linear_f32(value).clamp(0.0, 1.0) * 255.0).round() as u8
}

fn srgb_u8_to_linear_f32(value: u8) -> f32 {
    let channel = value as f32 / 255.0;
    if channel <= 0.04045 {
        channel / 12.92
    } else {
        ((channel + 0.055) / 1.055).powf(2.4)
    }
}

fn decode_texture_rgb_srgb_to_linear(texture: &mut PreparedTexture) {
    for pixel in texture.rgba.chunks_exact_mut(4) {
        pixel[0] = srgb_u8_to_linear_u8(pixel[0]);
        pixel[1] = srgb_u8_to_linear_u8(pixel[1]);
        pixel[2] = srgb_u8_to_linear_u8(pixel[2]);
    }
    for mipmap in &mut texture.mipmaps {
        for pixel in mipmap.rgba.chunks_exact_mut(4) {
            pixel[0] = srgb_u8_to_linear_u8(pixel[0]);
            pixel[1] = srgb_u8_to_linear_u8(pixel[1]);
            pixel[2] = srgb_u8_to_linear_u8(pixel[2]);
        }
    }
}

fn apply_displacement_map(
    vertices: &mut [Vertex],
    texture: &PreparedTexture,
    transform: [f32; 6],
    is_srgb: bool,
    uses_uv2: bool,
    scale: f32,
    bias: f32,
) {
    if texture.width == 0 || texture.height == 0 || scale == 0.0 && bias == 0.0 {
        return;
    }

    for vertex in vertices {
        let normal = Vec3::from_array(vertex.normal).normalize_or_zero();
        if normal.length_squared() <= 0.0 {
            continue;
        }
        let source_uv = if uses_uv2 { vertex.uv2 } else { vertex.uv };
        let uv = transform_uv(source_uv, transform);
        let sample = sample_texture_color_channel_uv(texture, uv[0], uv[1], 0, is_srgb);
        let displacement = sample * scale + bias;
        let position = Vec3::from_array(vertex.position) + normal * displacement;
        vertex.position = position.to_array();
    }
}

fn transform_uv(uv: [f32; 2], transform: [f32; 6]) -> [f32; 2] {
    [
        transform[0] * uv[0] + transform[1] * uv[1] + transform[2],
        transform[3] * uv[0] + transform[4] * uv[1] + transform[5],
    ]
}

fn sample_texture_channel_uv(texture: &PreparedTexture, u: f32, v: f32, channel: usize) -> f32 {
    match texture.mag_filter {
        TextureFilter::Nearest => sample_texture_channel_nearest(texture, u, v, channel),
        TextureFilter::Linear => sample_texture_channel_linear(texture, u, v, channel),
    }
}

fn sample_texture_color_channel_uv(
    texture: &PreparedTexture,
    u: f32,
    v: f32,
    channel: usize,
    is_srgb: bool,
) -> f32 {
    if !is_srgb || channel >= 3 {
        return sample_texture_channel_uv(texture, u, v, channel);
    }

    match texture.mag_filter {
        TextureFilter::Nearest => sample_texture_color_channel_nearest(texture, u, v, channel),
        TextureFilter::Linear => sample_texture_color_channel_linear(texture, u, v, channel),
    }
}

fn sample_texture_channel_nearest(
    texture: &PreparedTexture,
    u: f32,
    v: f32,
    channel: usize,
) -> f32 {
    let x = wrapped_texel_index(
        (u * texture.width as f32).floor() as i32,
        texture.width,
        texture.wrap_s,
    );
    let y = wrapped_texel_index(
        (v * texture.height as f32).floor() as i32,
        texture.height,
        texture.wrap_t,
    );
    texture.rgba[((y * texture.width + x) * 4) as usize + channel] as f32 / 255.0
}

fn sample_texture_color_channel_nearest(
    texture: &PreparedTexture,
    u: f32,
    v: f32,
    channel: usize,
) -> f32 {
    let x = wrapped_texel_index(
        (u * texture.width as f32).floor() as i32,
        texture.width,
        texture.wrap_s,
    );
    let y = wrapped_texel_index(
        (v * texture.height as f32).floor() as i32,
        texture.height,
        texture.wrap_t,
    );
    let value = texture.rgba[((y * texture.width + x) * 4) as usize + channel];
    srgb_u8_to_linear_f32(value)
}

fn sample_texture_channel_linear(texture: &PreparedTexture, u: f32, v: f32, channel: usize) -> f32 {
    let x = u * texture.width as f32 - 0.5;
    let y = v * texture.height as f32 - 0.5;
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let tx = x - x0 as f32;
    let ty = y - y0 as f32;

    let s00 = texel_channel(texture, x0, y0, channel);
    let s10 = texel_channel(texture, x0 + 1, y0, channel);
    let s01 = texel_channel(texture, x0, y0 + 1, channel);
    let s11 = texel_channel(texture, x0 + 1, y0 + 1, channel);
    let sx0 = s00 * (1.0 - tx) + s10 * tx;
    let sx1 = s01 * (1.0 - tx) + s11 * tx;
    sx0 * (1.0 - ty) + sx1 * ty
}

fn sample_texture_color_channel_linear(
    texture: &PreparedTexture,
    u: f32,
    v: f32,
    channel: usize,
) -> f32 {
    let x = u * texture.width as f32 - 0.5;
    let y = v * texture.height as f32 - 0.5;
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let tx = x - x0 as f32;
    let ty = y - y0 as f32;

    let s00 = texel_color_channel(texture, x0, y0, channel);
    let s10 = texel_color_channel(texture, x0 + 1, y0, channel);
    let s01 = texel_color_channel(texture, x0, y0 + 1, channel);
    let s11 = texel_color_channel(texture, x0 + 1, y0 + 1, channel);
    let sx0 = s00 * (1.0 - tx) + s10 * tx;
    let sx1 = s01 * (1.0 - tx) + s11 * tx;
    sx0 * (1.0 - ty) + sx1 * ty
}

fn texel_channel(texture: &PreparedTexture, x: i32, y: i32, channel: usize) -> f32 {
    let tx = wrapped_texel_index(x, texture.width, texture.wrap_s);
    let ty = wrapped_texel_index(y, texture.height, texture.wrap_t);
    texture.rgba[((ty * texture.width + tx) * 4) as usize + channel] as f32 / 255.0
}

fn texel_color_channel(texture: &PreparedTexture, x: i32, y: i32, channel: usize) -> f32 {
    let tx = wrapped_texel_index(x, texture.width, texture.wrap_s);
    let ty = wrapped_texel_index(y, texture.height, texture.wrap_t);
    let value = texture.rgba[((ty * texture.width + tx) * 4) as usize + channel];
    srgb_u8_to_linear_f32(value)
}

fn wrapped_texel_index(index: i32, size: u32, wrap: WrapMode) -> u32 {
    let size_i = size as i32;
    match wrap {
        WrapMode::ClampToEdge => index.clamp(0, size_i - 1) as u32,
        WrapMode::Repeat => index.rem_euclid(size_i) as u32,
        WrapMode::MirrorRepeat => {
            let period = size_i * 2;
            let wrapped = index.rem_euclid(period);
            if wrapped >= size_i {
                (period - wrapped - 1) as u32
            } else {
                wrapped as u32
            }
        }
    }
}

enum ColorMode<'a> {
    Uniform([f32; 4]),
    RgbPerVertex(&'a [f64]),
    RgbaPerVertex(&'a [f64]),
}

impl<'a> ColorMode<'a> {
    fn new(
        colors: Option<&'a [f64]>,
        vertex_count: usize,
        _material_color: [f64; 4],
        mesh_index: usize,
    ) -> Result<Self> {
        let Some(colors) = colors else {
            return Ok(Self::Uniform([1.0, 1.0, 1.0, 1.0]));
        };

        match colors.len() {
            3 | 4 => Ok(Self::Uniform(color_to_f32(parse_color(
                Some(colors),
                [1.0, 1.0, 1.0, 1.0],
                &format!("scene.meshes[{mesh_index}].colors"),
            )?))),
            len if len == vertex_count * 3 => {
                validate_color_values(colors, &format!("scene.meshes[{mesh_index}].colors"))?;
                Ok(Self::RgbPerVertex(colors))
            }
            len if len == vertex_count * 4 => {
                validate_color_values(colors, &format!("scene.meshes[{mesh_index}].colors"))?;
                Ok(Self::RgbaPerVertex(colors))
            }
            len => bail!(
                "scene.meshes[{mesh_index}].colors has length {len}; expected 3, 4, vertex_count * 3, or vertex_count * 4"
            ),
        }
    }

    fn color(&self, vertex_index: usize) -> [f32; 4] {
        match self {
            Self::Uniform(color) => *color,
            Self::RgbPerVertex(colors) => {
                let base = vertex_index * 3;
                [
                    clamp01(colors[base]) as f32,
                    clamp01(colors[base + 1]) as f32,
                    clamp01(colors[base + 2]) as f32,
                    1.0,
                ]
            }
            Self::RgbaPerVertex(colors) => {
                let base = vertex_index * 4;
                [
                    clamp01(colors[base]) as f32,
                    clamp01(colors[base + 1]) as f32,
                    clamp01(colors[base + 2]) as f32,
                    clamp01(colors[base + 3]) as f32,
                ]
            }
        }
    }
}

fn validate_color_values(values: &[f64], field: &str) -> Result<()> {
    for value in values {
        finite_color(*value, field)?;
    }
    Ok(())
}

fn compute_flat_normals(vertices: &mut [Vertex], indices: Option<&[u32]>) {
    match indices {
        Some(idx) => {
            for tri in idx.chunks_exact(3) {
                let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
                let p0 = Vec3::from(vertices[i0].position);
                let p1 = Vec3::from(vertices[i1].position);
                let p2 = Vec3::from(vertices[i2].position);
                let normal = (p1 - p0).cross(p2 - p0);
                let n = if normal.length_squared() > 0.0 {
                    normal.normalize()
                } else {
                    Vec3::Y
                };
                let arr = n.to_array();
                // Accumulate for shared vertices
                for &vi in &[i0, i1, i2] {
                    vertices[vi].normal[0] += arr[0];
                    vertices[vi].normal[1] += arr[1];
                    vertices[vi].normal[2] += arr[2];
                }
            }
            // Normalize accumulated
            for v in vertices.iter_mut() {
                let n = Vec3::from(v.normal);
                if n.length_squared() > 0.0 {
                    v.normal = n.normalize().to_array();
                } else {
                    v.normal = [0.0, 1.0, 0.0];
                }
            }
        }
        None => {
            for tri in vertices.chunks_exact_mut(3) {
                let p0 = Vec3::from(tri[0].position);
                let p1 = Vec3::from(tri[1].position);
                let p2 = Vec3::from(tri[2].position);
                let normal = (p1 - p0).cross(p2 - p0);
                let n = if normal.length_squared() > 0.0 {
                    normal.normalize()
                } else {
                    Vec3::Y
                };
                let arr = n.to_array();
                tri[0].normal = arr;
                tri[1].normal = arr;
                tri[2].normal = arr;
            }
        }
    }
}

/// Compute per-vertex tangents from positions, normals, and UVs.
/// Uses the standard MikkTSpace-like per-triangle method:
///   tangent = (dp1 * duv2.y - dp2 * duv1.y) / det
/// Tangent.w stores the handedness sign for the bitangent.
fn compute_tangents(vertices: &mut [Vertex], indices: Option<&[u32]>) {
    let vertex_count = vertices.len();
    let mut tan1 = vec![Vec3::ZERO; vertex_count];
    let mut tan2 = vec![Vec3::ZERO; vertex_count];

    let process_triangle =
        |i0: usize, i1: usize, i2: usize, tan1: &mut [Vec3], tan2: &mut [Vec3]| {
            let p0 = Vec3::from(vertices[i0].position);
            let p1 = Vec3::from(vertices[i1].position);
            let p2 = Vec3::from(vertices[i2].position);

            let uv0 = vertices[i0].uv;
            let uv1 = vertices[i1].uv;
            let uv2 = vertices[i2].uv;

            let dp1 = p1 - p0;
            let dp2 = p2 - p0;
            let duv1 = [uv1[0] - uv0[0], uv1[1] - uv0[1]];
            let duv2 = [uv2[0] - uv0[0], uv2[1] - uv0[1]];

            let det = duv1[0] * duv2[1] - duv1[1] * duv2[0];
            if det.abs() < 1e-8 {
                return;
            }
            let inv_det = 1.0 / det;

            let t = (dp1 * duv2[1] - dp2 * duv1[1]) * inv_det;
            let b = (dp2 * duv1[0] - dp1 * duv2[0]) * inv_det;

            tan1[i0] += t;
            tan1[i1] += t;
            tan1[i2] += t;
            tan2[i0] += b;
            tan2[i1] += b;
            tan2[i2] += b;
        };

    match indices {
        Some(idx) => {
            for tri in idx.chunks_exact(3) {
                process_triangle(
                    tri[0] as usize,
                    tri[1] as usize,
                    tri[2] as usize,
                    &mut tan1,
                    &mut tan2,
                );
            }
        }
        None => {
            for i in (0..vertex_count).step_by(3) {
                if i + 2 < vertex_count {
                    process_triangle(i, i + 1, i + 2, &mut tan1, &mut tan2);
                }
            }
        }
    }

    // Gram-Schmidt orthogonalize and compute handedness
    for i in 0..vertex_count {
        let n = Vec3::from(vertices[i].normal);
        let t = tan1[i];

        // Orthogonalize: t' = normalize(t - n * dot(n, t))
        let tangent = t - n * n.dot(t);
        if tangent.length_squared() > 1e-8 {
            let tangent = tangent.normalize();
            // Handedness: sign of dot(cross(n, t), tan2)
            let w = if n.cross(t).dot(tan2[i]) < 0.0 {
                -1.0
            } else {
                1.0
            };
            vertices[i].tangent = [tangent.x, tangent.y, tangent.z, w];
        } else {
            // Fallback tangent
            vertices[i].tangent = [1.0, 0.0, 0.0, 1.0];
        }
    }
}
