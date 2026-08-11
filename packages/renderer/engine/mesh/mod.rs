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

mod decode;
mod physical;
mod prepare;
mod sampling;
mod texture_inputs;
mod validation;

use decode::*;
#[allow(unused_imports)]
pub use decode::{decode_texture, decode_texture_with_label};
use physical::*;
pub use prepare::{prepare_meshes, texture_anisotropy};
use sampling::*;
use texture_inputs::*;
use validation::*;
