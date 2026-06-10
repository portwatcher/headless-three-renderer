use anyhow::{Context, Result, bail};
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};

use crate::types::SceneMesh;
use crate::util::{clamp01, color_to_f32, finite_f32, parse_color, parse_transform};

pub const MAX_CLIPPING_PLANES: usize = 8;

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
    pub vertices: Vec<Vertex>,
    pub indices: Option<Vec<u32>>,
    pub transform: Mat4,
    pub texture: Option<PreparedTexture>,
    pub normal_map: Option<PreparedTexture>,
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
    pub transmission: f32,
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
    pub premultiplied_alpha: bool,
    pub fog: bool,
    pub clipping_planes: [[f32; 4]; MAX_CLIPPING_PLANES],
    pub clipping_plane_count: u32,
    pub clipping_union_count: u32,
    pub is_transparent: bool,
    pub blending: BlendMode,
    pub custom_blend: Option<CustomBlendState>,
    pub depth_test: bool,
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
    pub shading_model: ShadingModel,
    pub topology: Topology,
    pub custom_fragment_shader: Option<String>,
    pub texture_transform: [f32; 6],
    pub texture_uses_uv2: bool,
    pub alpha_map_transform: [f32; 6],
    pub alpha_map_uses_uv2: bool,
    pub normal_map_transform: [f32; 6],
    pub normal_map_uses_uv2: bool,
    pub bump_map_transform: [f32; 6],
    pub bump_map_uses_uv2: bool,
    pub matcap_map_transform: [f32; 6],
    pub matcap_map_uses_uv2: bool,
    pub metallic_roughness_texture_transform: [f32; 6],
    pub metallic_roughness_texture_uses_uv2: bool,
    pub emissive_map_transform: [f32; 6],
    pub emissive_map_uses_uv2: bool,
    pub ao_map_transform: [f32; 6],
    pub light_map_transform: [f32; 6],
    pub specular_map_transform: [f32; 6],
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
    pub transmission_map_transform: [f32; 6],
    pub transmission_map_uses_uv2: bool,
    pub thickness_map_transform: [f32; 6],
    pub thickness_map_uses_uv2: bool,
    pub texture_is_srgb: bool,
    pub matcap_map_is_srgb: bool,
    pub emissive_map_is_srgb: bool,
    pub light_map_is_srgb: bool,
    pub cast_shadow: bool,
    pub receive_shadow: bool,
    pub group_order: f32,
    pub render_order: f32,
    pub sort_z: f32,
    pub sort_index: u32,
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
    pub physical_layers_sampler: TextureSamplerSettings,
    pub sheen_sampler: TextureSamplerSettings,
    pub specular_sampler: TextureSamplerSettings,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct TextureSamplerSettings {
    pub wrap_s: WrapMode,
    pub wrap_t: WrapMode,
    pub mag_filter: TextureFilter,
    pub min_filter: TextureFilter,
}

impl Default for TextureSamplerSettings {
    fn default() -> Self {
        Self {
            wrap_s: WrapMode::ClampToEdge,
            wrap_t: WrapMode::ClampToEdge,
            mag_filter: TextureFilter::Linear,
            min_filter: TextureFilter::Linear,
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

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
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

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
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

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BlendEquation {
    Add,
    Subtract,
    ReverseSubtract,
    Min,
    Max,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
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

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
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

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
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

pub struct PreparedTexture {
    pub rgba: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub wrap_s: WrapMode,
    pub wrap_t: WrapMode,
    pub mag_filter: TextureFilter,
    pub min_filter: TextureFilter,
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
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

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
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

    pub fn to_filter_mode(self) -> wgpu::FilterMode {
        match self {
            Self::Nearest => wgpu::FilterMode::Nearest,
            Self::Linear => wgpu::FilterMode::Linear,
        }
    }
}

pub fn prepare_meshes(scene: &crate::types::RenderScene) -> Result<Vec<PreparedMesh>> {
    if let Some(meshes) = scene.meshes.as_deref() {
        meshes.iter().enumerate().map(prepare_mesh).collect()
    } else {
        Ok(Vec::new())
    }
}

fn prepare_mesh((mesh_index, mesh): (usize, &SceneMesh)) -> Result<PreparedMesh> {
    let topology = Topology::from_str_opt(mesh.topology.as_deref());

    let min_positions = match topology {
        Topology::Triangles => 9, // at least 3 xyz
        Topology::Lines => 6,     // at least 2 xyz
        Topology::Points => 3,    // at least 1 xyz
    };
    if mesh.positions.len() < min_positions || mesh.positions.len() % 3 != 0 {
        bail!(
            "scene.meshes[{mesh_index}].positions must contain at least {} xyz vertices",
            min_positions / 3
        );
    }

    let vertex_count = mesh.positions.len() / 3;
    let material_color = parse_color(
        mesh.color.as_deref(),
        [0.82, 0.82, 0.82, 1.0],
        &format!("scene.meshes[{mesh_index}].color"),
    )?;

    let color_mode = ColorMode::new(
        mesh.colors.as_deref(),
        vertex_count,
        material_color,
        mesh_index,
    )?;

    let uvs = mesh.uvs.as_deref();
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

    let uvs2 = mesh.uvs2.as_deref();
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

    let normals = mesh.normals.as_deref();
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
                [n[base] as f32, n[base + 1] as f32, n[base + 2] as f32]
            } else {
                [0.0, 0.0, 0.0]
            },
            tangent: [0.0, 0.0, 0.0, 0.0],
            color: color_mode.color(vertex_index),
            uv: if has_uvs {
                let u = uvs.unwrap();
                [u[uv_base] as f32, u[uv_base + 1] as f32]
            } else {
                [0.0, 0.0]
            },
            uv2: if has_uvs2 {
                let u = uvs2.unwrap();
                [u[uv_base] as f32, u[uv_base + 1] as f32]
            } else if has_uvs {
                let u = uvs.unwrap();
                [u[uv_base] as f32, u[uv_base + 1] as f32]
            } else {
                [0.0, 0.0]
            },
        });
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
    } else if !has_normals && topology == Topology::Triangles {
        compute_flat_normals(&mut vertices, indices.as_deref());
    }

    let texture = match &mesh.texture {
        Some(tex_data) if !tex_data.is_empty() => {
            let mut tex = decode_texture(
                tex_data,
                mesh.texture_width,
                mesh.texture_height,
                mesh_index,
            )?;
            tex.wrap_s = WrapMode::from_str_opt(mesh.texture_wrap_s.as_deref());
            tex.wrap_t = WrapMode::from_str_opt(mesh.texture_wrap_t.as_deref());
            tex.mag_filter = TextureFilter::from_str_opt(mesh.texture_mag_filter.as_deref());
            tex.min_filter = TextureFilter::from_str_opt(mesh.texture_min_filter.as_deref());
            Some(tex)
        }
        _ => None,
    };
    let texture_transform = parse_texture_transform(mesh.texture_transform.as_deref(), mesh_index)?;
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
    let ao_map_transform = parse_texture_transform(mesh.ao_map_transform.as_deref(), mesh_index)?;
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
    let transmission_map_transform =
        parse_texture_transform(mesh.transmission_map_transform.as_deref(), mesh_index)?;
    let thickness_map_transform =
        parse_texture_transform(mesh.thickness_map_transform.as_deref(), mesh_index)?;
    let texture_is_srgb = matches!(mesh.texture_color_space.as_deref(), Some("srgb"));
    let matcap_map_is_srgb = matches!(mesh.matcap_map_color_space.as_deref(), Some("srgb"));
    let emissive_map_is_srgb = matches!(mesh.emissive_map_color_space.as_deref(), Some("srgb"));
    let light_map_is_srgb = matches!(mesh.light_map_color_space.as_deref(), Some("srgb"));

    let normal_map = match &mesh.normal_map {
        Some(tex_data) if !tex_data.is_empty() => {
            let mut tex = decode_texture(
                tex_data,
                mesh.normal_map_width,
                mesh.normal_map_height,
                mesh_index,
            )?;
            apply_texture_sampling(
                &mut tex,
                mesh.normal_map_wrap_s.as_deref(),
                mesh.normal_map_wrap_t.as_deref(),
                mesh.normal_map_mag_filter.as_deref(),
                mesh.normal_map_min_filter.as_deref(),
            );
            Some(tex)
        }
        _ => None,
    };

    let normal_scale = match mesh.normal_scale.as_deref() {
        Some(s) if s.len() == 2 => [s[0] as f32, s[1] as f32],
        _ => [1.0, 1.0],
    };
    let bump_map = match &mesh.bump_map {
        Some(tex_data) if !tex_data.is_empty() => {
            let mut tex = decode_texture(
                tex_data,
                mesh.bump_map_width,
                mesh.bump_map_height,
                mesh_index,
            )?;
            apply_texture_sampling(
                &mut tex,
                mesh.bump_map_wrap_s.as_deref(),
                mesh.bump_map_wrap_t.as_deref(),
                mesh.bump_map_mag_filter.as_deref(),
                mesh.bump_map_min_filter.as_deref(),
            );
            Some(tex)
        }
        _ => None,
    };
    let bump_scale = finite_f32(mesh.bump_scale.unwrap_or(1.0), "mesh bumpScale")?;

    let displacement_map = match &mesh.displacement_map {
        Some(tex_data) if !tex_data.is_empty() => Some(decode_texture(
            tex_data,
            mesh.displacement_map_width,
            mesh.displacement_map_height,
            mesh_index,
        )?),
        _ => None,
    };
    let displacement_scale = finite_f32(
        mesh.displacement_scale.unwrap_or(1.0),
        "mesh displacementScale",
    )?;
    let displacement_bias = finite_f32(
        mesh.displacement_bias.unwrap_or(0.0),
        "mesh displacementBias",
    )?;

    let gradient_map = match &mesh.gradient_map {
        Some(tex_data) if !tex_data.is_empty() => Some(decode_texture(
            tex_data,
            mesh.gradient_map_width,
            mesh.gradient_map_height,
            mesh_index,
        )?),
        _ => None,
    };
    let matcap_map = match &mesh.matcap_map {
        Some(tex_data) if !tex_data.is_empty() => Some(decode_texture(
            tex_data,
            mesh.matcap_map_width,
            mesh.matcap_map_height,
            mesh_index,
        )?),
        _ => None,
    };

    let metallic_roughness_texture = match &mesh.metallic_roughness_texture {
        Some(tex_data) if !tex_data.is_empty() => {
            let mut tex = decode_texture(
                tex_data,
                mesh.metallic_roughness_texture_width,
                mesh.metallic_roughness_texture_height,
                mesh_index,
            )?;
            apply_texture_sampling(
                &mut tex,
                mesh.metallic_roughness_texture_wrap_s.as_deref(),
                mesh.metallic_roughness_texture_wrap_t.as_deref(),
                mesh.metallic_roughness_texture_mag_filter.as_deref(),
                mesh.metallic_roughness_texture_min_filter.as_deref(),
            );
            Some(tex)
        }
        _ => None,
    };

    let specular_map = match &mesh.specular_map {
        Some(tex_data) if !tex_data.is_empty() => {
            let mut tex = decode_texture(
                tex_data,
                mesh.specular_map_width,
                mesh.specular_map_height,
                mesh_index,
            )?;
            apply_texture_sampling(
                &mut tex,
                mesh.specular_map_wrap_s.as_deref(),
                mesh.specular_map_wrap_t.as_deref(),
                mesh.specular_map_mag_filter.as_deref(),
                mesh.specular_map_min_filter.as_deref(),
            );
            Some(tex)
        }
        _ => None,
    };

    let emissive_map = match &mesh.emissive_map {
        Some(tex_data) if !tex_data.is_empty() => {
            let mut tex = decode_texture(
                tex_data,
                mesh.emissive_map_width,
                mesh.emissive_map_height,
                mesh_index,
            )?;
            apply_texture_sampling(
                &mut tex,
                mesh.emissive_map_wrap_s.as_deref(),
                mesh.emissive_map_wrap_t.as_deref(),
                mesh.emissive_map_mag_filter.as_deref(),
                mesh.emissive_map_min_filter.as_deref(),
            );
            Some(tex)
        }
        _ => None,
    };

    let ao_map = match &mesh.ao_map {
        Some(tex_data) if !tex_data.is_empty() => {
            let mut tex =
                decode_texture(tex_data, mesh.ao_map_width, mesh.ao_map_height, mesh_index)?;
            apply_texture_sampling(
                &mut tex,
                mesh.ao_map_wrap_s.as_deref(),
                mesh.ao_map_wrap_t.as_deref(),
                mesh.ao_map_mag_filter.as_deref(),
                mesh.ao_map_min_filter.as_deref(),
            );
            Some(tex)
        }
        _ => None,
    };
    let ao_map_intensity = clamp01(mesh.ao_map_intensity.unwrap_or(1.0)) as f32;

    let light_map = match &mesh.light_map {
        Some(tex_data) if !tex_data.is_empty() => {
            let mut tex = decode_texture(
                tex_data,
                mesh.light_map_width,
                mesh.light_map_height,
                mesh_index,
            )?;
            apply_texture_sampling(
                &mut tex,
                mesh.light_map_wrap_s.as_deref(),
                mesh.light_map_wrap_t.as_deref(),
                mesh.light_map_mag_filter.as_deref(),
                mesh.light_map_min_filter.as_deref(),
            );
            Some(tex)
        }
        _ => None,
    };
    let light_map_intensity = finite_f32(
        mesh.light_map_intensity.unwrap_or(1.0),
        "mesh lightMapIntensity",
    )?
    .max(0.0);

    let alpha_map = match &mesh.alpha_map {
        Some(tex_data) if !tex_data.is_empty() => {
            let mut tex = decode_texture(
                tex_data,
                mesh.alpha_map_width,
                mesh.alpha_map_height,
                mesh_index,
            )?;
            apply_texture_sampling(
                &mut tex,
                mesh.alpha_map_wrap_s.as_deref(),
                mesh.alpha_map_wrap_t.as_deref(),
                mesh.alpha_map_mag_filter.as_deref(),
                mesh.alpha_map_min_filter.as_deref(),
            );
            Some(tex)
        }
        _ => None,
    };

    let clearcoat_map = decode_optional_texture_with_sampling(
        mesh.clearcoat_map.as_deref(),
        mesh.clearcoat_map_width,
        mesh.clearcoat_map_height,
        mesh_index,
        mesh.clearcoat_map_wrap_s.as_deref(),
        mesh.clearcoat_map_wrap_t.as_deref(),
        mesh.clearcoat_map_mag_filter.as_deref(),
        mesh.clearcoat_map_min_filter.as_deref(),
    )?;
    let clearcoat_roughness_map = decode_optional_texture_with_sampling(
        mesh.clearcoat_roughness_map.as_deref(),
        mesh.clearcoat_roughness_map_width,
        mesh.clearcoat_roughness_map_height,
        mesh_index,
        mesh.clearcoat_roughness_map_wrap_s.as_deref(),
        mesh.clearcoat_roughness_map_wrap_t.as_deref(),
        mesh.clearcoat_roughness_map_mag_filter.as_deref(),
        mesh.clearcoat_roughness_map_min_filter.as_deref(),
    )?;
    let clearcoat_normal_map = decode_optional_texture_with_sampling(
        mesh.clearcoat_normal_map.as_deref(),
        mesh.clearcoat_normal_map_width,
        mesh.clearcoat_normal_map_height,
        mesh_index,
        mesh.clearcoat_normal_map_wrap_s.as_deref(),
        mesh.clearcoat_normal_map_wrap_t.as_deref(),
        mesh.clearcoat_normal_map_mag_filter.as_deref(),
        mesh.clearcoat_normal_map_min_filter.as_deref(),
    )?;
    let sheen_color_map = decode_optional_texture_with_sampling(
        mesh.sheen_color_map.as_deref(),
        mesh.sheen_color_map_width,
        mesh.sheen_color_map_height,
        mesh_index,
        mesh.sheen_color_map_wrap_s.as_deref(),
        mesh.sheen_color_map_wrap_t.as_deref(),
        mesh.sheen_color_map_mag_filter.as_deref(),
        mesh.sheen_color_map_min_filter.as_deref(),
    )?;
    let sheen_roughness_map = decode_optional_texture_with_sampling(
        mesh.sheen_roughness_map.as_deref(),
        mesh.sheen_roughness_map_width,
        mesh.sheen_roughness_map_height,
        mesh_index,
        mesh.sheen_roughness_map_wrap_s.as_deref(),
        mesh.sheen_roughness_map_wrap_t.as_deref(),
        mesh.sheen_roughness_map_mag_filter.as_deref(),
        mesh.sheen_roughness_map_min_filter.as_deref(),
    )?;
    let anisotropy_map = decode_optional_texture_with_sampling(
        mesh.anisotropy_map.as_deref(),
        mesh.anisotropy_map_width,
        mesh.anisotropy_map_height,
        mesh_index,
        mesh.anisotropy_map_wrap_s.as_deref(),
        mesh.anisotropy_map_wrap_t.as_deref(),
        mesh.anisotropy_map_mag_filter.as_deref(),
        mesh.anisotropy_map_min_filter.as_deref(),
    )?;
    let transmission_map = decode_optional_texture_with_sampling(
        mesh.transmission_map.as_deref(),
        mesh.transmission_map_width,
        mesh.transmission_map_height,
        mesh_index,
        mesh.transmission_map_wrap_s.as_deref(),
        mesh.transmission_map_wrap_t.as_deref(),
        mesh.transmission_map_mag_filter.as_deref(),
        mesh.transmission_map_min_filter.as_deref(),
    )?;
    let thickness_map = decode_optional_texture_with_sampling(
        mesh.thickness_map.as_deref(),
        mesh.thickness_map_width,
        mesh.thickness_map_height,
        mesh_index,
        mesh.thickness_map_wrap_s.as_deref(),
        mesh.thickness_map_wrap_t.as_deref(),
        mesh.thickness_map_mag_filter.as_deref(),
        mesh.thickness_map_min_filter.as_deref(),
    )?;
    let specular_color_map = decode_optional_texture_with_sampling(
        mesh.specular_color_map.as_deref(),
        mesh.specular_color_map_width,
        mesh.specular_color_map_height,
        mesh_index,
        mesh.specular_color_map_wrap_s.as_deref(),
        mesh.specular_color_map_wrap_t.as_deref(),
        mesh.specular_color_map_mag_filter.as_deref(),
        mesh.specular_color_map_min_filter.as_deref(),
    )?;
    let specular_intensity_map = decode_optional_texture_with_sampling(
        mesh.specular_intensity_map.as_deref(),
        mesh.specular_intensity_map_width,
        mesh.specular_intensity_map_height,
        mesh_index,
        mesh.specular_intensity_map_wrap_s.as_deref(),
        mesh.specular_intensity_map_wrap_t.as_deref(),
        mesh.specular_intensity_map_mag_filter.as_deref(),
        mesh.specular_intensity_map_min_filter.as_deref(),
    )?;
    let physical_maps = pack_physical_maps(PhysicalMapInputs {
        clearcoat: clearcoat_map.as_ref(),
        clearcoat_roughness: clearcoat_roughness_map.as_ref(),
        sheen_color: sheen_color_map.as_ref(),
        sheen_roughness: sheen_roughness_map.as_ref(),
        anisotropy: anisotropy_map.as_ref(),
        transmission: transmission_map.as_ref(),
        thickness: thickness_map.as_ref(),
        specular_color: specular_color_map.as_ref(),
        specular_intensity: specular_intensity_map.as_ref(),
        sheen_color_is_srgb: matches!(mesh.sheen_color_map_color_space.as_deref(), Some("srgb")),
        specular_color_is_srgb: matches!(
            mesh.specular_color_map_color_space.as_deref(),
            Some("srgb")
        ),
    });

    let metallic = clamp01(mesh.metallic.unwrap_or(0.0)) as f32;
    let roughness = clamp01(mesh.roughness.unwrap_or(1.0)) as f32;
    let clearcoat = clamp01(mesh.clearcoat.unwrap_or(0.0)) as f32;
    let clearcoat_roughness = clamp01(mesh.clearcoat_roughness.unwrap_or(0.0)) as f32;
    let clearcoat_normal_scale = match mesh.clearcoat_normal_scale.as_deref() {
        Some(s) if s.len() == 2 => [s[0] as f32, s[1] as f32],
        _ => [1.0, 1.0],
    };
    let sheen_color = match mesh.sheen_color.as_deref() {
        Some(s) if s.len() == 3 => [
            clamp01(s[0]) as f32,
            clamp01(s[1]) as f32,
            clamp01(s[2]) as f32,
        ],
        _ => [0.0, 0.0, 0.0],
    };
    let sheen_roughness = clamp01(mesh.sheen_roughness.unwrap_or(1.0)).max(0.0001) as f32;
    let anisotropy = clamp01(mesh.anisotropy.unwrap_or(0.0)) as f32;
    let anisotropy_rotation = finite_f32(
        mesh.anisotropy_rotation.unwrap_or(0.0),
        "mesh anisotropyRotation",
    )?;
    let transmission = clamp01(mesh.transmission.unwrap_or(0.0)) as f32;
    let ior = mesh.ior.unwrap_or(1.5).clamp(1.0, 2.333) as f32;
    let thickness = mesh.thickness.unwrap_or(0.0).max(0.0) as f32;
    let attenuation_distance = mesh
        .attenuation_distance
        .unwrap_or(1.0e20)
        .max(0.0)
        .min(1.0e20) as f32;
    let attenuation_color = match mesh.attenuation_color.as_deref() {
        Some(c) if c.len() == 3 => [
            clamp01(c[0]) as f32,
            clamp01(c[1]) as f32,
            clamp01(c[2]) as f32,
        ],
        _ => [1.0, 1.0, 1.0],
    };
    let physical_specular_color = match mesh.physical_specular_color.as_deref() {
        Some(c) if c.len() == 3 => [
            clamp01(c[0]) as f32,
            clamp01(c[1]) as f32,
            clamp01(c[2]) as f32,
        ],
        _ => [1.0, 1.0, 1.0],
    };
    let physical_specular_intensity =
        clamp01(mesh.physical_specular_intensity.unwrap_or(1.0)) as f32;
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
    let specular_color = match mesh.specular_color.as_deref() {
        Some(c) if c.len() == 3 => [
            clamp01(c[0]) as f32,
            clamp01(c[1]) as f32,
            clamp01(c[2]) as f32,
        ],
        _ => [17.0 / 255.0, 17.0 / 255.0, 17.0 / 255.0],
    };
    let shininess = finite_f32(mesh.shininess.unwrap_or(30.0), "mesh shininess")?.max(0.0001);

    if let Some(displacement_map) = displacement_map.as_ref() {
        if has_uvs && topology == Topology::Triangles {
            apply_displacement_map(
                &mut vertices,
                displacement_map,
                displacement_map_transform,
                mesh.displacement_map_uses_uv2.unwrap_or(false),
                displacement_scale,
                displacement_bias,
            );
        }
    }

    // Compute tangents when normal/bump mapping or anisotropic shading needs a frame.
    if (normal_map.is_some()
        || bump_map.is_some()
        || clearcoat_normal_map.is_some()
        || anisotropy > 0.0)
        && has_uvs
        && topology == Topology::Triangles
    {
        compute_tangents(&mut vertices, mesh.indices.as_deref());
    }

    let emissive_intensity = mesh.emissive_intensity.unwrap_or(1.0) as f32;
    let emissive = match mesh.emissive.as_deref() {
        Some(e) if e.len() == 3 => [
            clamp01(e[0]) as f32 * emissive_intensity,
            clamp01(e[1]) as f32 * emissive_intensity,
            clamp01(e[2]) as f32 * emissive_intensity,
        ],
        _ => [0.0, 0.0, 0.0],
    };

    let alpha_test = clamp01(mesh.alpha_test.unwrap_or(0.0)) as f32;
    let alpha_hash = mesh.alpha_hash.unwrap_or(false);
    let premultiplied_alpha = mesh.premultiplied_alpha.unwrap_or(false);
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
    let default_depth_write = !is_transparent;
    let depth_write = depth_test && mesh.depth_write.unwrap_or(default_depth_write);
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
    let side = MeshSide::from_str_opt(mesh.side.as_deref());
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

    Ok(PreparedMesh {
        vertices,
        indices,
        transform: parse_transform(mesh.transform.as_deref(), mesh_index)?,
        texture,
        normal_map,
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
        transmission,
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
        premultiplied_alpha,
        fog,
        clipping_planes,
        clipping_plane_count,
        clipping_union_count,
        is_transparent,
        blending,
        custom_blend,
        depth_test,
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
        shading_model,
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
        alpha_map_uses_uv2: mesh.alpha_map_uses_uv2.unwrap_or(false),
        normal_map_transform,
        normal_map_uses_uv2: mesh.normal_map_uses_uv2.unwrap_or(false),
        bump_map_transform,
        bump_map_uses_uv2: mesh.bump_map_uses_uv2.unwrap_or(false),
        matcap_map_transform,
        matcap_map_uses_uv2: mesh.matcap_map_uses_uv2.unwrap_or(false),
        metallic_roughness_texture_transform,
        metallic_roughness_texture_uses_uv2: mesh
            .metallic_roughness_texture_uses_uv2
            .unwrap_or(false),
        emissive_map_transform,
        emissive_map_uses_uv2: mesh.emissive_map_uses_uv2.unwrap_or(false),
        ao_map_transform,
        light_map_transform,
        specular_map_transform,
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
        transmission_map_transform,
        transmission_map_uses_uv2: mesh.transmission_map_uses_uv2.unwrap_or(false),
        thickness_map_transform,
        thickness_map_uses_uv2: mesh.thickness_map_uses_uv2.unwrap_or(false),
        texture_is_srgb,
        matcap_map_is_srgb,
        emissive_map_is_srgb,
        light_map_is_srgb,
        cast_shadow: mesh.cast_shadow.unwrap_or(false),
        receive_shadow: mesh.receive_shadow.unwrap_or(false),
        group_order: finite_f32(mesh.group_order.unwrap_or(0.0), "mesh groupOrder")?,
        render_order: finite_f32(mesh.render_order.unwrap_or(0.0), "mesh renderOrder")?,
        sort_z: finite_f32(mesh.sort_z.unwrap_or(0.0), "mesh sortZ")?,
        sort_index: mesh.sort_index.unwrap_or(mesh_index as u32),
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
    let blend_alpha = clamp01(mesh.blend_alpha.unwrap_or(0.0)) as f32;

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

    if w > 0 && h > 0 && data.len() == (w as usize) * (h as usize) * 4 {
        return Ok(PreparedTexture {
            rgba: data.to_vec(),
            width: w,
            height: h,
            wrap_s: WrapMode::ClampToEdge,
            wrap_t: WrapMode::ClampToEdge,
            mag_filter: TextureFilter::Linear,
            min_filter: TextureFilter::Linear,
        });
    }

    let img = image::load_from_memory(data)
        .with_context(|| format!("{label}: failed to decode image ({} bytes)", data.len()))?;
    let rgba = img.to_rgba8();
    Ok(PreparedTexture {
        width: rgba.width(),
        height: rgba.height(),
        rgba: rgba.into_raw(),
        wrap_s: WrapMode::ClampToEdge,
        wrap_t: WrapMode::ClampToEdge,
        mag_filter: TextureFilter::Linear,
        min_filter: TextureFilter::Linear,
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
) -> Result<Option<PreparedTexture>> {
    let mut texture = decode_optional_texture(data, width_hint, height_hint, mesh_index)?;
    if let Some(tex) = texture.as_mut() {
        apply_texture_sampling(tex, wrap_s, wrap_t, mag_filter, min_filter);
    }
    Ok(texture)
}

fn apply_texture_sampling(
    texture: &mut PreparedTexture,
    wrap_s: Option<&str>,
    wrap_t: Option<&str>,
    mag_filter: Option<&str>,
    min_filter: Option<&str>,
) {
    texture.wrap_s = WrapMode::from_str_opt(wrap_s);
    texture.wrap_t = WrapMode::from_str_opt(wrap_t);
    texture.mag_filter = TextureFilter::from_str_opt(mag_filter);
    texture.min_filter = TextureFilter::from_str_opt(min_filter);
}

struct PhysicalMapInputs<'a> {
    clearcoat: Option<&'a PreparedTexture>,
    clearcoat_roughness: Option<&'a PreparedTexture>,
    sheen_color: Option<&'a PreparedTexture>,
    sheen_roughness: Option<&'a PreparedTexture>,
    anisotropy: Option<&'a PreparedTexture>,
    transmission: Option<&'a PreparedTexture>,
    thickness: Option<&'a PreparedTexture>,
    specular_color: Option<&'a PreparedTexture>,
    specular_intensity: Option<&'a PreparedTexture>,
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
                scalar[out] = sample_texture_channel(map, x, y, width, height, 0);
            }
            if let Some(map) = inputs.clearcoat_roughness {
                scalar[out + 1] = sample_texture_channel(map, x, y, width, height, 1);
            }
            if let Some(map) = inputs.transmission {
                scalar[out + 2] = sample_texture_channel(map, x, y, width, height, 0);
            }
            if let Some(map) = inputs.thickness {
                scalar[out + 3] = sample_texture_channel(map, x, y, width, height, 1);
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
                anisotropy[out] = sample_texture_channel(map, x, y, width, height, 0);
                anisotropy[out + 1] = sample_texture_channel(map, x, y, width, height, 1);
                anisotropy[out + 2] = sample_texture_channel(map, x, y, width, height, 2);
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

    Some(PreparedPhysicalMaps {
        scalar_map: packed_texture(scalar, width, height),
        sheen_map: packed_texture(sheen, width, height),
        anisotropy_map: packed_texture(anisotropy, width, height),
        specular_map: packed_texture(specular, width, height),
        physical_layers_sampler: TextureSamplerSettings::first_from_textures(&[
            inputs.clearcoat,
            inputs.clearcoat_roughness,
            inputs.transmission,
            inputs.thickness,
            inputs.anisotropy,
        ]),
        sheen_sampler: TextureSamplerSettings::first_from_textures(&[
            inputs.sheen_color,
            inputs.sheen_roughness,
        ]),
        specular_sampler: TextureSamplerSettings::first_from_textures(&[
            inputs.specular_color,
            inputs.specular_intensity,
        ]),
    })
}

fn packed_texture(rgba: Vec<u8>, width: u32, height: u32) -> PreparedTexture {
    PreparedTexture {
        rgba,
        width,
        height,
        wrap_s: WrapMode::ClampToEdge,
        wrap_t: WrapMode::ClampToEdge,
        mag_filter: TextureFilter::Linear,
        min_filter: TextureFilter::Linear,
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
    if is_srgb {
        srgb_u8_to_linear_u8(value)
    } else {
        value
    }
}

fn srgb_u8_to_linear_u8(value: u8) -> u8 {
    let channel = value as f32 / 255.0;
    let linear = if channel <= 0.04045 {
        channel / 12.92
    } else {
        ((channel + 0.055) / 1.055).powf(2.4)
    };
    (linear.clamp(0.0, 1.0) * 255.0).round() as u8
}

fn apply_displacement_map(
    vertices: &mut [Vertex],
    texture: &PreparedTexture,
    transform: [f32; 6],
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
        let sample = sample_texture_channel_uv(texture, uv[0], uv[1], 0);
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
    let x = (u.clamp(0.0, 1.0) * texture.width as f32)
        .floor()
        .clamp(0.0, (texture.width - 1) as f32) as u32;
    let y = (v.clamp(0.0, 1.0) * texture.height as f32)
        .floor()
        .clamp(0.0, (texture.height - 1) as f32) as u32;
    texture.rgba[((y * texture.width + x) * 4) as usize + channel] as f32 / 255.0
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
        material_color: [f64; 4],
        mesh_index: usize,
    ) -> Result<Self> {
        let Some(colors) = colors else {
            return Ok(Self::Uniform(color_to_f32(material_color)));
        };

        match colors.len() {
            3 | 4 => Ok(Self::Uniform(color_to_f32(parse_color(
                Some(colors),
                material_color,
                &format!("scene.meshes[{mesh_index}].colors"),
            )?))),
            len if len == vertex_count * 3 => Ok(Self::RgbPerVertex(colors)),
            len if len == vertex_count * 4 => Ok(Self::RgbaPerVertex(colors)),
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
