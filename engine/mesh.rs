use anyhow::{Context, Result, bail};
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};

use crate::types::SceneMesh;
use crate::util::{clamp01, color_to_f32, finite_f32, parse_color, parse_transform};

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub tangent: [f32; 4],
    pub color: [f32; 4],
    pub uv: [f32; 2],
}

impl Vertex {
    const ATTRIBUTES: [wgpu::VertexAttribute; 5] = wgpu::vertex_attr_array![
        0 => Float32x3,
        1 => Float32x3,
        2 => Float32x4,
        3 => Float32x4,
        4 => Float32x2,
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
    pub metallic_roughness_texture: Option<PreparedTexture>,
    pub emissive_map: Option<PreparedTexture>,
    pub ao_map: Option<PreparedTexture>,
    pub ao_map_intensity: f32,
    pub metallic: f32,
    pub roughness: f32,
    pub emissive: [f32; 3],
    pub base_color: [f32; 4],
    pub alpha_test: f32,
    pub is_transparent: bool,
    pub side: MeshSide,
    pub shading_model: ShadingModel,
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum ShadingModel {
    #[default]
    Standard,
    Basic,
    Lambert,
}

impl ShadingModel {
    pub fn from_str_opt(value: Option<&str>) -> Self {
        match value {
            Some("basic") => Self::Basic,
            Some("lambert") => Self::Lambert,
            _ => Self::Standard,
        }
    }

    pub fn as_u32(self) -> u32 {
        match self {
            Self::Standard => 0,
            Self::Basic => 1,
            Self::Lambert => 2,
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

pub struct PreparedTexture {
    pub rgba: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub wrap_s: WrapMode,
    pub wrap_t: WrapMode,
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

pub fn prepare_meshes(scene: &crate::types::RenderScene) -> Result<Vec<PreparedMesh>> {
    if let Some(meshes) = scene.meshes.as_deref() {
        meshes.iter().enumerate().map(prepare_mesh).collect()
    } else {
        Ok(Vec::new())
    }
}

fn prepare_mesh((mesh_index, mesh): (usize, &SceneMesh)) -> Result<PreparedMesh> {
    if mesh.positions.len() < 9 || mesh.positions.len() % 3 != 0 {
        bail!("scene.meshes[{mesh_index}].positions must contain at least 3 xyz vertices");
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
        });
    }

    // If no normals were provided, compute flat normals from triangle faces
    if !has_normals {
        compute_flat_normals(&mut vertices, mesh.indices.as_deref());
    }

    let indices = match &mesh.indices {
        Some(indices) => {
            if indices.len() < 3 || indices.len() % 3 != 0 {
                bail!("scene.meshes[{mesh_index}].indices must contain triangle-list indices");
            }
            for &index in indices {
                if index as usize >= vertex_count {
                    bail!("scene.meshes[{mesh_index}].indices contains out-of-range index {index}");
                }
            }
            Some(indices.clone())
        }
        None => {
            if vertex_count % 3 != 0 {
                bail!(
                    "scene.meshes[{mesh_index}] has no indices, so positions must define complete triangles"
                );
            }
            None
        }
    };

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
            Some(tex)
        }
        _ => None,
    };

    let normal_map = match &mesh.normal_map {
        Some(tex_data) if !tex_data.is_empty() => Some(decode_texture(
            tex_data,
            mesh.normal_map_width,
            mesh.normal_map_height,
            mesh_index,
        )?),
        _ => None,
    };

    let normal_scale = match mesh.normal_scale.as_deref() {
        Some(s) if s.len() == 2 => [s[0] as f32, s[1] as f32],
        _ => [1.0, 1.0],
    };

    let metallic_roughness_texture = match &mesh.metallic_roughness_texture {
        Some(tex_data) if !tex_data.is_empty() => Some(decode_texture(
            tex_data,
            mesh.metallic_roughness_texture_width,
            mesh.metallic_roughness_texture_height,
            mesh_index,
        )?),
        _ => None,
    };

    let emissive_map = match &mesh.emissive_map {
        Some(tex_data) if !tex_data.is_empty() => Some(decode_texture(
            tex_data,
            mesh.emissive_map_width,
            mesh.emissive_map_height,
            mesh_index,
        )?),
        _ => None,
    };

    let ao_map = match &mesh.ao_map {
        Some(tex_data) if !tex_data.is_empty() => Some(decode_texture(
            tex_data,
            mesh.ao_map_width,
            mesh.ao_map_height,
            mesh_index,
        )?),
        _ => None,
    };
    let ao_map_intensity = clamp01(mesh.ao_map_intensity.unwrap_or(1.0)) as f32;

    // Compute tangents when we have a normal map and UVs
    if normal_map.is_some() && has_uvs {
        compute_tangents(&mut vertices, mesh.indices.as_deref());
    }

    let metallic = clamp01(mesh.metallic.unwrap_or(0.0)) as f32;
    let roughness = clamp01(mesh.roughness.unwrap_or(1.0)) as f32;
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
    let is_transparent = mesh.transparent.unwrap_or(material_color[3] < 0.999);
    let side = MeshSide::from_str_opt(mesh.side.as_deref());
    let shading_model = ShadingModel::from_str_opt(mesh.shading_model.as_deref());

    Ok(PreparedMesh {
        vertices,
        indices,
        transform: parse_transform(mesh.transform.as_deref(), mesh_index)?,
        texture,
        normal_map,
        normal_scale,
        metallic_roughness_texture,
        emissive_map,
        ao_map,
        ao_map_intensity,
        metallic,
        roughness,
        emissive,
        base_color: color_to_f32(material_color),
        alpha_test,
        is_transparent,
        side,
        shading_model,
    })
}

pub fn decode_texture(
    data: &[u8],
    width_hint: Option<u32>,
    height_hint: Option<u32>,
    mesh_index: usize,
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
        });
    }

    let img = image::load_from_memory(data).with_context(|| {
        format!("scene.meshes[{mesh_index}].texture: failed to decode image ({} bytes)", data.len())
    })?;
    let rgba = img.to_rgba8();
    Ok(PreparedTexture {
        width: rgba.width(),
        height: rgba.height(),
        rgba: rgba.into_raw(),
        wrap_s: WrapMode::ClampToEdge,
        wrap_t: WrapMode::ClampToEdge,
    })
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

    let process_triangle = |i0: usize, i1: usize, i2: usize, tan1: &mut [Vec3], tan2: &mut [Vec3]| {
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
                process_triangle(tri[0] as usize, tri[1] as usize, tri[2] as usize, &mut tan1, &mut tan2);
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
            let w = if n.cross(t).dot(tan2[i]) < 0.0 { -1.0 } else { 1.0 };
            vertices[i].tangent = [tangent.x, tangent.y, tangent.z, w];
        } else {
            // Fallback tangent
            vertices[i].tangent = [1.0, 0.0, 0.0, 1.0];
        }
    }
}
