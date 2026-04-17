use anyhow::{Result, bail};
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};

use crate::types::SceneMesh;
use crate::util::{clamp01, color_to_f32, finite_f32, parse_color, parse_transform};

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 4],
    pub uv: [f32; 2],
}

impl Vertex {
    const ATTRIBUTES: [wgpu::VertexAttribute; 4] = wgpu::vertex_attr_array![
        0 => Float32x3,
        1 => Float32x3,
        2 => Float32x4,
        3 => Float32x2,
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
    pub metallic: f32,
    pub roughness: f32,
    pub emissive: [f32; 3],
    pub base_color: [f32; 4],
}

pub struct PreparedTexture {
    pub rgba: Vec<u8>,
    pub width: u32,
    pub height: u32,
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
        Some(tex_data) if !tex_data.is_empty() => Some(decode_texture(
            tex_data,
            mesh.texture_width,
            mesh.texture_height,
            mesh_index,
        )?),
        _ => None,
    };

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

    Ok(PreparedMesh {
        vertices,
        indices,
        transform: parse_transform(mesh.transform.as_deref(), mesh_index)?,
        texture,
        metallic,
        roughness,
        emissive,
        base_color: color_to_f32(material_color),
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

use anyhow::Context;
