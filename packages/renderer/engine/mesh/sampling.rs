use super::*;

pub(super) fn apply_displacement_map(
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

pub(super) fn transform_uv(uv: [f32; 2], transform: [f32; 6]) -> [f32; 2] {
    [
        transform[0] * uv[0] + transform[1] * uv[1] + transform[2],
        transform[3] * uv[0] + transform[4] * uv[1] + transform[5],
    ]
}

pub(super) fn sample_texture_channel_uv(
    texture: &PreparedTexture,
    u: f32,
    v: f32,
    channel: usize,
) -> f32 {
    match texture.mag_filter {
        TextureFilter::Nearest => sample_texture_channel_nearest(texture, u, v, channel),
        TextureFilter::Linear => sample_texture_channel_linear(texture, u, v, channel),
    }
}

pub(super) fn sample_texture_color_channel_uv(
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

pub(super) fn sample_texture_channel_nearest(
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

pub(super) fn sample_texture_color_channel_nearest(
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

pub(super) fn sample_texture_channel_linear(
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

    let s00 = texel_channel(texture, x0, y0, channel);
    let s10 = texel_channel(texture, x0 + 1, y0, channel);
    let s01 = texel_channel(texture, x0, y0 + 1, channel);
    let s11 = texel_channel(texture, x0 + 1, y0 + 1, channel);
    let sx0 = s00 * (1.0 - tx) + s10 * tx;
    let sx1 = s01 * (1.0 - tx) + s11 * tx;
    sx0 * (1.0 - ty) + sx1 * ty
}

pub(super) fn sample_texture_color_channel_linear(
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

pub(super) fn texel_channel(texture: &PreparedTexture, x: i32, y: i32, channel: usize) -> f32 {
    let tx = wrapped_texel_index(x, texture.width, texture.wrap_s);
    let ty = wrapped_texel_index(y, texture.height, texture.wrap_t);
    texture.rgba[((ty * texture.width + tx) * 4) as usize + channel] as f32 / 255.0
}

pub(super) fn texel_color_channel(
    texture: &PreparedTexture,
    x: i32,
    y: i32,
    channel: usize,
) -> f32 {
    let tx = wrapped_texel_index(x, texture.width, texture.wrap_s);
    let ty = wrapped_texel_index(y, texture.height, texture.wrap_t);
    let value = texture.rgba[((ty * texture.width + tx) * 4) as usize + channel];
    srgb_u8_to_linear_f32(value)
}

pub(super) fn wrapped_texel_index(index: i32, size: u32, wrap: WrapMode) -> u32 {
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

pub(super) enum ColorMode<'a> {
    Uniform([f32; 4]),
    RgbPerVertex(&'a [f64]),
    RgbaPerVertex(&'a [f64]),
}

impl<'a> ColorMode<'a> {
    pub(super) fn new(
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

    pub(super) fn color(&self, vertex_index: usize) -> [f32; 4] {
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

pub(super) fn validate_color_values(values: &[f64], field: &str) -> Result<()> {
    for value in values {
        finite_color(*value, field)?;
    }
    Ok(())
}

pub(super) fn compute_flat_normals(vertices: &mut [Vertex], indices: Option<&[u32]>) {
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
pub(super) fn compute_tangents(vertices: &mut [Vertex], indices: Option<&[u32]>) {
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
