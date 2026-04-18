use anyhow::{Context, Result};

/// IBL data precomputed on the CPU from an equirectangular HDR/LDR environment map.
///
/// We produce:
/// 1. A diffuse irradiance cubemap (low-res, one color per face direction)
/// 2. A prefiltered specular cubemap with roughness mip levels
/// 3. A BRDF integration LUT (2D texture, NdotV vs roughness)
///
/// The split-sum approximation is:
///   L_specular ≈ prefilteredColor(R, roughness) * (F0 * brdf.x + brdf.y)
///   L_diffuse  ≈ irradiance(N) * albedo

const BRDF_LUT_SIZE: u32 = 128;
const IRRADIANCE_SIZE: u32 = 32;
const PREFILTER_BASE_SIZE: u32 = 128;
const PREFILTER_MIP_LEVELS: u32 = 5;

pub struct IblMaps {
    /// Diffuse irradiance cubemap: 6 faces, IRRADIANCE_SIZE x IRRADIANCE_SIZE, RGBA32F stored as RGBA8.
    pub irradiance_faces: Vec<Vec<u8>>,
    pub irradiance_size: u32,
    /// Prefiltered specular cubemap: 6 faces × mip_levels, RGBA8.
    /// Indexed as [mip * 6 + face].
    pub prefilter_faces: Vec<Vec<u8>>,
    pub prefilter_base_size: u32,
    pub prefilter_mip_levels: u32,
    /// BRDF integration LUT: BRDF_LUT_SIZE × BRDF_LUT_SIZE, RG stored in RGBA8 (b=0, a=255).
    pub brdf_lut: Vec<u8>,
    pub brdf_lut_size: u32,
}

/// An HDR equirect environment map stored as linear f32 RGB pixels.
pub struct EnvMap {
    pub pixels: Vec<[f32; 3]>,
    pub width: u32,
    pub height: u32,
}

impl EnvMap {
    /// Decode from raw image bytes (PNG, JPEG, WebP, or HDR Radiance).
    /// Also accepts raw RGBA8 bytes if width/height hints are given.
    pub fn from_bytes(data: &[u8], width_hint: Option<u32>, height_hint: Option<u32>) -> Result<Self> {
        let w = width_hint.unwrap_or(0);
        let h = height_hint.unwrap_or(0);

        // Raw RGBA8 bytes?
        if w > 0 && h > 0 && data.len() == (w as usize) * (h as usize) * 4 {
            let mut pixels = Vec::with_capacity((w * h) as usize);
            for i in 0..(w * h) as usize {
                pixels.push([
                    srgb_to_linear(data[i * 4] as f32 / 255.0),
                    srgb_to_linear(data[i * 4 + 1] as f32 / 255.0),
                    srgb_to_linear(data[i * 4 + 2] as f32 / 255.0),
                ]);
            }
            return Ok(Self { pixels, width: w, height: h });
        }

        // Raw RGBA16F (half-float)? Three.js HalfFloatType = data is Float16 (2 bytes per component, 8 per pixel)
        if w > 0 && h > 0 && data.len() == (w as usize) * (h as usize) * 8 {
            let mut pixels = Vec::with_capacity((w * h) as usize);
            for i in 0..(w * h) as usize {
                let offset = i * 8;
                let r = half_to_f32(u16::from_le_bytes([data[offset], data[offset + 1]]));
                let g = half_to_f32(u16::from_le_bytes([data[offset + 2], data[offset + 3]]));
                let b = half_to_f32(u16::from_le_bytes([data[offset + 4], data[offset + 5]]));
                pixels.push([r, g, b]);
            }
            return Ok(Self { pixels, width: w, height: h });
        }

        // Raw RGBA32F?
        if w > 0 && h > 0 && data.len() == (w as usize) * (h as usize) * 16 {
            let mut pixels = Vec::with_capacity((w * h) as usize);
            for i in 0..(w * h) as usize {
                let offset = i * 16;
                let r = f32::from_le_bytes([data[offset], data[offset + 1], data[offset + 2], data[offset + 3]]);
                let g = f32::from_le_bytes([data[offset + 4], data[offset + 5], data[offset + 6], data[offset + 7]]);
                let b = f32::from_le_bytes([data[offset + 8], data[offset + 9], data[offset + 10], data[offset + 11]]);
                pixels.push([r, g, b]);
            }
            return Ok(Self { pixels, width: w, height: h });
        }

        // Try decoding as an image file
        let img = image::load_from_memory(data)
            .context("failed to decode environment map image")?;
        let rgba = img.to_rgba8();
        let w = rgba.width();
        let h = rgba.height();
        let raw = rgba.into_raw();
        let mut pixels = Vec::with_capacity((w * h) as usize);
        for i in 0..(w * h) as usize {
            pixels.push([
                srgb_to_linear(raw[i * 4] as f32 / 255.0),
                srgb_to_linear(raw[i * 4 + 1] as f32 / 255.0),
                srgb_to_linear(raw[i * 4 + 2] as f32 / 255.0),
            ]);
        }
        Ok(Self { pixels, width: w, height: h })
    }

    /// Sample the equirectangular map at a world-space direction.
    fn sample(&self, dir: [f32; 3]) -> [f32; 3] {
        let (dx, dy, dz) = (dir[0], dir[1], dir[2]);
        // Equirectangular: u = atan2(dx, dz) / (2π) + 0.5, v = asin(dy) / π + 0.5
        let u = dx.atan2(dz) * (0.5 / std::f32::consts::PI) + 0.5;
        let v = (-dy).asin() * (1.0 / std::f32::consts::PI) + 0.5;
        self.sample_uv(u, v)
    }

    fn sample_uv(&self, u: f32, v: f32) -> [f32; 3] {
        let u = u.fract();
        let u = if u < 0.0 { u + 1.0 } else { u };
        let v = v.clamp(0.0, 1.0);
        let x = (u * self.width as f32) as u32 % self.width;
        let y = (v * self.height as f32).min(self.height as f32 - 1.0) as u32;
        self.pixels[(y * self.width + x) as usize]
    }
}

pub fn compute_ibl(env_map: &EnvMap) -> IblMaps {
    let irradiance_faces = compute_irradiance(env_map);
    let prefilter_faces = compute_prefiltered_env(env_map);
    let brdf_lut = compute_brdf_lut();

    IblMaps {
        irradiance_faces,
        irradiance_size: IRRADIANCE_SIZE,
        prefilter_faces,
        prefilter_base_size: PREFILTER_BASE_SIZE,
        prefilter_mip_levels: PREFILTER_MIP_LEVELS,
        brdf_lut,
        brdf_lut_size: BRDF_LUT_SIZE,
    }
}

// ── Irradiance cubemap ──────────────────────────────────────────────

fn compute_irradiance(env_map: &EnvMap) -> Vec<Vec<u8>> {
    let size = IRRADIANCE_SIZE;
    let mut faces = Vec::with_capacity(6);
    for face in 0..6 {
        let mut rgba = vec![0u8; (size * size * 4) as usize];
        for y in 0..size {
            for x in 0..size {
                let dir = cube_dir(face, x, y, size);
                let color = convolve_diffuse(env_map, dir);
                let idx = ((y * size + x) * 4) as usize;
                rgba[idx] = linear_to_srgb8(color[0]);
                rgba[idx + 1] = linear_to_srgb8(color[1]);
                rgba[idx + 2] = linear_to_srgb8(color[2]);
                rgba[idx + 3] = 255;
            }
        }
        faces.push(rgba);
    }
    faces
}

fn convolve_diffuse(env_map: &EnvMap, normal: [f32; 3]) -> [f32; 3] {
    // Hemisphere convolution with cosine weighting.
    // Use a modest sample count for CPU perf.
    let n = normalize(normal);
    let (up, right) = make_tangent_frame(n);
    let sample_count = 128u32;
    let mut result = [0.0f32; 3];
    let mut total_weight = 0.0f32;

    for i in 0..sample_count {
        // Cosine-weighted hemisphere sampling using Hammersley sequence
        let xi = hammersley(i, sample_count);
        let (sin_theta, cos_theta) = cosine_sample_hemisphere(xi);
        let phi = 2.0 * std::f32::consts::PI * xi[0];
        let sample_dir = [
            right[0] * phi.cos() * sin_theta + up[0] * phi.sin() * sin_theta + n[0] * cos_theta,
            right[1] * phi.cos() * sin_theta + up[1] * phi.sin() * sin_theta + n[1] * cos_theta,
            right[2] * phi.cos() * sin_theta + up[2] * phi.sin() * sin_theta + n[2] * cos_theta,
        ];
        let sample_dir = normalize(sample_dir);
        let n_dot_l = dot(n, sample_dir).max(0.0);
        let color = env_map.sample(sample_dir);
        result[0] += color[0] * n_dot_l;
        result[1] += color[1] * n_dot_l;
        result[2] += color[2] * n_dot_l;
        total_weight += n_dot_l;
    }

    if total_weight > 0.0 {
        result[0] /= total_weight;
        result[1] /= total_weight;
        result[2] /= total_weight;
    }
    result
}

// ── Prefiltered specular cubemap ────────────────────────────────────

fn compute_prefiltered_env(env_map: &EnvMap) -> Vec<Vec<u8>> {
    let mut all_faces = Vec::with_capacity((PREFILTER_MIP_LEVELS * 6) as usize);
    for mip in 0..PREFILTER_MIP_LEVELS {
        let roughness = mip as f32 / (PREFILTER_MIP_LEVELS - 1).max(1) as f32;
        let size = (PREFILTER_BASE_SIZE >> mip).max(1);
        for face in 0..6 {
            let mut rgba = vec![0u8; (size * size * 4) as usize];
            for y in 0..size {
                for x in 0..size {
                    let dir = cube_dir(face, x, y, size);
                    let color = prefilter_env_sample(env_map, dir, roughness);
                    let idx = ((y * size + x) * 4) as usize;
                    rgba[idx] = linear_to_srgb8(color[0]);
                    rgba[idx + 1] = linear_to_srgb8(color[1]);
                    rgba[idx + 2] = linear_to_srgb8(color[2]);
                    rgba[idx + 3] = 255;
                }
            }
            all_faces.push(rgba);
        }
    }
    all_faces
}

fn prefilter_env_sample(env_map: &EnvMap, reflection: [f32; 3], roughness: f32) -> [f32; 3] {
    let n = normalize(reflection);
    let v = n; // Assume V = N for prefiltering (split-sum assumption)
    let (up, right) = make_tangent_frame(n);

    let sample_count = if roughness < 0.05 { 32u32 } else { 128 };
    let mut result = [0.0f32; 3];
    let mut total_weight = 0.0f32;

    for i in 0..sample_count {
        let xi = hammersley(i, sample_count);
        let h = importance_sample_ggx(xi, n, up, right, roughness);
        let l = reflect_over(v, h);
        let n_dot_l = dot(n, l).max(0.0);
        if n_dot_l > 0.0 {
            let color = env_map.sample(l);
            result[0] += color[0] * n_dot_l;
            result[1] += color[1] * n_dot_l;
            result[2] += color[2] * n_dot_l;
            total_weight += n_dot_l;
        }
    }

    if total_weight > 0.0 {
        result[0] /= total_weight;
        result[1] /= total_weight;
        result[2] /= total_weight;
    }
    result
}

// ── BRDF Integration LUT ───────────────────────────────────────────

fn compute_brdf_lut() -> Vec<u8> {
    let size = BRDF_LUT_SIZE;
    let mut rgba = vec![0u8; (size * size * 4) as usize];
    let sample_count = 256u32;

    for y in 0..size {
        for x in 0..size {
            let n_dot_v = (x as f32 + 0.5) / size as f32;
            let roughness = (y as f32 + 0.5) / size as f32;
            let n_dot_v = n_dot_v.max(0.001);
            let (scale, bias) = integrate_brdf(n_dot_v, roughness, sample_count);
            let idx = ((y * size + x) * 4) as usize;
            rgba[idx] = (scale.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            rgba[idx + 1] = (bias.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            rgba[idx + 2] = 0;
            rgba[idx + 3] = 255;
        }
    }
    rgba
}

fn integrate_brdf(n_dot_v: f32, roughness: f32, sample_count: u32) -> (f32, f32) {
    let v = [
        (1.0 - n_dot_v * n_dot_v).sqrt(),
        0.0,
        n_dot_v,
    ];
    let n = [0.0f32, 0.0, 1.0];
    let (up, right) = ([0.0f32, 1.0, 0.0], [1.0, 0.0, 0.0]);

    let mut a = 0.0f32;
    let mut b = 0.0f32;

    for i in 0..sample_count {
        let xi = hammersley(i, sample_count);
        let h = importance_sample_ggx(xi, n, up, right, roughness);
        let l = reflect_over(v, h);
        let n_dot_l = l[2].max(0.0);
        let n_dot_h = h[2].max(0.0);
        let v_dot_h = dot(v, h).max(0.0);

        if n_dot_l > 0.0 {
            let g = geometry_smith_ibl(n_dot_v, n_dot_l, roughness);
            let g_vis = (g * v_dot_h) / (n_dot_h * n_dot_v + 0.0001);
            let fc = (1.0 - v_dot_h).powi(5);
            a += (1.0 - fc) * g_vis;
            b += fc * g_vis;
        }
    }

    (a / sample_count as f32, b / sample_count as f32)
}

fn geometry_smith_ibl(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    let a = roughness;
    let k = (a * a) / 2.0;
    let ggx_v = n_dot_v / (n_dot_v * (1.0 - k) + k + 0.0001);
    let ggx_l = n_dot_l / (n_dot_l * (1.0 - k) + k + 0.0001);
    ggx_v * ggx_l
}

// ── Cubemap direction helper ────────────────────────────────────────

/// Convert a face index (0..5) and pixel coordinate to a world-space direction.
/// Face order: +X, -X, +Y, -Y, +Z, -Z (same as WebGL cubemap convention).
fn cube_dir(face: u32, x: u32, y: u32, size: u32) -> [f32; 3] {
    let u = (x as f32 + 0.5) / size as f32 * 2.0 - 1.0;
    let v = (y as f32 + 0.5) / size as f32 * 2.0 - 1.0;
    let dir = match face {
        0 => [1.0, -v, -u],  // +X
        1 => [-1.0, -v, u],  // -X
        2 => [u, 1.0, v],    // +Y
        3 => [u, -1.0, -v],  // -Y
        4 => [u, -v, 1.0],   // +Z
        _ => [-u, -v, -1.0], // -Z
    };
    normalize(dir)
}

// ── Math helpers ────────────────────────────────────────────────────

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len < 1e-8 {
        return [0.0, 1.0, 0.0];
    }
    [v[0] / len, v[1] / len, v[2] / len]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn reflect_over(v: [f32; 3], h: [f32; 3]) -> [f32; 3] {
    let d = 2.0 * dot(v, h);
    normalize([d * h[0] - v[0], d * h[1] - v[1], d * h[2] - v[2]])
}

fn make_tangent_frame(n: [f32; 3]) -> ([f32; 3], [f32; 3]) {
    let up = if n[1].abs() < 0.999 { [0.0, 1.0, 0.0] } else { [1.0, 0.0, 0.0] };
    let right = normalize(cross(up, n));
    let up = cross(n, right);
    (up, right)
}

fn hammersley(i: u32, n: u32) -> [f32; 2] {
    [i as f32 / n as f32, radical_inverse_vdc(i)]
}

fn radical_inverse_vdc(mut bits: u32) -> f32 {
    bits = (bits << 16) | (bits >> 16);
    bits = ((bits & 0x55555555) << 1) | ((bits & 0xAAAAAAAA) >> 1);
    bits = ((bits & 0x33333333) << 2) | ((bits & 0xCCCCCCCC) >> 2);
    bits = ((bits & 0x0F0F0F0F) << 4) | ((bits & 0xF0F0F0F0) >> 4);
    bits = ((bits & 0x00FF00FF) << 8) | ((bits & 0xFF00FF00) >> 8);
    bits as f32 * 2.3283064365386963e-10
}

fn cosine_sample_hemisphere(xi: [f32; 2]) -> (f32, f32) {
    let cos_theta = (1.0 - xi[1]).sqrt();
    let sin_theta = xi[1].sqrt();
    (sin_theta, cos_theta)
}

fn importance_sample_ggx(
    xi: [f32; 2],
    n: [f32; 3],
    up: [f32; 3],
    right: [f32; 3],
    roughness: f32,
) -> [f32; 3] {
    let a = roughness * roughness;
    let phi = 2.0 * std::f32::consts::PI * xi[0];
    let cos_theta = ((1.0 - xi[1]) / (1.0 + (a * a - 1.0) * xi[1])).sqrt();
    let sin_theta = (1.0 - cos_theta * cos_theta).sqrt().max(0.0);
    let h = [
        right[0] * phi.cos() * sin_theta + up[0] * phi.sin() * sin_theta + n[0] * cos_theta,
        right[1] * phi.cos() * sin_theta + up[1] * phi.sin() * sin_theta + n[1] * cos_theta,
        right[2] * phi.cos() * sin_theta + up[2] * phi.sin() * sin_theta + n[2] * cos_theta,
    ];
    normalize(h)
}

// ── Color space ─────────────────────────────────────────────────────

fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb8(c: f32) -> u8 {
    let c = c.max(0.0);
    let s = if c <= 0.0031308 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };
    (s.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

fn half_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x3FF) as u32;

    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign << 31);
        }
        // Subnormal
        let mut e = 0i32;
        let mut m = mant;
        while (m & 0x400) == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x3FF;
        let f_exp = (127 - 15 + 1 + e) as u32;
        return f32::from_bits((sign << 31) | (f_exp << 23) | (m << 13));
    }
    if exp == 31 {
        if mant == 0 {
            return f32::from_bits((sign << 31) | (0xFF << 23));
        }
        return f32::NAN;
    }
    let f_exp = exp + (127 - 15);
    f32::from_bits((sign << 31) | (f_exp << 23) | (mant << 13))
}
