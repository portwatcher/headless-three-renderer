use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;

use anyhow::{Context, Result};

use crate::pmrem::{Equirect, EquirectSampling, equirect_to_cube_uv};

/// Image-based lighting data, prepared on the CPU from an equirectangular HDR/LDR environment as
/// Three.js r180 WebGLRenderer does:
/// 1. the PMREM CubeUV atlas (`PMREMGenerator.fromEquirectangular`) that MeshStandardMaterial and
///    MeshPhysicalMaterial sample with `textureCubeUV` for diffuse and specular light, and
/// 2. the unblurred cube map (`WebGLCubeMaps`) for the legacy MeshBasic/Lambert/Phong envMap.
///
/// Both hold linear RGBA16F texels, so HDR values above 1 survive.
const MAX_ENV_CUBE_SIZE: u32 = 512;
const MAX_IBL_CACHE_ENTRIES: usize = 8;

static IBL_MAPS: OnceLock<Mutex<HashMap<IblCacheKey, Arc<IblMaps>>>> = OnceLock::new();

pub struct IblMaps {
    /// Legacy environment cube: 6 faces of RGBA16F texels (little-endian bytes), WebGPU face order.
    pub env_cube_faces: Vec<Vec<u8>>,
    pub env_cube_size: u32,
    /// PMREM CubeUV atlas as RGBA16F bytes; row 0 is the bottom row of the WebGL render target.
    pub cube_uv: Vec<u8>,
    pub cube_uv_width: u32,
    pub cube_uv_height: u32,
    /// `CUBEUV_MAX_MIP`: log2 of the PMREM cube face size.
    pub cube_uv_max_mip: f32,
    /// Hash of the source pixels and sampling state, for GPU upload caches.
    pub content_key: u64,
}

/// Which environment representations the scene's materials sample.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct IblNeeds {
    pub cube_uv: bool,
    pub env_cube: bool,
}

/// An HDR equirect environment map stored as linear f32 RGB pixels in data order.
pub struct EnvMap {
    pub pixels: Vec<[f32; 3]>,
    pub width: u32,
    pub height: u32,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct IblCacheKey {
    width: u32,
    height: u32,
    pixels_hash: u64,
    sampling: EquirectSampling,
    needs: IblNeeds,
}

impl IblCacheKey {
    fn new(env_map: &EnvMap, sampling: EquirectSampling, needs: IblNeeds) -> Self {
        let mut hasher = DefaultHasher::new();
        for pixel in &env_map.pixels {
            for channel in pixel {
                f32_key(*channel).hash(&mut hasher);
            }
        }
        Self {
            width: env_map.width,
            height: env_map.height,
            pixels_hash: hasher.finish(),
            sampling,
            needs,
        }
    }

    fn content_key(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.hash(&mut hasher);
        hasher.finish()
    }
}

fn f32_key(value: f32) -> u32 {
    if value == 0.0 { 0 } else { value.to_bits() }
}

impl EnvMap {
    /// Decode from raw image bytes (PNG, JPEG, WebP, or HDR Radiance).
    /// Also accepts raw RGBA8 bytes if width/height hints are given.
    pub fn from_bytes(
        data: &[u8],
        width_hint: Option<u32>,
        height_hint: Option<u32>,
        is_srgb: bool,
    ) -> Result<Self> {
        let w = width_hint.unwrap_or(0);
        let h = height_hint.unwrap_or(0);

        // Raw RGBA8 bytes?
        if w > 0 && h > 0 && data.len() == (w as usize) * (h as usize) * 4 {
            let mut pixels = Vec::with_capacity((w * h) as usize);
            for i in 0..(w * h) as usize {
                pixels.push([
                    decode_ldr_environment_channel(data[i * 4], is_srgb),
                    decode_ldr_environment_channel(data[i * 4 + 1], is_srgb),
                    decode_ldr_environment_channel(data[i * 4 + 2], is_srgb),
                ]);
            }
            return Ok(Self {
                pixels,
                width: w,
                height: h,
            });
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
            return Ok(Self {
                pixels,
                width: w,
                height: h,
            });
        }

        // Raw RGBA32F?
        if w > 0 && h > 0 && data.len() == (w as usize) * (h as usize) * 16 {
            let mut pixels = Vec::with_capacity((w * h) as usize);
            for i in 0..(w * h) as usize {
                let offset = i * 16;
                let r = f32::from_le_bytes([
                    data[offset],
                    data[offset + 1],
                    data[offset + 2],
                    data[offset + 3],
                ]);
                let g = f32::from_le_bytes([
                    data[offset + 4],
                    data[offset + 5],
                    data[offset + 6],
                    data[offset + 7],
                ]);
                let b = f32::from_le_bytes([
                    data[offset + 8],
                    data[offset + 9],
                    data[offset + 10],
                    data[offset + 11],
                ]);
                pixels.push([r, g, b]);
            }
            return Ok(Self {
                pixels,
                width: w,
                height: h,
            });
        }

        // Try decoding as an image file
        let img =
            image::load_from_memory(data).context("failed to decode environment map image")?;
        let rgba = img.to_rgba8();
        let w = rgba.width();
        let h = rgba.height();
        let raw = rgba.into_raw();
        let mut pixels = Vec::with_capacity((w * h) as usize);
        for i in 0..(w * h) as usize {
            pixels.push([
                decode_ldr_environment_channel(raw[i * 4], is_srgb),
                decode_ldr_environment_channel(raw[i * 4 + 1], is_srgb),
                decode_ldr_environment_channel(raw[i * 4 + 2], is_srgb),
            ]);
        }
        Ok(Self {
            pixels,
            width: w,
            height: h,
        })
    }
}

fn decode_ldr_environment_channel(value: u8, is_srgb: bool) -> f32 {
    let normalized = value as f32 / 255.0;
    if is_srgb {
        srgb_to_linear(normalized)
    } else {
        normalized
    }
}

pub fn compute_ibl(env_map: &EnvMap, sampling: EquirectSampling, needs: IblNeeds) -> Arc<IblMaps> {
    let key = IblCacheKey::new(env_map, sampling, needs);
    let cache = IBL_MAPS.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(maps) = cache
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .get(&key)
        .cloned()
    {
        return maps;
    }

    let maps = Arc::new(compute_ibl_uncached(
        env_map,
        sampling,
        needs,
        key.content_key(),
    ));
    let mut guard = cache
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if guard.len() >= MAX_IBL_CACHE_ENTRIES && !guard.contains_key(&key) {
        guard.clear();
    }
    guard.entry(key).or_insert_with(|| maps.clone()).clone()
}

fn compute_ibl_uncached(
    env_map: &EnvMap,
    sampling: EquirectSampling,
    needs: IblNeeds,
    content_key: u64,
) -> IblMaps {
    let source = Equirect {
        pixels: &env_map.pixels,
        width: env_map.width,
        height: env_map.height,
        sampling,
    };
    // Without an atlas (no standard material uses it, or an input narrower than 64 px like
    // Three.js) a 1x1 black atlas gives no image-based light.
    let atlas = if needs.cube_uv {
        equirect_to_cube_uv(&source)
    } else {
        None
    };
    let (cube_uv, cube_uv_width, cube_uv_height, cube_uv_max_mip) = match atlas {
        Some(atlas) => (
            rgba16f_bytes(&atlas.texels),
            atlas.width,
            atlas.height,
            atlas.lod_max as f32,
        ),
        None => (rgba16f_bytes(&[[0.0; 3]]), 1, 1, 0.0),
    };
    let (env_cube_faces, env_cube_size) = if needs.env_cube {
        let size = env_map.height.clamp(1, MAX_ENV_CUBE_SIZE);
        (equirect_to_cube_faces(&source, size), size)
    } else {
        (vec![rgba16f_bytes(&[[0.0; 3]]); 6], 1)
    };
    IblMaps {
        env_cube_faces,
        env_cube_size,
        cube_uv,
        cube_uv_width,
        cube_uv_height,
        cube_uv_max_mip,
        content_key,
    }
}

/// `WebGLCubeRenderTarget.fromEquirectangularTexture`: each cube texel samples the equirect in
/// its direction. Faces follow the WebGPU order +X, -X, +Y, -Y, +Z, -Z.
fn equirect_to_cube_faces(source: &Equirect<'_>, size: u32) -> Vec<Vec<u8>> {
    thread::scope(|scope| {
        let handles = (0..6u32)
            .map(|face| {
                scope.spawn(move || {
                    let mut texels = Vec::with_capacity((size * size) as usize);
                    for y in 0..size {
                        for x in 0..size {
                            texels.push(source.sample_direction(cube_dir(face, x, y, size)));
                        }
                    }
                    rgba16f_bytes(&texels)
                })
            })
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .map(|handle| handle.join().expect("environment cube worker panicked"))
            .collect()
    })
}

/// Direction of texel (x, y) on a WebGPU cube face.
fn cube_dir(face: u32, x: u32, y: u32, size: u32) -> [f32; 3] {
    let u = (x as f32 + 0.5) / size as f32 * 2.0 - 1.0;
    let v = (y as f32 + 0.5) / size as f32 * 2.0 - 1.0;
    match face {
        0 => [1.0, -v, -u],
        1 => [-1.0, -v, u],
        2 => [u, 1.0, v],
        3 => [u, -1.0, -v],
        4 => [u, -v, 1.0],
        _ => [-u, -v, -1.0],
    }
}

fn rgba16f_bytes(texels: &[[f32; 3]]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(texels.len() * 8);
    for texel in texels {
        for value in [texel[0], texel[1], texel[2], 1.0] {
            bytes.extend_from_slice(&f32_to_f16(value).to_le_bytes());
        }
    }
    bytes
}

/// IEEE 754 binary16 with round-to-nearest-even.
pub(crate) fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x7f_ffff;
    if exponent == 0xff {
        return sign | 0x7c00 | if mantissa != 0 { 0x200 } else { 0 };
    }
    let half_exponent = exponent - 127 + 15;
    if half_exponent >= 0x1f {
        return sign | 0x7c00;
    }
    if half_exponent <= 0 {
        if half_exponent < -10 {
            return sign;
        }
        let full = mantissa | 0x80_0000;
        let shift = (14 - half_exponent) as u32;
        let truncated = full >> shift;
        let remainder = full & ((1 << shift) - 1);
        let halfway = 1 << (shift - 1);
        let rounded = if remainder > halfway || (remainder == halfway && truncated & 1 == 1) {
            truncated + 1
        } else {
            truncated
        };
        return sign | rounded as u16;
    }
    let truncated = mantissa >> 13;
    let remainder = mantissa & 0x1fff;
    let mut result = ((half_exponent as u32) << 10) | truncated;
    if remainder > 0x1000 || (remainder == 0x1000 && truncated & 1 == 1) {
        result += 1;
    }
    sign | result as u16
}

fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
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

#[cfg(test)]
mod tests {
    use super::{EnvMap, IblCacheKey, IblNeeds, f32_to_f16, half_to_f32};
    use crate::pmrem::{EnvWrap, EquirectSampling};

    const SAMPLING: EquirectSampling = EquirectSampling {
        flip_y: false,
        linear: true,
        wrap_s: EnvWrap::Clamp,
        wrap_t: EnvWrap::Clamp,
    };
    const NEEDS: IblNeeds = IblNeeds {
        cube_uv: true,
        env_cube: false,
    };

    #[test]
    fn ibl_cache_keys_track_pixels_and_sampling() {
        let env = EnvMap {
            pixels: vec![[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            width: 2,
            height: 1,
        };
        assert_eq!(
            IblCacheKey::new(&env, SAMPLING, NEEDS),
            IblCacheKey::new(&env, SAMPLING, NEEDS)
        );
        let flipped = EquirectSampling {
            flip_y: true,
            ..SAMPLING
        };
        assert_ne!(
            IblCacheKey::new(&env, SAMPLING, NEEDS),
            IblCacheKey::new(&env, flipped, NEEDS)
        );
        let both = IblNeeds {
            cube_uv: true,
            env_cube: true,
        };
        assert_ne!(
            IblCacheKey::new(&env, SAMPLING, NEEDS),
            IblCacheKey::new(&env, SAMPLING, both)
        );
        let changed_env = EnvMap {
            pixels: vec![[0.1, 0.2, 0.35], [0.4, 0.5, 0.6]],
            width: 2,
            height: 1,
        };
        assert_ne!(
            IblCacheKey::new(&env, SAMPLING, NEEDS),
            IblCacheKey::new(&changed_env, SAMPLING, NEEDS)
        );
    }

    #[test]
    fn half_float_conversion_round_trips_hdr_values() {
        for value in [0.0f32, 1.0, 0.5, 0.001, 12.0, 1000.0, -3.25, 6.0e-6] {
            let round_trip = half_to_f32(f32_to_f16(value));
            assert!(
                (round_trip - value).abs() <= value.abs() * 1e-3 + 1e-7,
                "{value} -> {round_trip}"
            );
        }
        assert_eq!(f32_to_f16(1.0), 0x3c00);
        assert_eq!(f32_to_f16(65520.0), 0x7c00);
    }
}
