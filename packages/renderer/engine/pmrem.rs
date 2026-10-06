//! CPU port of the Three.js r180 `PMREMGenerator` (MIT) for equirectangular inputs.
//!
//! `WebGLRenderer` converts `scene.environment` and `MeshStandardMaterial.envMap` into a
//! prefiltered CubeUV atlas: level 0 holds the environment as six cube faces with a one-texel
//! border, each later level is a two-pass spherical Gaussian blur of the previous one, and six
//! extra 16-pixel levels hold the rough reflections and the diffuse lookup (roughness 1). The
//! renderer samples the atlas with the same `textureCubeUV` function (see `shader/ibl.wgsl`).
//! Rows are stored bottom-up as in a WebGL render target: row 0 is GL window y 0.

use std::f32::consts::PI;
use std::thread;

const LOD_MIN: i32 = 4;

/// Three.js `PMREMGenerator.fromCubemap` builds a usable CubeUV map only for cube faces of at
/// least 2^LOD_MIN (16) px; smaller cubes give no image-based light in WebGL.
pub fn cube_face_size_has_atlas(face_size: u32) -> bool {
    face_size >= 1 << LOD_MIN
}

// Standard deviations (radians) of the extra levels; they approximate GGX at high roughness.
const EXTRA_LOD_SIGMA: [f32; 6] = [0.125, 0.215, 0.35, 0.446, 0.526, 0.582];
const MAX_SAMPLES: usize = 20;
const PHI: f32 = 1.618_034;
const INV_PHI: f32 = 1.0 / PHI;
// Dodecahedron vertices (without opposites): blur pole axes spread over the sphere.
const AXIS_DIRECTIONS: [[f32; 3]; 10] = [
    [-PHI, INV_PHI, 0.0],
    [PHI, INV_PHI, 0.0],
    [-INV_PHI, 0.0, PHI],
    [INV_PHI, 0.0, PHI],
    [0.0, PHI, -INV_PHI],
    [0.0, PHI, INV_PHI],
    [-1.0, 1.0, -1.0],
    [1.0, 1.0, -1.0],
    [-1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0],
];

/// Texture wrap mode of the equirectangular source, as Three.js wrapS/wrapT.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum EnvWrap {
    Clamp,
    Repeat,
    Mirror,
}

impl EnvWrap {
    pub fn from_str_opt(value: Option<&str>) -> Self {
        match value {
            Some("repeat") => Self::Repeat,
            Some("mirror") => Self::Mirror,
            _ => Self::Clamp,
        }
    }

    fn index(self, index: i64, size: i64) -> usize {
        let wrapped = match self {
            Self::Clamp => index.clamp(0, size - 1),
            Self::Repeat => index.rem_euclid(size),
            Self::Mirror => {
                let period = index.rem_euclid(2 * size);
                if period < size {
                    period
                } else {
                    2 * size - 1 - period
                }
            }
        };
        wrapped as usize
    }
}

/// How WebGL samples the equirectangular texture: `flipY`, filter, and wrap modes.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct EquirectSampling {
    /// Three.js `texture.flipY`: when true, data row 0 is the top of the image (GL row H-1).
    pub flip_y: bool,
    /// Bilinear (LinearFilter) or nearest (NearestFilter) sampling.
    pub linear: bool,
    pub wrap_s: EnvWrap,
    pub wrap_t: EnvWrap,
}

/// Linear RGB equirectangular pixels in data order, sampled like a WebGL texture.
pub struct Equirect<'a> {
    pub pixels: &'a [[f32; 3]],
    pub width: u32,
    pub height: u32,
    pub sampling: EquirectSampling,
}

impl Equirect<'_> {
    fn texel(&self, x: i64, gl_y: i64) -> [f32; 3] {
        let w = self.width as i64;
        let h = self.height as i64;
        let x = self.sampling.wrap_s.index(x, w);
        let gl_y = self.sampling.wrap_t.index(gl_y, h);
        let row = if self.sampling.flip_y {
            h as usize - 1 - gl_y
        } else {
            gl_y
        };
        self.pixels[row * self.width as usize + x]
    }

    /// `texture2D(envMap, uv)` with GL texture coordinates (v = 0 is GL row 0).
    pub fn sample(&self, u: f32, v: f32) -> [f32; 3] {
        let fx = u * self.width as f32;
        let fy = v * self.height as f32;
        if !self.sampling.linear {
            return self.texel(fx.floor() as i64, fy.floor() as i64);
        }
        let x = fx - 0.5;
        let y = fy - 0.5;
        let x0 = x.floor();
        let y0 = y.floor();
        let tx = x - x0;
        let ty = y - y0;
        let (x0, y0) = (x0 as i64, y0 as i64);
        let c00 = self.texel(x0, y0);
        let c10 = self.texel(x0 + 1, y0);
        let c01 = self.texel(x0, y0 + 1);
        let c11 = self.texel(x0 + 1, y0 + 1);
        let mut out = [0.0; 3];
        for c in 0..3 {
            let top = c00[c] + (c10[c] - c00[c]) * tx;
            let bottom = c01[c] + (c11[c] - c01[c]) * tx;
            out[c] = top + (bottom - top) * ty;
        }
        out
    }

    /// Three.js `equirectUv` for a unit direction, then `texture2D`.
    pub fn sample_direction(&self, direction: [f32; 3]) -> [f32; 3] {
        let d = normalize(direction);
        let u = d[2].atan2(d[0]) * (0.5 / PI) + 0.5;
        let v = d[1].clamp(-1.0, 1.0).asin() / PI + 0.5;
        self.sample(u, v)
    }
}

/// The CubeUV atlas of `PMREMGenerator.fromEquirectangular`, in linear RGB.
pub struct CubeUvAtlas {
    pub width: u32,
    pub height: u32,
    /// `CUBEUV_MAX_MIP`: log2 of the cube face size of level 0.
    pub lod_max: i32,
    pub texels: Vec<[f32; 3]>,
}

impl CubeUvAtlas {
    fn new(width: u32, height: u32, lod_max: i32) -> Self {
        Self {
            width,
            height,
            lod_max,
            texels: vec![[0.0; 3]; (width * height) as usize],
        }
    }

    fn fetch(&self, x: i64, y: i64) -> [f32; 3] {
        let x = x.clamp(0, self.width as i64 - 1) as usize;
        let y = y.clamp(0, self.height as i64 - 1) as usize;
        self.texels[y * self.width as usize + x]
    }

    /// Bilinear `texture2D` with ClampToEdge at a coordinate in texel units.
    fn bilinear(&self, x: f32, y: f32) -> [f32; 3] {
        let x = x - 0.5;
        let y = y - 0.5;
        let x0 = x.floor();
        let y0 = y.floor();
        let tx = x - x0;
        let ty = y - y0;
        let (x0, y0) = (x0 as i64, y0 as i64);
        let c00 = self.fetch(x0, y0);
        let c10 = self.fetch(x0 + 1, y0);
        let c01 = self.fetch(x0, y0 + 1);
        let c11 = self.fetch(x0 + 1, y0 + 1);
        let mut out = [0.0; 3];
        for c in 0..3 {
            let bottom = c00[c] + (c10[c] - c00[c]) * tx;
            let top = c01[c] + (c11[c] - c01[c]) * tx;
            out[c] = bottom + (top - bottom) * ty;
        }
        out
    }

    /// Three.js `bilinearCubeUV`.
    fn bilinear_cube_uv(&self, direction: [f32; 3], mip_int: f32) -> [f32; 3] {
        let mut face = face_of(direction);
        let filter_int = (LOD_MIN as f32 - mip_int).max(0.0);
        let mip_int = mip_int.max(LOD_MIN as f32);
        let face_size = mip_int.exp2();
        let uv = face_uv(direction, face);
        let mut x = uv[0] * (face_size - 2.0) + 1.0;
        let mut y = uv[1] * (face_size - 2.0) + 1.0;
        if face > 2 {
            y += face_size;
            face -= 3;
        }
        x += face as f32 * face_size;
        x += filter_int * 3.0 * 16.0;
        y += 4.0 * ((self.lod_max as f32).exp2() - face_size);
        self.bilinear(x, y)
    }
}

/// Runs `PMREMGenerator.fromEquirectangular` on the CPU.
///
/// Returns `None` for inputs narrower than 64 px: their Three.js cube is smaller than LOD_MIN
/// (16 px), `textureCubeUV` reads outside the atlas, and WebGL renders no image-based light.
pub fn equirect_to_cube_uv(source: &Equirect<'_>) -> Option<CubeUvAtlas> {
    // _setSize( texture.image.width / 4 )
    let lod_max = (source.width as f32 / 4.0).log2().floor() as i32;
    if lod_max < LOD_MIN {
        return None;
    }
    let cube_size = 1u32 << lod_max;
    let width = 3 * cube_size.max(16 * 7);
    let height = 4 * cube_size;
    let lods = create_lods(lod_max);

    let mut atlas = CubeUvAtlas::new(width, height, lod_max);
    texture_to_cube_uv(source, &mut atlas, cube_size);

    let mut ping_pong = CubeUvAtlas::new(width, height, lod_max);
    let n = lods.len();
    for i in 1..n {
        let sigma = (lods[i].sigma * lods[i].sigma - lods[i - 1].sigma * lods[i - 1].sigma).sqrt();
        let pole_axis = AXIS_DIRECTIONS[(n - i - 1) % AXIS_DIRECTIONS.len()];
        half_blur(
            &atlas,
            &mut ping_pong,
            &lods,
            i - 1,
            i,
            sigma,
            true,
            pole_axis,
            cube_size,
        );
        half_blur(
            &ping_pong, &mut atlas, &lods, i, i, sigma, false, pole_axis, cube_size,
        );
    }
    Some(atlas)
}

#[derive(Copy, Clone)]
struct Lod {
    size: u32,
    sigma: f32,
}

/// Three.js `_createPlanes`: face sizes and blur sigmas of all levels.
fn create_lods(lod_max: i32) -> Vec<Lod> {
    let total = (lod_max - LOD_MIN + 1 + EXTRA_LOD_SIGMA.len() as i32).max(1) as usize;
    let mut lod = lod_max;
    let mut lods = Vec::with_capacity(total);
    for i in 0..total as i32 {
        let size = 1u32 << lod.max(0);
        let mut sigma = 1.0 / size as f32;
        if i > lod_max - LOD_MIN {
            let extra = (i - lod_max + LOD_MIN - 1).clamp(0, EXTRA_LOD_SIGMA.len() as i32 - 1);
            sigma = EXTRA_LOD_SIGMA[extra as usize];
        } else if i == 0 {
            sigma = 0.0;
        }
        lods.push(Lod { size, sigma });
        if lod > LOD_MIN {
            lod -= 1;
        }
    }
    lods
}

/// `_textureToCubeUV` with the equirectangular material: level 0 with borders.
fn texture_to_cube_uv(source: &Equirect<'_>, atlas: &mut CubeUvAtlas, size: u32) {
    let faces: Vec<Vec<[f32; 3]>> = thread::scope(|scope| {
        let handles = (0..6u32)
            .map(|face| {
                scope.spawn(move || {
                    let mut texels = Vec::with_capacity((size * size) as usize);
                    for k in 0..size {
                        for j in 0..size {
                            let direction = level_direction(j, k, size, face);
                            texels.push(source.sample_direction(direction));
                        }
                    }
                    texels
                })
            })
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .map(|handle| handle.join().expect("PMREM level 0 worker panicked"))
            .collect()
    });
    for (face, texels) in faces.into_iter().enumerate() {
        let x0 = (face as u32 % 3) * size;
        let y0 = if face > 2 { size } else { 0 };
        for k in 0..size {
            for j in 0..size {
                let index = ((y0 + k) * atlas.width + x0 + j) as usize;
                atlas.texels[index] = texels[(k * size + j) as usize];
            }
        }
    }
}

/// Output direction of texel (j, k) of a face quad of a level with `size` pixels. The quad UVs
/// run from -1/(size-2) to 1+1/(size-2), so the outer texels are borders beyond the face.
fn level_direction(j: u32, k: u32, size: u32, face: u32) -> [f32; 3] {
    let inner = (size as f32 - 2.0).max(1.0);
    let u = (j as f32 - 0.5) / inner;
    let v = (k as f32 - 0.5) / inner;
    get_direction([u, v], face)
}

/// Three.js PMREM `getDirection` (not normalized).
fn get_direction(uv: [f32; 2], face: u32) -> [f32; 3] {
    let u = 2.0 * uv[0] - 1.0;
    let v = 2.0 * uv[1] - 1.0;
    match face {
        0 => [1.0, v, u],
        1 => [-u, 1.0, -v],
        2 => [-u, v, 1.0],
        3 => [-1.0, v, -u],
        4 => [-u, -1.0, v],
        _ => [u, v, -1.0],
    }
}

#[allow(clippy::too_many_arguments)]
fn half_blur(
    source: &CubeUvAtlas,
    target: &mut CubeUvAtlas,
    lods: &[Lod],
    lod_in: usize,
    lod_out: usize,
    sigma_radians: f32,
    latitudinal: bool,
    pole_axis: [f32; 3],
    cube_size: u32,
) {
    const STANDARD_DEVIATIONS: f32 = 3.0;
    let pixels = lods[lod_in].size as f32 - 1.0;
    let finite = sigma_radians.is_finite();
    let radians_per_pixel = if finite {
        PI / (2.0 * pixels)
    } else {
        2.0 * PI / (2.0 * MAX_SAMPLES as f32 - 1.0)
    };
    let sigma_pixels = sigma_radians / radians_per_pixel;
    let samples = if finite {
        1 + (STANDARD_DEVIATIONS * sigma_pixels).floor() as usize
    } else {
        MAX_SAMPLES
    };
    let mut weights = [0.0f32; MAX_SAMPLES];
    let mut sum = 0.0;
    for (i, weight) in weights.iter_mut().enumerate() {
        let x = i as f32 / sigma_pixels;
        *weight = (-x * x / 2.0).exp();
        if i == 0 {
            sum += *weight;
        } else if i < samples {
            sum += 2.0 * *weight;
        }
    }
    for weight in &mut weights {
        *weight /= sum;
    }
    let samples = samples.min(MAX_SAMPLES);
    let lod_max = source.lod_max;
    // mipInt = lodMax - lodIn; below LOD_MIN it selects the extra filtered tiles.
    let mip_int = (lod_max - lod_in as i32) as f32;
    let output_size = lods[lod_out].size;
    let extra_index = if lod_out as i32 > lod_max - LOD_MIN {
        lod_out as i32 - lod_max + LOD_MIN
    } else {
        0
    } as u32;
    let x_origin = 3 * output_size * extra_index;
    let y_origin = 4 * (cube_size - output_size);

    let faces: Vec<Vec<[f32; 3]>> = thread::scope(|scope| {
        let handles = (0..6u32)
            .map(|face| {
                let weights = &weights;
                scope.spawn(move || {
                    let mut texels = Vec::with_capacity((output_size * output_size) as usize);
                    for k in 0..output_size {
                        for j in 0..output_size {
                            let direction = level_direction(j, k, output_size, face);
                            texels.push(blur_texel(
                                source,
                                direction,
                                latitudinal,
                                pole_axis,
                                weights,
                                samples,
                                radians_per_pixel,
                                mip_int,
                            ));
                        }
                    }
                    texels
                })
            })
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .map(|handle| handle.join().expect("PMREM blur worker panicked"))
            .collect()
    });
    for (face, texels) in faces.into_iter().enumerate() {
        let x0 = x_origin + (face as u32 % 3) * output_size;
        let y0 = y_origin + if face > 2 { output_size } else { 0 };
        for k in 0..output_size {
            for j in 0..output_size {
                let x = x0 + j;
                let y = y0 + k;
                if x < target.width && y < target.height {
                    target.texels[(y * target.width + x) as usize] =
                        texels[(k * output_size + j) as usize];
                }
            }
        }
    }
}

/// The SphericalGaussianBlur fragment shader for one output direction.
#[allow(clippy::too_many_arguments)]
fn blur_texel(
    source: &CubeUvAtlas,
    output_direction: [f32; 3],
    latitudinal: bool,
    pole_axis: [f32; 3],
    weights: &[f32; MAX_SAMPLES],
    samples: usize,
    d_theta: f32,
    mip_int: f32,
) -> [f32; 3] {
    let mut axis = if latitudinal {
        pole_axis
    } else {
        cross(pole_axis, output_direction)
    };
    if axis == [0.0, 0.0, 0.0] {
        axis = [output_direction[2], 0.0, -output_direction[0]];
    }
    let axis = normalize(axis);
    let sample = |theta: f32| -> [f32; 3] {
        let cos_theta = theta.cos();
        let sin_theta = theta.sin();
        let c = cross(axis, output_direction);
        let d = dot(axis, output_direction) * (1.0 - cos_theta);
        let direction = [
            output_direction[0] * cos_theta + c[0] * sin_theta + axis[0] * d,
            output_direction[1] * cos_theta + c[1] * sin_theta + axis[1] * d,
            output_direction[2] * cos_theta + c[2] * sin_theta + axis[2] * d,
        ];
        source.bilinear_cube_uv(direction, mip_int)
    };
    let center = sample(0.0);
    let mut color = [
        center[0] * weights[0],
        center[1] * weights[0],
        center[2] * weights[0],
    ];
    for i in 1..samples {
        let theta = d_theta * i as f32;
        let a = sample(-theta);
        let b = sample(theta);
        for c in 0..3 {
            color[c] += weights[i] * a[c] + weights[i] * b[c];
        }
    }
    color
}

/// Three.js `getFace` of cube_uv_reflection_fragment.
fn face_of(direction: [f32; 3]) -> u32 {
    let a = [direction[0].abs(), direction[1].abs(), direction[2].abs()];
    if a[0] > a[2] {
        if a[0] > a[1] {
            if direction[0] > 0.0 { 0 } else { 3 }
        } else if direction[1] > 0.0 {
            1
        } else {
            4
        }
    } else if a[2] > a[1] {
        if direction[2] > 0.0 { 2 } else { 5 }
    } else if direction[1] > 0.0 {
        1
    } else {
        4
    }
}

/// Three.js `getUV` of cube_uv_reflection_fragment.
fn face_uv(d: [f32; 3], face: u32) -> [f32; 2] {
    let uv = match face {
        0 => [d[2] / d[0].abs(), d[1] / d[0].abs()],
        1 => [-d[0] / d[1].abs(), -d[2] / d[1].abs()],
        2 => [-d[0] / d[2].abs(), d[1] / d[2].abs()],
        3 => [-d[2] / d[0].abs(), d[1] / d[0].abs()],
        4 => [-d[0] / d[1].abs(), d[2] / d[1].abs()],
        _ => [d[0] / d[2].abs(), d[1] / d[2].abs()],
    };
    [0.5 * (uv[0] + 1.0), 0.5 * (uv[1] + 1.0)]
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let length = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if length <= 1e-20 {
        return [0.0, 1.0, 0.0];
    }
    [v[0] / length, v[1] / length, v[2] / length]
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

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform_source(value: f32) -> (Vec<[f32; 3]>, EquirectSampling) {
        (
            vec![[value, value * 0.5, value * 0.25]; 64 * 32],
            EquirectSampling {
                flip_y: false,
                linear: true,
                wrap_s: EnvWrap::Clamp,
                wrap_t: EnvWrap::Clamp,
            },
        )
    }

    #[test]
    fn lods_follow_three_create_planes() {
        let lods = create_lods(6);
        let sizes: Vec<u32> = lods.iter().map(|lod| lod.size).collect();
        assert_eq!(sizes, vec![64, 32, 16, 16, 16, 16, 16, 16, 16]);
        assert_eq!(lods[0].sigma, 0.0);
        assert!((lods[1].sigma - 1.0 / 32.0).abs() < 1e-7);
        assert!((lods[2].sigma - 1.0 / 16.0).abs() < 1e-7);
        assert_eq!(lods[3].sigma, 0.125);
        assert_eq!(lods[8].sigma, 0.582);
    }

    #[test]
    fn uniform_environment_stays_uniform_at_every_level() {
        let (pixels, sampling) = uniform_source(2.0);
        let source = Equirect {
            pixels: &pixels,
            width: 64,
            height: 32,
            sampling,
        };
        let atlas = equirect_to_cube_uv(&source).expect("a 64 px input builds an atlas");
        assert_eq!((atlas.width, atlas.height, atlas.lod_max), (336, 64, 4));
        for mip in [4.0f32, 3.0, 1.0, -2.0] {
            for direction in [[1.0, 0.2, 0.1], [0.0, -1.0, 0.0], [-0.3, 0.4, -0.8]] {
                let color = atlas.bilinear_cube_uv(direction, mip);
                assert!(
                    (color[0] - 2.0).abs() < 1e-3,
                    "mip {mip} {direction:?}: {color:?}"
                );
                assert!(
                    (color[2] - 0.5).abs() < 1e-3,
                    "mip {mip} {direction:?}: {color:?}"
                );
            }
        }
    }

    #[test]
    fn inputs_narrower_than_64_px_have_no_atlas_like_three() {
        let (pixels, sampling) = uniform_source(2.0);
        let source = Equirect {
            pixels: &pixels[..32 * 16],
            width: 32,
            height: 16,
            sampling,
        };
        assert!(equirect_to_cube_uv(&source).is_none());
    }

    #[test]
    fn level_zero_matches_equirect_directions_and_flip_y() {
        let mut pixels = vec![[0.0f32; 3]; 64 * 32];
        // GL row 0 (v = 0) is the -Y pole when flipY is false.
        for x in 0..64 {
            pixels[x] = [1.0, 0.0, 0.0];
        }
        let sampling = EquirectSampling {
            flip_y: false,
            linear: false,
            wrap_s: EnvWrap::Clamp,
            wrap_t: EnvWrap::Clamp,
        };
        let source = Equirect {
            pixels: &pixels,
            width: 64,
            height: 32,
            sampling,
        };
        assert_eq!(source.sample_direction([0.0, -1.0, 0.0])[0], 1.0);
        assert_eq!(source.sample_direction([0.0, 1.0, 0.0])[0], 0.0);
        let flipped = Equirect {
            sampling: EquirectSampling {
                flip_y: true,
                ..sampling
            },
            ..source
        };
        assert_eq!(flipped.sample_direction([0.0, 1.0, 0.0])[0], 1.0);
    }
}
