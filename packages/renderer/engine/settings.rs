use std::thread;

use anyhow::{Result, bail};
use glam::{Mat4, Vec3};

use crate::ibl::{EnvMap, IblMaps, compute_ibl};
use crate::lights::{GpuLight, MAX_LIGHTS, prepare_lights};
use crate::mesh::{
    MipmapFilter, PreparedTexture, TextureFilter, WrapMode, decode_texture_with_label,
    texture_anisotropy,
};
use crate::types::{Camera, RenderScene};
use crate::util::{
    finite_f32, finite_positive, parse_color, parse_mat4, parse_vec3, validate_dimension,
};
use crate::{DEFAULT_HEIGHT, DEFAULT_WIDTH};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ShadowKind {
    DirectionalOrSpot,
    Point,
    Cascaded,
}

pub const MAX_SHADOW_LAYERS: usize = 12;

/// Shadow caster resolved from a directional, spot, or point light with
/// `castShadow = true`.
pub struct ShadowCaster {
    /// Light-space matrices (proj * view) in WebGPU clip space. Directional
    /// and spot shadows use layer 0; point shadows use six cube-face layers.
    pub light_vps: [Mat4; 6],
    /// Shadow projection kind.
    pub kind: ShadowKind,
    /// Index of the shadow-casting light in `RenderSettings::lights`.
    pub light_index: u32,
    /// First layer assigned to this caster in the shared depth texture array.
    pub layer_base: u32,
    /// Number of array layers in the shadow depth texture.
    pub layer_count: u32,
    /// Camera-distance split points for cascaded directional shadows.
    pub cascade_splits: [f32; 4],
    /// Depth bias applied when comparing against the shadow map.
    pub bias: f32,
    /// World-space normal offset applied at the receiver.
    pub normal_bias: f32,
    /// PCF radius multiplier.
    pub radius: f32,
}

pub struct ShadowMapSet {
    pub casters: Vec<ShadowCaster>,
    pub layer_count: u32,
    pub map_width: u32,
    pub map_height: u32,
}

pub struct RenderSettings {
    pub width: u32,
    pub height: u32,
    pub background: [f64; 4],
    pub background_intensity: f32,
    pub viewport: Option<ViewportRect>,
    pub scissor: Option<ScissorRect>,
    pub background_texture: Option<BackgroundTexture>,
    pub output_format: OutputFormat,
    pub output_color_space: OutputColorSpace,
    pub tone_mapping: f32,
    pub tone_mapping_exposure: f32,
    pub sample_count: u32,
    pub view: Mat4,
    pub view_projection: Mat4,
    pub camera_pos: Vec3,
    pub near: f32,
    pub far: f32,
    pub lights: Vec<GpuLight>,
    pub ambient_color: [f32; 3],
    pub ambient_intensity: f32,
    pub light_probe: [[f32; 3]; 9],
    pub has_light_probe: bool,
    pub ibl: Option<IblMaps>,
    pub env_intensity: f32,
    pub fog: FogSettings,
    pub shadow: Option<ShadowMapSet>,
    pub post_processing: PostProcessingSettings,
}

#[derive(Copy, Clone, Debug)]
pub struct ViewportRect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

#[derive(Copy, Clone, Debug)]
pub struct ScissorRect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

pub struct BackgroundTexture {
    pub texture: PreparedTexture,
    pub transform: [f32; 6],
    pub is_srgb: bool,
    pub mapping: BackgroundTextureMapping,
    pub rotation: [[f32; 4]; 3],
    pub intensity: f32,
    pub blurriness: f32,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BackgroundTextureMapping {
    Uv,
    Equirectangular,
}

impl BackgroundTextureMapping {
    fn from_scene(value: Option<&str>) -> Result<Self> {
        match value.unwrap_or("uv").to_ascii_lowercase().as_str() {
            "uv" | "2d" => Ok(Self::Uv),
            "equirectangular" | "equirect" => Ok(Self::Equirectangular),
            other => bail!(
                "unsupported scene.backgroundTextureMapping `{other}`; expected `uv` or `equirectangular`"
            ),
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct FogSettings {
    pub active: bool,
    pub mode: f32,
    pub color: [f32; 3],
    pub near: f32,
    pub far: f32,
    pub density: f32,
}

#[derive(Copy, Clone, Debug)]
pub struct PostProcessingSettings {
    pub active: bool,
    pub exposure: f32,
    pub contrast: f32,
    pub saturation: f32,
    pub vignette: f32,
    pub grayscale: f32,
    pub invert: f32,
}

impl RenderSettings {
    pub fn from_scene(scene: &RenderScene, camera: &Camera, limits: wgpu::Limits) -> Result<Self> {
        let width = camera.width.or(scene.width).unwrap_or(DEFAULT_WIDTH);
        let height = camera.height.or(scene.height).unwrap_or(DEFAULT_HEIGHT);

        validate_dimension("width", width, limits.max_texture_dimension_2d)?;
        validate_dimension("height", height, limits.max_texture_dimension_2d)?;

        let viewport = parse_viewport_rect(scene.viewport.as_deref(), width, height)?;
        let scissor = parse_scissor_rect(scene.scissor.as_deref(), width, height)?;

        let background = parse_color(
            scene.background.as_deref(),
            [0.04, 0.045, 0.05, 1.0],
            "scene.background",
        )?;
        let background_intensity = finite_f32(
            scene.background_intensity.unwrap_or(1.0),
            "scene.backgroundIntensity",
        )?
        .max(0.0);
        thread::scope(|scope| {
            let background_handle =
                scope.spawn(|| prepare_background_texture(scene, background_intensity));
            let ibl_handle = scope.spawn(|| prepare_scene_ibl(scene));

            let background_texture =
                join_prepare_worker(background_handle, "background texture worker")?;

            let eye = parse_vec3(camera.eye.as_deref(), [2.5, 1.8, 3.2], "camera.eye")?;
            let target = parse_vec3(camera.target.as_deref(), [0.0, 0.0, 0.0], "camera.target")?;
            let up = parse_vec3(camera.up.as_deref(), [0.0, 1.0, 0.0], "camera.up")?;

            if eye.distance_squared(target) <= f32::EPSILON {
                bail!("camera.eye must not equal camera.target");
            }
            if up.length_squared() <= f32::EPSILON {
                bail!("camera.up must not be a zero vector");
            }

            let fov_y_degrees = camera.fov_y_degrees.unwrap_or(45.0);
            if !fov_y_degrees.is_finite() || !(1.0..179.0).contains(&fov_y_degrees) {
                bail!("camera.fov_y_degrees must be finite and between 1 and 179");
            }

            let near = finite_positive(camera.near.unwrap_or(0.01), "camera.near")?;
            let far = finite_positive(camera.far.unwrap_or(100.0), "camera.far")?;
            if far <= near {
                bail!("camera.far must be greater than camera.near");
            }

            let view = match camera.view_matrix.as_deref() {
                Some(matrix) => parse_mat4(matrix, "camera.viewMatrix")?,
                None => Mat4::look_at_rh(eye, target, up.normalize()),
            };

            let view_projection = match camera.view_projection.as_deref() {
                Some(matrix) => parse_mat4(matrix, "camera.viewProjection")?,
                None => {
                    let aspect = width as f32 / height as f32;
                    let projection =
                        Mat4::perspective_rh(fov_y_degrees.to_radians() as f32, aspect, near, far);
                    projection * view
                }
            };

            let camera_pos = parse_vec3(
                camera.camera_position.as_deref(),
                eye.to_array(),
                "camera.cameraPosition",
            )?;

            let output_format = OutputFormat::from_scene(scene)?;
            let output_color_space = OutputColorSpace::from_scene(scene)?;
            let tone_mapping = resolve_tone_mapping(scene.tone_mapping)?;
            let tone_mapping_exposure = finite_f32(
                scene.tone_mapping_exposure.unwrap_or(1.0),
                "scene.toneMappingExposure",
            )?;
            if tone_mapping_exposure < 0.0 {
                bail!("scene.toneMappingExposure must be non-negative");
            }
            let sample_count = resolve_sample_count(scene.sample_count)?;
            let lights = prepare_lights(scene)?;
            let ambient_color = parse_color(
                scene.ambient_light.as_deref(),
                [1.0, 1.0, 1.0, 1.0],
                "scene.ambientLight",
            )?;
            let ambient_intensity = scene.ambient_intensity.unwrap_or(0.0) as f32;
            let (light_probe, has_light_probe) = parse_light_probe(scene.light_probe.as_deref())?;
            let ibl = join_prepare_worker(ibl_handle, "environment map worker")?;
            let env_intensity = scene.environment_map_intensity.unwrap_or(1.0) as f32;

            let fog = FogSettings::from_scene(scene, background)?;
            let shadow = resolve_shadow_maps(scene)?;
            let post_processing = PostProcessingSettings::from_scene(scene);

            Ok(Self {
                width,
                height,
                background,
                background_intensity,
                viewport,
                scissor,
                background_texture,
                output_format,
                output_color_space,
                tone_mapping,
                tone_mapping_exposure,
                sample_count,
                view,
                view_projection,
                camera_pos,
                near,
                far,
                lights,
                ambient_color: [
                    ambient_color[0] as f32,
                    ambient_color[1] as f32,
                    ambient_color[2] as f32,
                ],
                ambient_intensity,
                light_probe,
                has_light_probe,
                ibl,
                env_intensity,
                fog,
                shadow,
                post_processing,
            })
        })
    }
}

fn join_prepare_worker<T>(
    handle: thread::ScopedJoinHandle<'_, Result<T>>,
    label: &str,
) -> Result<T> {
    handle.join().unwrap_or_else(|_| panic!("{label} panicked"))
}

fn prepare_background_texture(
    scene: &RenderScene,
    background_intensity: f32,
) -> Result<Option<BackgroundTexture>> {
    match &scene.background_texture {
        Some(data) if !data.is_empty() => {
            let mut texture = decode_texture_with_label(
                data,
                scene.background_texture_width,
                scene.background_texture_height,
                "scene.backgroundTexture",
            )?;
            texture.wrap_s = WrapMode::from_str_opt(scene.background_texture_wrap_s.as_deref());
            texture.wrap_t = WrapMode::from_str_opt(scene.background_texture_wrap_t.as_deref());
            texture.mag_filter =
                TextureFilter::from_str_opt(scene.background_texture_mag_filter.as_deref());
            texture.min_filter =
                TextureFilter::from_min_filter_str(scene.background_texture_min_filter.as_deref());
            texture.mipmap_filter =
                MipmapFilter::from_min_filter_str(scene.background_texture_min_filter.as_deref());
            texture.anisotropy = texture_anisotropy(scene.background_texture_anisotropy);
            Ok(Some(BackgroundTexture {
                texture,
                transform: parse_texture_transform(
                    scene.background_texture_transform.as_deref(),
                    "scene.backgroundTextureTransform",
                )?,
                is_srgb: matches!(
                    scene.background_texture_color_space.as_deref(),
                    Some("srgb")
                ),
                mapping: BackgroundTextureMapping::from_scene(
                    scene.background_texture_mapping.as_deref(),
                )?,
                rotation: parse_rotation_columns(
                    scene.background_texture_rotation.as_deref(),
                    "scene.backgroundTextureRotation",
                )?,
                intensity: background_intensity,
                blurriness: finite_f32(
                    scene.background_texture_blurriness.unwrap_or(0.0),
                    "scene.backgroundTextureBlurriness",
                )?
                .clamp(0.0, 1.0),
            }))
        }
        _ => Ok(None),
    }
}

fn prepare_scene_ibl(scene: &RenderScene) -> Result<Option<IblMaps>> {
    match &scene.environment_map {
        Some(data) if !data.is_empty() => {
            let rotation = parse_rotation_columns(
                scene.environment_map_rotation.as_deref(),
                "scene.environmentMapRotation",
            )?;
            let env_map = EnvMap::from_bytes(
                data,
                scene.environment_map_width,
                scene.environment_map_height,
                parse_environment_color_space(scene.environment_map_color_space.as_deref())?,
            )?;
            Ok(Some(compute_ibl(&env_map, rotation)))
        }
        _ => Ok(None),
    }
}

fn parse_viewport_rect(
    values: Option<&[f64]>,
    target_width: u32,
    target_height: u32,
) -> Result<Option<ViewportRect>> {
    let Some(values) = values else {
        return Ok(None);
    };
    if values.len() != 4 {
        bail!("scene.viewport must be `[x, y, width, height]`");
    }

    let x = finite_f32(values[0], "scene.viewport")?;
    let y = finite_f32(values[1], "scene.viewport")?;
    let width = finite_f32(values[2], "scene.viewport")?;
    let height = finite_f32(values[3], "scene.viewport")?;
    if x < 0.0 || y < 0.0 {
        bail!("scene.viewport x and y must be greater than or equal to 0");
    }
    if width <= 0.0 || height <= 0.0 {
        bail!("scene.viewport width and height must be greater than 0");
    }
    if x + width > target_width as f32 || y + height > target_height as f32 {
        bail!("scene.viewport must fit inside the render target");
    }

    Ok(Some(ViewportRect {
        x,
        y,
        width,
        height,
    }))
}

fn parse_scissor_rect(
    values: Option<&[f64]>,
    target_width: u32,
    target_height: u32,
) -> Result<Option<ScissorRect>> {
    let Some(values) = values else {
        return Ok(None);
    };
    if values.len() != 4 {
        bail!("scene.scissor must be `[x, y, width, height]`");
    }

    let x = finite_u32_pixel(values[0], "scene.scissor")?;
    let y = finite_u32_pixel(values[1], "scene.scissor")?;
    let width = finite_u32_pixel(values[2], "scene.scissor")?;
    let height = finite_u32_pixel(values[3], "scene.scissor")?;
    if width == 0 || height == 0 {
        bail!("scene.scissor width and height must be greater than 0");
    }
    if x.checked_add(width)
        .is_none_or(|right| right > target_width)
        || y.checked_add(height)
            .is_none_or(|bottom| bottom > target_height)
    {
        bail!("scene.scissor must fit inside the render target");
    }

    Ok(Some(ScissorRect {
        x,
        y,
        width,
        height,
    }))
}

fn finite_u32_pixel(value: f64, field: &str) -> Result<u32> {
    if !value.is_finite() || value < 0.0 || value > u32::MAX as f64 || value.fract() != 0.0 {
        bail!("{field} must contain non-negative whole pixel numbers");
    }
    Ok(value as u32)
}

impl FogSettings {
    fn from_scene(scene: &RenderScene, background: [f64; 4]) -> Result<Self> {
        let color = parse_color(
            scene.fog_color.as_deref(),
            [background[0], background[1], background[2], 1.0],
            "scene.fogColor",
        )?;
        let base = Self {
            active: false,
            mode: 0.0,
            color: [color[0] as f32, color[1] as f32, color[2] as f32],
            near: 1.0,
            far: 1000.0,
            density: 0.00025,
        };

        let Some(fog_type) = scene.fog_type.as_deref() else {
            return Ok(base);
        };

        match fog_type.to_ascii_lowercase().as_str() {
            "linear" | "fog" => {
                let near = finite_f32(scene.fog_near.unwrap_or(1.0), "scene.fogNear")?;
                let far = finite_f32(scene.fog_far.unwrap_or(1000.0), "scene.fogFar")?;
                if far <= near {
                    bail!("scene.fogFar must be greater than scene.fogNear");
                }
                Ok(Self {
                    active: true,
                    mode: 1.0,
                    near,
                    far,
                    ..base
                })
            }
            "exp2" | "fogexp2" => {
                let density = finite_f32(scene.fog_density.unwrap_or(0.00025), "scene.fogDensity")?;
                if density < 0.0 {
                    bail!("scene.fogDensity must be greater than or equal to 0");
                }
                Ok(Self {
                    active: density > 0.0,
                    mode: 2.0,
                    density,
                    ..base
                })
            }
            other => bail!("unsupported scene.fogType `{other}`; expected `linear` or `exp2`"),
        }
    }
}

impl PostProcessingSettings {
    fn from_scene(scene: &RenderScene) -> Self {
        let exposure = scene.post_exposure.unwrap_or(0.0).clamp(-16.0, 16.0) as f32;
        let contrast = scene.post_contrast.unwrap_or(1.0).clamp(0.0, 8.0) as f32;
        let saturation = scene.post_saturation.unwrap_or(1.0).clamp(0.0, 8.0) as f32;
        let vignette = scene.post_vignette.unwrap_or(0.0).clamp(0.0, 1.0) as f32;
        let grayscale = scene.post_grayscale.unwrap_or(0.0).clamp(0.0, 1.0) as f32;
        let invert = scene.post_invert.unwrap_or(0.0).clamp(0.0, 1.0) as f32;
        let active = exposure.abs() > 0.0001
            || (contrast - 1.0).abs() > 0.0001
            || (saturation - 1.0).abs() > 0.0001
            || vignette > 0.0001
            || grayscale > 0.0001
            || invert > 0.0001;

        Self {
            active,
            exposure,
            contrast,
            saturation,
            vignette,
            grayscale,
            invert,
        }
    }
}

#[derive(Copy, Clone)]
pub enum OutputFormat {
    Png,
    Rgba,
}

impl OutputFormat {
    fn from_scene(scene: &RenderScene) -> Result<Self> {
        let Some(format) = scene.format.as_deref() else {
            return Ok(Self::Png);
        };

        match format.to_ascii_lowercase().as_str() {
            "png" => Ok(Self::Png),
            "rgba" | "raw" | "raw-rgba" => Ok(Self::Rgba),
            other => bail!("unsupported scene.format `{other}`; expected `png` or `rgba`"),
        }
    }
}

fn resolve_sample_count(value: Option<u32>) -> Result<u32> {
    match value.unwrap_or(1) {
        0 | 1 => Ok(1),
        4 => Ok(4),
        other => bail!(
            "unsupported scene.sampleCount `{other}`; expected 1 for single-sample rendering or 4 for 4x MSAA"
        ),
    }
}

fn resolve_tone_mapping(value: Option<u32>) -> Result<f32> {
    match value.unwrap_or(4) {
        0 => Ok(0.0), // THREE.NoToneMapping
        1 => Ok(1.0), // THREE.LinearToneMapping
        2 => Ok(2.0), // THREE.ReinhardToneMapping
        3 => Ok(3.0), // THREE.CineonToneMapping
        4 => Ok(4.0), // THREE.ACESFilmicToneMapping
        other => bail!(
            "unsupported scene.toneMapping `{other}`; expected NoToneMapping, LinearToneMapping, ReinhardToneMapping, CineonToneMapping, or ACESFilmicToneMapping"
        ),
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum OutputColorSpace {
    Srgb,
    LinearSrgb,
}

impl OutputColorSpace {
    fn from_scene(scene: &RenderScene) -> Result<Self> {
        let Some(color_space) = scene.output_color_space.as_deref() else {
            return Ok(Self::Srgb);
        };

        match color_space.to_ascii_lowercase().as_str() {
            "srgb" => Ok(Self::Srgb),
            "srgb-linear" | "linear-srgb" | "linearsrgb" | "linear" => Ok(Self::LinearSrgb),
            other => bail!(
                "unsupported scene.outputColorSpace `{other}`; expected `srgb`, `srgb-linear`, `linear-srgb`, `linearsrgb`, or `linear`"
            ),
        }
    }

    pub fn is_linear(self) -> bool {
        matches!(self, Self::LinearSrgb)
    }
}

fn parse_environment_color_space(value: Option<&str>) -> Result<bool> {
    let Some(color_space) = value else {
        return Ok(true);
    };

    match color_space.to_ascii_lowercase().as_str() {
        "srgb" => Ok(true),
        "srgb-linear" | "linear-srgb" | "linearsrgb" | "linear" => Ok(false),
        other => bail!(
            "unsupported scene.environmentMapColorSpace `{other}`; expected `srgb`, `srgb-linear`, `linear-srgb`, `linearsrgb`, or `linear`"
        ),
    }
}

fn parse_texture_transform(values: Option<&[f64]>, label: &str) -> Result<[f32; 6]> {
    let Some(values) = values else {
        return Ok([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    };
    if values.len() != 6 {
        bail!("{label} must be an array of 6 numbers");
    }
    Ok([
        finite_f32(values[0], label)?,
        finite_f32(values[1], label)?,
        finite_f32(values[2], label)?,
        finite_f32(values[3], label)?,
        finite_f32(values[4], label)?,
        finite_f32(values[5], label)?,
    ])
}

fn parse_rotation_columns(values: Option<&[f64]>, label: &str) -> Result<[[f32; 4]; 3]> {
    let Some(values) = values else {
        return Ok([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]);
    };
    if values.len() != 9 {
        bail!("{label} must be an array of 9 numbers");
    }
    Ok([
        [
            finite_f32(values[0], label)?,
            finite_f32(values[1], label)?,
            finite_f32(values[2], label)?,
            0.0,
        ],
        [
            finite_f32(values[3], label)?,
            finite_f32(values[4], label)?,
            finite_f32(values[5], label)?,
            0.0,
        ],
        [
            finite_f32(values[6], label)?,
            finite_f32(values[7], label)?,
            finite_f32(values[8], label)?,
            0.0,
        ],
    ])
}

fn parse_light_probe(values: Option<&[f64]>) -> Result<([[f32; 3]; 9], bool)> {
    let Some(values) = values else {
        return Ok(([[0.0; 3]; 9], false));
    };
    if values.len() != 27 {
        bail!("scene.lightProbe must be an array of 27 numbers");
    }
    let mut coefficients = [[0.0; 3]; 9];
    for i in 0..9 {
        coefficients[i] = [
            finite_f32(values[i * 3], "scene.lightProbe")?,
            finite_f32(values[i * 3 + 1], "scene.lightProbe")?,
            finite_f32(values[i * 3 + 2], "scene.lightProbe")?,
        ];
    }
    Ok((coefficients, true))
}

/// Resolve optional shadow casters from the scene into a shared depth array.
fn resolve_shadow_maps(scene: &RenderScene) -> Result<Option<ShadowMapSet>> {
    let Some(lights) = scene.lights.as_deref() else {
        return Ok(None);
    };
    let mut casters = Vec::new();
    let mut total_layers = 0u32;
    let mut atlas_width = 0u32;
    let mut atlas_height = 0u32;

    for (i, light) in lights.iter().take(MAX_LIGHTS).enumerate() {
        let light_type = light.light_type.to_ascii_lowercase();
        if light_type != "directional" && light_type != "spot" && light_type != "point" {
            continue;
        }
        if !light.cast_shadow.unwrap_or(false) {
            continue;
        }
        let prefix = format!("scene.lights[{i}]");

        let pos = parse_vec3(
            light.position.as_deref(),
            [0.0, 10.0, 0.0],
            &format!("{prefix}.position"),
        )?;
        let dir = parse_vec3(
            light.direction.as_deref(),
            [0.0, -1.0, 0.0],
            &format!("{prefix}.direction"),
        )?;
        let dir = if dir.length_squared() > 0.0 {
            dir.normalize()
        } else {
            Vec3::new(0.0, -1.0, 0.0)
        };

        let near = light.shadow_camera_near.unwrap_or(0.5) as f32;
        let default_far = if light_type == "point" {
            let distance = light.distance.unwrap_or(0.0);
            if distance > 0.0 { distance } else { 500.0 }
        } else {
            500.0
        };
        let far = light.shadow_camera_far.unwrap_or(default_far) as f32;
        if far <= near {
            bail!("{prefix}.shadow.camera has invalid near/far bounds");
        }
        let (map_width, map_height) = shadow_map_dimensions(light, &light_type, &prefix)?;

        let mut light_vps = [Mat4::IDENTITY; 6];
        let mut cascade_splits = [f32::MAX; 4];
        let mut layer_count = 1u32;
        let kind = if light_type == "point" {
            let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, 1.0, near, far);
            let faces = [
                (Vec3::X, -Vec3::Y),
                (-Vec3::X, -Vec3::Y),
                (Vec3::Y, Vec3::Z),
                (-Vec3::Y, -Vec3::Z),
                (Vec3::Z, -Vec3::Y),
                (-Vec3::Z, -Vec3::Y),
            ];
            for (face, (face_dir, up)) in faces.into_iter().enumerate() {
                light_vps[face] = proj * Mat4::look_at_rh(pos, pos + face_dir, up);
            }
            layer_count = 6;
            ShadowKind::Point
        } else if light_type == "spot" {
            // View: look from light position along the light's direction. Pick an
            // up vector that is not collinear with `dir`.
            let up = if dir.y.abs() > 0.99 { Vec3::Z } else { Vec3::Y };
            let view = Mat4::look_at_rh(pos, pos + dir, up);
            let angle = light
                .angle
                .unwrap_or(std::f64::consts::FRAC_PI_3)
                .clamp(0.001, std::f64::consts::FRAC_PI_2) as f32;
            let aspect = map_width as f32 / map_height as f32;
            let proj = Mat4::perspective_rh(
                (angle * 2.0).min(std::f32::consts::PI - 0.001),
                aspect,
                near,
                far,
            );
            light_vps[0] = proj * view;
            ShadowKind::DirectionalOrSpot
        } else {
            // View: look from light position along the light's direction. Pick an
            // up vector that is not collinear with `dir`.
            let up = if dir.y.abs() > 0.99 { Vec3::Z } else { Vec3::Y };
            let view = Mat4::look_at_rh(pos, pos + dir, up);
            if let Some(bounds) = light.shadow_cascade_bounds.as_deref() {
                let cascade_count = (bounds.len() / 6).min(4);
                if cascade_count >= 2 {
                    for cascade in 0..cascade_count {
                        let base = cascade * 6;
                        let left = bounds[base] as f32;
                        let right = bounds[base + 1] as f32;
                        let top = bounds[base + 2] as f32;
                        let bottom = bounds[base + 3] as f32;
                        let cascade_near = bounds[base + 4] as f32;
                        let cascade_far = bounds[base + 5] as f32;
                        if right <= left || top <= bottom || cascade_far <= cascade_near {
                            bail!("{prefix}.shadow.cascades[{cascade}] has invalid bounds");
                        }
                        light_vps[cascade] = Mat4::orthographic_rh(
                            left,
                            right,
                            bottom,
                            top,
                            cascade_near,
                            cascade_far,
                        ) * view;
                    }
                    if let Some(splits) = light.shadow_cascade_splits.as_deref() {
                        for (slot, value) in splits.iter().take(cascade_count - 1).enumerate() {
                            cascade_splits[slot] = (*value as f32).max(0.0);
                        }
                    }
                    layer_count = cascade_count as u32;
                    ShadowKind::Cascaded
                } else {
                    // Orthographic bounds (three.js DirectionalLightShadow defaults: ±5).
                    let left = light.shadow_camera_left.unwrap_or(-5.0) as f32;
                    let right = light.shadow_camera_right.unwrap_or(5.0) as f32;
                    let top = light.shadow_camera_top.unwrap_or(5.0) as f32;
                    let bottom = light.shadow_camera_bottom.unwrap_or(-5.0) as f32;
                    if right <= left || top <= bottom {
                        bail!("{prefix}.shadow.camera has invalid orthographic bounds");
                    }
                    light_vps[0] =
                        Mat4::orthographic_rh(left, right, bottom, top, near, far) * view;
                    ShadowKind::DirectionalOrSpot
                }
            } else {
                // Orthographic bounds (three.js DirectionalLightShadow defaults: ±5).
                let left = light.shadow_camera_left.unwrap_or(-5.0) as f32;
                let right = light.shadow_camera_right.unwrap_or(5.0) as f32;
                let top = light.shadow_camera_top.unwrap_or(5.0) as f32;
                let bottom = light.shadow_camera_bottom.unwrap_or(-5.0) as f32;
                if right <= left || top <= bottom {
                    bail!("{prefix}.shadow.camera has invalid orthographic bounds");
                }
                light_vps[0] = Mat4::orthographic_rh(left, right, bottom, top, near, far) * view;
                ShadowKind::DirectionalOrSpot
            }
        };

        let bias = light.shadow_bias.unwrap_or(0.0) as f32;
        let normal_bias = light.shadow_normal_bias.unwrap_or(0.0) as f32;
        let radius = shadow_radius(light, &prefix)?;

        let requested_layers = total_layers + layer_count;
        if requested_layers > MAX_SHADOW_LAYERS as u32 {
            bail!(
                "More than {MAX_SHADOW_LAYERS} shadow map layers are not supported by @headless-three/renderer yet ({requested_layers} requested). Directional and spot shadows use 1 layer, point shadows use 6 layers, and cascaded directional shadows use one layer per cascade."
            );
        }

        casters.push(ShadowCaster {
            light_vps,
            kind,
            light_index: i as u32,
            layer_base: total_layers,
            layer_count,
            cascade_splits,
            bias,
            normal_bias,
            radius,
        });
        total_layers = requested_layers;
        atlas_width = atlas_width.max(map_width);
        atlas_height = atlas_height.max(map_height);
    }
    if casters.is_empty() {
        Ok(None)
    } else {
        Ok(Some(ShadowMapSet {
            casters,
            layer_count: total_layers,
            map_width: atlas_width,
            map_height: atlas_height,
        }))
    }
}

fn shadow_radius(light: &crate::types::SceneLight, prefix: &str) -> Result<f32> {
    Ok(finite_f32(
        light.shadow_radius.unwrap_or(1.0),
        &format!("{prefix}.shadow.radius"),
    )?
    .max(0.0))
}

fn shadow_map_dimensions(
    light: &crate::types::SceneLight,
    light_type: &str,
    prefix: &str,
) -> Result<(u32, u32)> {
    let width_hint = light.shadow_map_width.or(light.shadow_map_size);
    let height_hint = light.shadow_map_height.or(light.shadow_map_size);
    let width = width_hint.or(height_hint).unwrap_or(512).clamp(32, 4096);
    let height = height_hint.or(width_hint).unwrap_or(512).clamp(32, 4096);
    if light_type == "point" && width != height {
        bail!("{prefix}.shadow.mapSize must be square for point-light cube shadows");
    }
    Ok((width, height))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Camera;

    #[test]
    fn render_settings_prepares_background_texture_on_worker() {
        let scene = RenderScene {
            width: Some(16),
            height: Some(16),
            background_texture: Some(vec![64u8, 128, 255, 255].into()),
            background_texture_width: Some(1),
            background_texture_height: Some(1),
            background_texture_wrap_s: Some("repeat".into()),
            background_texture_mag_filter: Some("nearest".into()),
            background_texture_min_filter: Some("nearest".into()),
            background_texture_blurriness: Some(0.25),
            ..RenderScene::default()
        };
        let camera = Camera::default();
        let limits = wgpu::Limits {
            max_texture_dimension_2d: 8192,
            ..wgpu::Limits::default()
        };

        let settings = RenderSettings::from_scene(&scene, &camera, limits).unwrap();
        let background = settings.background_texture.unwrap();

        assert_eq!(background.texture.width, 1);
        assert_eq!(background.texture.height, 1);
        assert_eq!(background.texture.rgba, vec![64, 128, 255, 255]);
        assert!(matches!(background.texture.wrap_s, WrapMode::Repeat));
        assert!(matches!(
            background.texture.mag_filter,
            TextureFilter::Nearest
        ));
        assert!((background.blurriness - 0.25).abs() < f32::EPSILON);
        assert!(settings.ibl.is_none());
    }
}
