use anyhow::{Result, bail};
use glam::{Mat4, Vec3};

use crate::ibl::{EnvMap, IblMaps, compute_ibl};
use crate::lights::{GpuLight, MAX_LIGHTS, prepare_lights};
use crate::types::{Camera, RenderScene};
use crate::util::{finite_positive, parse_color, parse_mat4, parse_vec3, validate_dimension};
use crate::{DEFAULT_HEIGHT, DEFAULT_WIDTH};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ShadowKind {
    DirectionalOrSpot,
    Point,
    Cascaded,
}

/// Shadow caster resolved from the first directional, spot, or point light
/// with `castShadow = true`. We only support a single shadow map.
pub struct ShadowCaster {
    /// Light-space matrices (proj * view) in WebGPU clip space. Directional
    /// and spot shadows use layer 0; point shadows use six cube-face layers.
    pub light_vps: [Mat4; 6],
    /// Normalized light direction (from light toward scene).
    pub light_dir: Vec3,
    /// Shadow projection kind.
    pub kind: ShadowKind,
    /// Index of the shadow-casting light in `RenderSettings::lights`.
    pub light_index: u32,
    /// Number of array layers in the shadow depth texture.
    pub layer_count: u32,
    /// Camera-distance split points for cascaded directional shadows.
    pub cascade_splits: [f32; 4],
    /// Shadow map resolution (square, pixels).
    pub map_size: u32,
    /// Depth bias applied when comparing against the shadow map.
    pub bias: f32,
    /// World-space normal offset applied at the receiver.
    pub normal_bias: f32,
}

pub struct RenderSettings {
    pub width: u32,
    pub height: u32,
    pub background: [f64; 4],
    pub output_format: OutputFormat,
    pub view_projection: Mat4,
    pub camera_pos: Vec3,
    pub lights: Vec<GpuLight>,
    pub ambient_color: [f32; 3],
    pub ambient_intensity: f32,
    pub ibl: Option<IblMaps>,
    pub env_intensity: f32,
    pub shadow: Option<ShadowCaster>,
    pub post_processing: PostProcessingSettings,
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

        let background = parse_color(
            scene.background.as_deref(),
            [0.04, 0.045, 0.05, 1.0],
            "scene.background",
        )?;

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

        let view_projection = match camera.view_projection.as_deref() {
            Some(matrix) => parse_mat4(matrix, "camera.viewProjection")?,
            None => {
                let aspect = width as f32 / height as f32;
                let projection =
                    Mat4::perspective_rh(fov_y_degrees.to_radians() as f32, aspect, near, far);
                let view = Mat4::look_at_rh(eye, target, up.normalize());
                projection * view
            }
        };

        let camera_pos = parse_vec3(
            camera.camera_position.as_deref(),
            eye.to_array(),
            "camera.cameraPosition",
        )?;

        let output_format = OutputFormat::from_scene(scene)?;
        let lights = prepare_lights(scene)?;
        let ambient_color = parse_color(
            scene.ambient_light.as_deref(),
            [1.0, 1.0, 1.0, 1.0],
            "scene.ambientLight",
        )?;
        let ambient_intensity = scene.ambient_intensity.unwrap_or(0.0) as f32;

        let ibl = match &scene.environment_map {
            Some(data) if !data.is_empty() => {
                let env_map = EnvMap::from_bytes(
                    data,
                    scene.environment_map_width,
                    scene.environment_map_height,
                )?;
                Some(compute_ibl(&env_map))
            }
            _ => None,
        };
        let env_intensity = scene.environment_map_intensity.unwrap_or(1.0) as f32;

        let shadow = resolve_shadow_caster(scene)?;
        let post_processing = PostProcessingSettings::from_scene(scene);

        Ok(Self {
            width,
            height,
            background,
            output_format,
            view_projection,
            camera_pos,
            lights,
            ambient_color: [
                ambient_color[0] as f32,
                ambient_color[1] as f32,
                ambient_color[2] as f32,
            ],
            ambient_intensity,
            ibl,
            env_intensity,
            shadow,
            post_processing,
        })
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

/// Resolve the (optional) shadow caster from the scene.
/// We pick the first directional, spot, or point light with `castShadow = true`.
fn resolve_shadow_caster(scene: &RenderScene) -> Result<Option<ShadowCaster>> {
    let Some(lights) = scene.lights.as_deref() else {
        return Ok(None);
    };
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
            let proj = Mat4::perspective_rh(
                (angle * 2.0).min(std::f32::consts::PI - 0.001),
                1.0,
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
                    return Ok(Some(ShadowCaster {
                        light_vps,
                        light_dir: dir,
                        kind: ShadowKind::Cascaded,
                        light_index: i as u32,
                        layer_count,
                        cascade_splits,
                        map_size: light.shadow_map_size.unwrap_or(512).clamp(32, 4096),
                        bias: light.shadow_bias.unwrap_or(0.0) as f32,
                        normal_bias: light.shadow_normal_bias.unwrap_or(0.0) as f32,
                    }));
                }
            }
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
        };

        let map_size = light.shadow_map_size.unwrap_or(512).clamp(32, 4096);
        let bias = light.shadow_bias.unwrap_or(0.0) as f32;
        let normal_bias = light.shadow_normal_bias.unwrap_or(0.0) as f32;

        return Ok(Some(ShadowCaster {
            light_vps,
            light_dir: dir,
            kind,
            light_index: i as u32,
            layer_count,
            cascade_splits,
            map_size,
            bias,
            normal_bias,
        }));
    }
    Ok(None)
}
