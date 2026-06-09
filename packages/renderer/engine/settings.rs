use anyhow::{Result, bail};
use glam::{Mat4, Vec3};

use crate::ibl::{EnvMap, IblMaps, compute_ibl};
use crate::lights::{GpuLight, MAX_LIGHTS, prepare_lights};
use crate::types::{Camera, RenderScene};
use crate::util::{finite_positive, parse_color, parse_mat4, parse_vec3, validate_dimension};
use crate::{DEFAULT_HEIGHT, DEFAULT_WIDTH};

/// Directional shadow caster resolved from the first directional light with
/// `castShadow = true`. We only support a single directional shadow map.
pub struct ShadowCaster {
    /// Orthographic light-space matrix (proj * view) in WebGPU clip space.
    pub light_vp: Mat4,
    /// Normalized light direction (from light toward scene).
    pub light_dir: Vec3,
    /// Index of the shadow-casting light in `RenderSettings::lights`.
    pub light_index: u32,
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

        Ok(Self {
            width,
            height,
            background,
            output_format,
            view_projection,
            camera_pos,
            lights,
            ambient_color: [ambient_color[0] as f32, ambient_color[1] as f32, ambient_color[2] as f32],
            ambient_intensity,
            ibl,
            env_intensity,
            shadow,
        })
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

/// Resolve the (optional) directional shadow caster from the scene.
/// We pick the first directional light with `castShadow = true`.
fn resolve_shadow_caster(scene: &RenderScene) -> Result<Option<ShadowCaster>> {
    let Some(lights) = scene.lights.as_deref() else {
        return Ok(None);
    };
    for (i, light) in lights.iter().take(MAX_LIGHTS).enumerate() {
        if light.light_type.to_ascii_lowercase() != "directional" {
            continue;
        }
        if !light.cast_shadow.unwrap_or(false) {
            continue;
        }
        let prefix = format!("scene.lights[{i}]");

        let pos = parse_vec3(light.position.as_deref(), [0.0, 10.0, 0.0], &format!("{prefix}.position"))?;
        let dir = parse_vec3(
            light.direction.as_deref(),
            [0.0, -1.0, 0.0],
            &format!("{prefix}.direction"),
        )?;
        let dir = if dir.length_squared() > 0.0 { dir.normalize() } else { Vec3::new(0.0, -1.0, 0.0) };

        // Orthographic bounds (three.js defaults: ±5, near=0.5, far=500)
        let left = light.shadow_camera_left.unwrap_or(-5.0) as f32;
        let right = light.shadow_camera_right.unwrap_or(5.0) as f32;
        let top = light.shadow_camera_top.unwrap_or(5.0) as f32;
        let bottom = light.shadow_camera_bottom.unwrap_or(-5.0) as f32;
        let near = light.shadow_camera_near.unwrap_or(0.5) as f32;
        let far = light.shadow_camera_far.unwrap_or(500.0) as f32;
        if right <= left || top <= bottom || far <= near {
            bail!("{prefix}.shadow.camera has invalid frustum bounds");
        }

        // View: look from light position along the light's direction. Pick an
        // up vector that is not collinear with `dir`.
        let up = if dir.y.abs() > 0.99 { Vec3::Z } else { Vec3::Y };
        let view = Mat4::look_at_rh(pos, pos + dir, up);
        let proj = Mat4::orthographic_rh(left, right, bottom, top, near, far);
        let light_vp = proj * view;

        let map_size = light.shadow_map_size.unwrap_or(512).clamp(32, 4096);
        let bias = light.shadow_bias.unwrap_or(0.0) as f32;
        let normal_bias = light.shadow_normal_bias.unwrap_or(0.0) as f32;

        return Ok(Some(ShadowCaster {
            light_vp,
            light_dir: dir,
            light_index: i as u32,
            map_size,
            bias,
            normal_bias,
        }));
    }
    Ok(None)
}
