use anyhow::{Result, bail};
use bytemuck::{Pod, Zeroable};

use crate::types::RenderScene;
use crate::util::{parse_color, parse_vec3};

pub const MAX_LIGHTS: usize = 16;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct GpuLight {
    /// 0 = directional, 1 = point, 2 = spot, 3 = hemisphere
    pub light_type: u32,
    pub _pad0: [u32; 3],
    pub color_intensity: [f32; 4],
    /// xyz = position (point/spot) or ground_color (hemisphere), w = distance
    pub position: [f32; 4],
    /// xyz = direction, w = decay
    pub direction: [f32; 4],
    /// spot: [cos_outer_angle, cos_inner_angle, 0, 0]
    pub params: [f32; 4],
}

pub fn prepare_lights(scene: &RenderScene) -> Result<Vec<GpuLight>> {
    let Some(lights) = scene.lights.as_deref() else {
        return Ok(Vec::new());
    };

    let mut gpu_lights = Vec::with_capacity(lights.len().min(MAX_LIGHTS));
    for (i, light) in lights.iter().take(MAX_LIGHTS).enumerate() {
        let light_type = match light.light_type.to_ascii_lowercase().as_str() {
            "directional" => 0u32,
            "point" => 1,
            "spot" => 2,
            "hemisphere" => 3,
            other => bail!("scene.lights[{i}].lightType `{other}` is not supported"),
        };
        let color = parse_color(
            light.color.as_deref(),
            [1.0, 1.0, 1.0, 1.0],
            &format!("scene.lights[{i}].color"),
        )?;
        let intensity = light.intensity.unwrap_or(1.0) as f32;

        let distance = light.distance.unwrap_or(0.0).max(0.0) as f32;
        let decay = light.decay.unwrap_or(2.0) as f32;

        let position = if light_type == 3 {
            // Hemisphere: pack ground color into position.xyz
            let ground = parse_color(
                light.ground_color.as_deref(),
                [0.04, 0.02, 0.0, 1.0],
                &format!("scene.lights[{i}].groundColor"),
            )?;
            [ground[0] as f32, ground[1] as f32, ground[2] as f32, 0.0]
        } else {
            let pos = parse_vec3(
                light.position.as_deref(),
                [0.0, 0.0, 0.0],
                &format!("scene.lights[{i}].position"),
            )?;
            [pos.x, pos.y, pos.z, distance]
        };

        let direction = parse_vec3(
            light.direction.as_deref(),
            if light_type == 3 { [0.0, 1.0, 0.0] } else { [0.0, -1.0, 0.0] },
            &format!("scene.lights[{i}].direction"),
        )?;

        // Spot light cone parameters
        let params = if light_type == 2 {
            let angle = light.angle.unwrap_or(std::f64::consts::FRAC_PI_3).clamp(0.0, std::f64::consts::FRAC_PI_2) as f32;
            let penumbra = light.penumbra.unwrap_or(0.0).clamp(0.0, 1.0) as f32;
            let cos_outer = angle.cos();
            let cos_inner = (angle * (1.0 - penumbra)).cos();
            [cos_outer, cos_inner, 0.0, 0.0]
        } else {
            [0.0; 4]
        };

        gpu_lights.push(GpuLight {
            light_type,
            _pad0: [0; 3],
            color_intensity: [color[0] as f32, color[1] as f32, color[2] as f32, intensity],
            position,
            direction: [direction.x, direction.y, direction.z, decay],
            params,
        });
    }
    Ok(gpu_lights)
}
