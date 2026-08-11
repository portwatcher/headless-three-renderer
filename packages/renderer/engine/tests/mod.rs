use super::lights::prepare_lights;
use super::mesh::{decode_texture, prepare_meshes};
use super::renderer::GpuRenderer;
use super::settings::RenderSettings;
use super::types::{Camera, RenderScene, SceneLight, SceneMesh};

fn scene_with_light(light: SceneLight) -> RenderScene {
    RenderScene {
        lights: Some(vec![light]),
        ..RenderScene::default()
    }
}

fn directional_shadow_light() -> SceneLight {
    SceneLight {
        light_type: "directional".into(),
        cast_shadow: Some(true),
        ..SceneLight::default()
    }
}

fn render_limits() -> wgpu::Limits {
    wgpu::Limits {
        max_texture_dimension_2d: 8192,
        ..wgpu::Limits::default()
    }
}

mod preparation;
mod rendering;
