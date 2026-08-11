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
    /// Effective shadow filter radius multiplier.
    pub radius: f32,
    /// Shadow darkness multiplier. Defaults to 1; 0 disables received shadow darkening.
    pub intensity: f32,
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
    pub transmission_resolution_scale: f32,
    pub sample_count: u32,
    pub shadow_map_type: f32,
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

mod fog;
mod output;
mod prepare;
mod shadows;

pub(super) use output::*;
use shadows::*;

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

    #[test]
    fn rejects_non_finite_background_texture_anisotropy() {
        let scene = RenderScene {
            width: Some(16),
            height: Some(16),
            background_texture: Some(vec![64u8, 128, 255, 255].into()),
            background_texture_width: Some(1),
            background_texture_height: Some(1),
            background_texture_anisotropy: Some(f64::NAN),
            ..RenderScene::default()
        };
        let camera = Camera::default();
        let limits = wgpu::Limits {
            max_texture_dimension_2d: 8192,
            ..wgpu::Limits::default()
        };

        let error = match RenderSettings::from_scene(&scene, &camera, limits) {
            Ok(_) => panic!("background texture anisotropy should fail"),
            Err(error) => error.to_string(),
        };

        assert!(
            error.contains("scene.backgroundTextureAnisotropy"),
            "error should name background texture anisotropy, got: {error}",
        );
        assert!(
            error.contains("must contain finite f32-compatible numbers"),
            "error should reject non-finite anisotropy, got: {error}",
        );
    }

    #[test]
    fn skips_shadow_maps_when_scene_has_shadow_lights_but_no_shadow_casters() {
        let scene = RenderScene {
            width: Some(16),
            height: Some(16),
            lights: Some(vec![crate::types::SceneLight {
                light_type: "directional".into(),
                cast_shadow: Some(true),
                ..crate::types::SceneLight::default()
            }]),
            meshes: Some(vec![crate::types::SceneMesh {
                positions: vec![-1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 0.0, 1.0, 0.0],
                receive_shadow: Some(true),
                ..crate::types::SceneMesh::default()
            }]),
            ..RenderScene::default()
        };
        let camera = Camera::default();
        let limits = wgpu::Limits {
            max_texture_dimension_2d: 8192,
            ..wgpu::Limits::default()
        };

        let settings = RenderSettings::from_scene(&scene, &camera, limits).unwrap();

        assert!(settings.shadow.is_none());
    }
}
