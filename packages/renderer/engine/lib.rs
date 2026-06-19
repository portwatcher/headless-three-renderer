mod ibl;
mod lights;
mod mesh;
mod renderer;
mod settings;
mod shader;
mod types;
mod util;

use std::sync::OnceLock;

use napi::bindgen_prelude::Buffer;
use napi_derive::napi;

use renderer::GpuRenderer;
use types::{Camera, RenderScene};
use util::encode_png;

const COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth24PlusStencil8;
const MAX_DIMENSION: u32 = 8192;
const DEFAULT_WIDTH: u32 = 512;
const DEFAULT_HEIGHT: u32 = 512;

static SHARED_RENDERER: OnceLock<std::result::Result<GpuRenderer, String>> = OnceLock::new();

#[napi]
pub struct NativeRenderer {
    inner: GpuRenderer,
}

#[napi]
impl NativeRenderer {
    #[napi(constructor)]
    pub fn new() -> napi::Result<Self> {
        Ok(Self {
            inner: GpuRenderer::new().map_err(to_napi_error)?,
        })
    }

    #[napi]
    pub fn render(&self, scene: RenderScene, camera: Camera) -> napi::Result<Buffer> {
        self.inner
            .render(&scene, &camera)
            .map(Buffer::from)
            .map_err(to_napi_error)
    }
}

#[napi]
pub fn render_native(scene: RenderScene, camera: Camera) -> napi::Result<Buffer> {
    let renderer = SHARED_RENDERER
        .get_or_init(|| GpuRenderer::new().map_err(|error| error.to_string()))
        .as_ref()
        .map_err(|reason| napi::Error::from_reason(reason.clone()))?;

    renderer
        .render(&scene, &camera)
        .map(Buffer::from)
        .map_err(to_napi_error)
}

#[napi(object)]
pub struct DecodedImage {
    pub data: Buffer,
    pub width: u32,
    pub height: u32,
}

#[napi]
pub fn decode_image(data: Buffer) -> napi::Result<DecodedImage> {
    let image = image::load_from_memory(&data)
        .map_err(|error| napi::Error::from_reason(format!("failed to decode image: {error}")))?;
    let rgba = image.to_rgba8();
    Ok(DecodedImage {
        width: rgba.width(),
        height: rgba.height(),
        data: Buffer::from(rgba.into_raw()),
    })
}

#[napi(js_name = "encodePng")]
pub fn encode_png_binding(data: Buffer, width: u32, height: u32) -> napi::Result<Buffer> {
    let expected_len = width as usize * height as usize * 4;
    if data.len() != expected_len {
        return Err(napi::Error::from_reason(format!(
            "encodePng expected RGBA data length {expected_len} for {width}x{height}, received {} bytes",
            data.len()
        )));
    }
    encode_png(&data, width, height)
        .map(Buffer::from)
        .map_err(to_napi_error)
}

fn to_napi_error(error: anyhow::Error) -> napi::Error {
    napi::Error::from_reason(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::mesh::{decode_texture, prepare_meshes};
    use super::renderer::GpuRenderer;
    use super::types::{Camera, RenderScene, SceneLight, SceneMesh};

    #[test]
    fn empty_scene_prepares_no_meshes() {
        let scene = RenderScene::default();
        let meshes = prepare_meshes(&scene).unwrap();
        assert!(meshes.is_empty());
    }

    #[test]
    fn rejects_non_triangle_unindexed_mesh() {
        let scene = RenderScene {
            meshes: Some(vec![SceneMesh {
                positions: vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                ..SceneMesh::default()
            }]),
            ..RenderScene::default()
        };

        assert!(prepare_meshes(&scene).is_err());
    }

    #[test]
    fn accepts_indexed_quad() {
        let scene = RenderScene {
            meshes: Some(vec![SceneMesh {
                positions: vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                indices: Some(vec![0, 1, 2, 0, 2, 3]),
                color: Some(vec![1.0, 0.0, 0.0, 1.0]),
                ..SceneMesh::default()
            }]),
            ..RenderScene::default()
        };

        let meshes = prepare_meshes(&scene).unwrap();
        assert_eq!(meshes[0].vertices.len(), 4);
        assert_eq!(meshes[0].indices.as_ref().unwrap().len(), 6);
    }

    #[test]
    fn prepares_many_meshes_in_source_order() {
        let meshes = (0..12)
            .map(|index| SceneMesh {
                positions: vec![0.0, 0.0, 0.0, 0.04, 0.0, 0.0, 0.0, 0.04, 0.0],
                color: Some(vec![index as f64 / 12.0, 0.2, 0.6, 1.0]),
                sort_index: Some(index),
                ..SceneMesh::default()
            })
            .collect();
        let scene = RenderScene {
            meshes: Some(meshes),
            ..RenderScene::default()
        };

        let prepared = prepare_meshes(&scene).unwrap();
        assert_eq!(prepared.len(), 12);
        for (index, mesh) in prepared.iter().enumerate() {
            assert_eq!(mesh.sort_index, index as u32);
            assert!((mesh.base_color[0] - index as f32 / 12.0).abs() < 0.0001);
        }
    }

    #[test]
    fn accepts_mesh_with_uvs() {
        let scene = RenderScene {
            meshes: Some(vec![SceneMesh {
                positions: vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                uvs: Some(vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0]),
                ..SceneMesh::default()
            }]),
            ..RenderScene::default()
        };

        let meshes = prepare_meshes(&scene).unwrap();
        assert_eq!(meshes[0].vertices[0].uv, [0.0, 0.0]);
        assert_eq!(meshes[0].vertices[1].uv, [1.0, 0.0]);
        assert_eq!(meshes[0].vertices[2].uv, [0.0, 1.0]);
    }

    #[test]
    fn rejects_bad_uv_length() {
        let scene = RenderScene {
            meshes: Some(vec![SceneMesh {
                positions: vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                uvs: Some(vec![0.0, 0.0, 1.0]), // wrong length
                ..SceneMesh::default()
            }]),
            ..RenderScene::default()
        };

        assert!(prepare_meshes(&scene).is_err());
    }

    #[test]
    fn decodes_raw_rgba_texture() {
        let rgba = vec![
            255u8, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 0, 255,
        ];
        let tex = decode_texture(&rgba, Some(2), Some(2), 0).unwrap();
        assert_eq!(tex.width, 2);
        assert_eq!(tex.height, 2);
        assert_eq!(tex.rgba.len(), 16);
    }

    #[test]
    fn accepts_mesh_with_metallic_roughness_texture() {
        let scene = RenderScene {
            meshes: Some(vec![SceneMesh {
                positions: vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                uvs: Some(vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0]),
                metallic: Some(1.0),
                roughness: Some(0.5),
                metallic_roughness_texture: Some(vec![0u8, 128, 255, 255].into()),
                metallic_roughness_texture_width: Some(1),
                metallic_roughness_texture_height: Some(1),
                ..SceneMesh::default()
            }]),
            ..RenderScene::default()
        };
        let meshes = prepare_meshes(&scene).unwrap();
        assert!(meshes[0].metallic_roughness_texture.is_some());
        let mr_tex = meshes[0].metallic_roughness_texture.as_ref().unwrap();
        assert_eq!(mr_tex.width, 1);
        assert_eq!(mr_tex.height, 1);
    }

    #[test]
    fn accepts_mesh_with_physical_extension_maps() {
        let scene = RenderScene {
            meshes: Some(vec![SceneMesh {
                positions: vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                uvs: Some(vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0]),
                clearcoat_map: Some(vec![255u8, 0, 0, 255].into()),
                clearcoat_map_width: Some(1),
                clearcoat_map_height: Some(1),
                clearcoat_normal_map: Some(vec![128u8, 128, 255, 255].into()),
                clearcoat_normal_map_width: Some(1),
                clearcoat_normal_map_height: Some(1),
                transmission_map: Some(vec![128u8, 0, 0, 255].into()),
                transmission_map_width: Some(1),
                transmission_map_height: Some(1),
                thickness_map: Some(vec![0u8, 64, 0, 255].into()),
                thickness_map_width: Some(1),
                thickness_map_height: Some(1),
                specular_color_map: Some(vec![10u8, 20, 30, 255].into()),
                specular_color_map_width: Some(1),
                specular_color_map_height: Some(1),
                specular_intensity_map: Some(vec![0u8, 0, 0, 77].into()),
                specular_intensity_map_width: Some(1),
                specular_intensity_map_height: Some(1),
                ..SceneMesh::default()
            }]),
            ..RenderScene::default()
        };

        let meshes = prepare_meshes(&scene).unwrap();
        let mesh = &meshes[0];
        assert!(mesh.clearcoat_normal_map.is_some());
        let physical_maps = mesh.physical_maps.as_ref().unwrap();
        assert_eq!(physical_maps.scalar_map.width, 1);
        assert_eq!(physical_maps.scalar_map.height, 1);
        assert_eq!(physical_maps.scalar_map.rgba[0], 255);
        assert_eq!(physical_maps.scalar_map.rgba[2], 128);
        assert_eq!(physical_maps.scalar_map.rgba[3], 64);
        assert_eq!(&physical_maps.specular_map.rgba[..4], &[10, 20, 30, 77]);
    }

    /// End-to-end smoke test for directional shadow maps. Renders a ground
    /// quad that receives shadows from a box above it under a downward-
    /// pointing directional light. We can't pixel-match without a reference,
    /// but this exercises the shadow pipeline, WGSL compilation, and PCF
    /// sampling path end-to-end.
    #[test]
    fn renders_scene_with_directional_shadow() {
        let renderer = match GpuRenderer::new() {
            Ok(r) => r,
            Err(err) => {
                eprintln!("skipping: no wgpu adapter available ({err})");
                return;
            }
        };

        // Ground plane (receives shadow)
        let ground = SceneMesh {
            positions: vec![
                -5.0, 0.0, -5.0, 5.0, 0.0, -5.0, 5.0, 0.0, 5.0, -5.0, 0.0, 5.0,
            ],
            indices: Some(vec![0, 1, 2, 0, 2, 3]),
            normals: Some(vec![
                0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
            ]),
            color: Some(vec![0.8, 0.8, 0.8, 1.0]),
            receive_shadow: Some(true),
            ..SceneMesh::default()
        };
        // Occluder box (casts shadow)
        let occluder = SceneMesh {
            positions: vec![
                -0.5, 1.0, -0.5, 0.5, 1.0, -0.5, 0.5, 2.0, -0.5, -0.5, 2.0, -0.5, -0.5, 1.0, 0.5,
                0.5, 1.0, 0.5, 0.5, 2.0, 0.5, -0.5, 2.0, 0.5,
            ],
            indices: Some(vec![
                0, 1, 2, 0, 2, 3, 4, 6, 5, 4, 7, 6, 0, 3, 7, 0, 7, 4, 1, 5, 6, 1, 6, 2, 3, 2, 6, 3,
                6, 7, 0, 4, 5, 0, 5, 1,
            ]),
            color: Some(vec![0.9, 0.2, 0.2, 1.0]),
            cast_shadow: Some(true),
            ..SceneMesh::default()
        };

        let light = SceneLight {
            light_type: "directional".into(),
            color: Some(vec![1.0, 1.0, 1.0]),
            intensity: Some(1.0),
            position: Some(vec![3.0, 5.0, 3.0]),
            direction: Some(vec![-3.0, -5.0, -3.0]),
            cast_shadow: Some(true),
            shadow_map_size: Some(256),
            shadow_bias: Some(-0.0005),
            shadow_normal_bias: Some(0.02),
            shadow_camera_left: Some(-5.0),
            shadow_camera_right: Some(5.0),
            shadow_camera_top: Some(5.0),
            shadow_camera_bottom: Some(-5.0),
            shadow_camera_near: Some(0.1),
            shadow_camera_far: Some(20.0),
            ..SceneLight::default()
        };

        let scene = RenderScene {
            width: Some(64),
            height: Some(64),
            format: Some("rgba".into()),
            meshes: Some(vec![ground, occluder]),
            lights: Some(vec![light]),
            ..RenderScene::default()
        };
        let camera = Camera {
            eye: Some(vec![4.0, 3.0, 6.0]),
            target: Some(vec![0.0, 0.5, 0.0]),
            ..Camera::default()
        };

        let rgba = renderer
            .render(&scene, &camera)
            .expect("render should succeed");
        assert_eq!(rgba.len(), 64 * 64 * 4);
    }

    #[test]
    fn renders_scene_with_spot_shadow() {
        let renderer = match GpuRenderer::new() {
            Ok(r) => r,
            Err(err) => {
                eprintln!("skipping: no wgpu adapter available ({err})");
                return;
            }
        };

        let ground = SceneMesh {
            positions: vec![
                -4.0, 0.0, -4.0, 4.0, 0.0, -4.0, 4.0, 0.0, 4.0, -4.0, 0.0, 4.0,
            ],
            indices: Some(vec![0, 1, 2, 0, 2, 3]),
            normals: Some(vec![
                0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
            ]),
            color: Some(vec![0.75, 0.75, 0.75, 1.0]),
            receive_shadow: Some(true),
            ..SceneMesh::default()
        };
        let caster = SceneMesh {
            positions: vec![
                -0.5, 0.7, -0.5, 0.5, 0.7, -0.5, 0.5, 1.7, -0.5, -0.5, 1.7, -0.5, -0.5, 0.7, 0.5,
                0.5, 0.7, 0.5, 0.5, 1.7, 0.5, -0.5, 1.7, 0.5,
            ],
            indices: Some(vec![
                0, 1, 2, 0, 2, 3, 4, 6, 5, 4, 7, 6, 0, 3, 7, 0, 7, 4, 1, 5, 6, 1, 6, 2, 3, 2, 6, 3,
                6, 7, 0, 4, 5, 0, 5, 1,
            ]),
            color: Some(vec![0.9, 0.25, 0.2, 1.0]),
            cast_shadow: Some(true),
            ..SceneMesh::default()
        };
        let light = SceneLight {
            light_type: "spot".into(),
            color: Some(vec![1.0, 1.0, 1.0]),
            intensity: Some(8.0),
            position: Some(vec![0.0, 5.0, 3.0]),
            direction: Some(vec![0.0, -5.0, -3.0]),
            distance: Some(10.0),
            decay: Some(2.0),
            angle: Some(0.7),
            penumbra: Some(0.2),
            cast_shadow: Some(true),
            shadow_map_size: Some(256),
            shadow_bias: Some(-0.0005),
            shadow_normal_bias: Some(0.02),
            shadow_camera_near: Some(0.1),
            shadow_camera_far: Some(12.0),
            ..SceneLight::default()
        };

        let scene = RenderScene {
            width: Some(64),
            height: Some(64),
            format: Some("rgba".into()),
            meshes: Some(vec![ground, caster]),
            lights: Some(vec![light]),
            ..RenderScene::default()
        };
        let camera = Camera {
            eye: Some(vec![4.0, 3.0, 6.0]),
            target: Some(vec![0.0, 0.5, 0.0]),
            ..Camera::default()
        };

        let rgba = renderer
            .render(&scene, &camera)
            .expect("render should succeed");
        assert_eq!(rgba.len(), 64 * 64 * 4);
    }

    #[test]
    fn renders_scene_with_point_shadow() {
        let renderer = match GpuRenderer::new() {
            Ok(r) => r,
            Err(err) => {
                eprintln!("skipping: no wgpu adapter available ({err})");
                return;
            }
        };

        let ground = SceneMesh {
            positions: vec![
                -4.0, 0.0, -4.0, 4.0, 0.0, -4.0, 4.0, 0.0, 4.0, -4.0, 0.0, 4.0,
            ],
            indices: Some(vec![0, 1, 2, 0, 2, 3]),
            normals: Some(vec![
                0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
            ]),
            color: Some(vec![0.75, 0.75, 0.75, 1.0]),
            receive_shadow: Some(true),
            ..SceneMesh::default()
        };
        let caster = SceneMesh {
            positions: vec![
                -0.5, 0.7, -0.5, 0.5, 0.7, -0.5, 0.5, 1.7, -0.5, -0.5, 1.7, -0.5, -0.5, 0.7, 0.5,
                0.5, 0.7, 0.5, 0.5, 1.7, 0.5, -0.5, 1.7, 0.5,
            ],
            indices: Some(vec![
                0, 1, 2, 0, 2, 3, 4, 6, 5, 4, 7, 6, 0, 3, 7, 0, 7, 4, 1, 5, 6, 1, 6, 2, 3, 2, 6, 3,
                6, 7, 0, 4, 5, 0, 5, 1,
            ]),
            color: Some(vec![0.9, 0.25, 0.2, 1.0]),
            cast_shadow: Some(true),
            ..SceneMesh::default()
        };
        let light = SceneLight {
            light_type: "point".into(),
            color: Some(vec![1.0, 1.0, 1.0]),
            intensity: Some(45.0),
            position: Some(vec![0.0, 4.0, 2.0]),
            distance: Some(12.0),
            decay: Some(2.0),
            cast_shadow: Some(true),
            shadow_map_size: Some(256),
            shadow_bias: Some(-0.0005),
            shadow_normal_bias: Some(0.02),
            shadow_camera_near: Some(0.1),
            shadow_camera_far: Some(12.0),
            ..SceneLight::default()
        };

        let scene = RenderScene {
            width: Some(64),
            height: Some(64),
            format: Some("rgba".into()),
            meshes: Some(vec![ground, caster]),
            lights: Some(vec![light]),
            ..RenderScene::default()
        };
        let camera = Camera {
            eye: Some(vec![4.0, 3.0, 6.0]),
            target: Some(vec![0.0, 0.5, 0.0]),
            ..Camera::default()
        };

        let rgba = renderer
            .render(&scene, &camera)
            .expect("render should succeed");
        assert_eq!(rgba.len(), 64 * 64 * 4);
    }
}
