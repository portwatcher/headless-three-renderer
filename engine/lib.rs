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

const COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth24Plus;
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

fn to_napi_error(error: anyhow::Error) -> napi::Error {
    napi::Error::from_reason(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::mesh::{decode_texture, prepare_meshes};
    use super::types::{RenderScene, SceneMesh};

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
        let rgba = vec![255u8, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 0, 255];
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
}
