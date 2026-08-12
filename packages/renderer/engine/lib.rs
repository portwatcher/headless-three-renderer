mod ibl;
mod lights;
mod media_output;
mod mesh;
mod native_output;
mod renderer;
mod settings;
mod shader;
mod types;
mod util;

use std::sync::{Arc, OnceLock};

use napi::bindgen_prelude::Buffer;
use napi_derive::napi;

use media_output::{NativeGpuFramePool, NativeGpuFramePoolOptions, parse_pool_options};
use native_output::{NativeGpuFrameLease, NativeGpuOutputCapabilities};
use renderer::{GpuFramePool, GpuRenderer};
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
    inner: Arc<GpuRenderer>,
}

#[napi]
impl NativeRenderer {
    #[napi(constructor)]
    pub fn new() -> napi::Result<Self> {
        Ok(Self {
            inner: Arc::new(GpuRenderer::new().map_err(to_napi_error)?),
        })
    }

    #[napi]
    pub fn render(&self, scene: RenderScene, camera: Camera) -> napi::Result<Buffer> {
        self.inner
            .render(&scene, &camera)
            .map(Buffer::from)
            .map_err(to_napi_error)
    }

    #[napi]
    pub fn get_gpu_output_capabilities(&self) -> NativeGpuOutputCapabilities {
        self.inner.gpu_output_capabilities().into()
    }

    #[napi]
    pub fn render_gpu_frame(
        &self,
        scene: RenderScene,
        camera: Camera,
    ) -> napi::Result<NativeGpuFrameLease> {
        self.inner
            .render_gpu_frame(&scene, &camera)
            .map(NativeGpuFrameLease::new)
            .map_err(to_napi_error)
    }

    #[napi]
    pub fn create_gpu_frame_pool(
        &self,
        options: NativeGpuFramePoolOptions,
    ) -> napi::Result<NativeGpuFramePool> {
        let options = parse_pool_options(options)?;
        GpuFramePool::new(Arc::clone(&self.inner), options)
            .map(NativeGpuFramePool::new)
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
mod tests;
