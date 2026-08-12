use napi::bindgen_prelude::BigInt;
use napi_derive::napi;

use crate::renderer::{GpuFrame, GpuOutputCapabilities};

#[napi(object)]
pub struct NativeGpuTextureCapability {
    pub supported: bool,
    pub handle_type: Option<String>,
    pub synchronization: String,
    pub scope: String,
    pub format: String,
    pub usage: String,
    pub layout: String,
    pub reason: Option<String>,
}

#[napi(object)]
pub struct NativeDmaBufCapability {
    pub supported: bool,
    pub reason: Option<String>,
}

#[napi(object)]
pub struct NativeGpuMediaFormatCapability {
    pub format: String,
    pub supported: bool,
    pub storage: String,
    pub plane_formats: Vec<String>,
    pub reason: Option<String>,
    pub color_matrix: Option<String>,
    pub color_range: Option<String>,
    pub chroma_siting: Option<String>,
}

#[napi(object)]
pub struct NativeEncoderSurfaceCapability {
    pub supported: bool,
    pub reason: String,
}

#[napi(object)]
pub struct NativeGpuOutputCapabilities {
    pub backend: String,
    pub texture: NativeGpuTextureCapability,
    pub dma_buf: NativeDmaBufCapability,
    pub encoder_surface: NativeEncoderSurfaceCapability,
    pub media_formats: Vec<NativeGpuMediaFormatCapability>,
}

impl From<GpuOutputCapabilities> for NativeGpuOutputCapabilities {
    fn from(value: GpuOutputCapabilities) -> Self {
        Self {
            backend: value.backend.to_owned(),
            texture: NativeGpuTextureCapability {
                supported: value.texture_supported,
                handle_type: value.texture_handle_type.map(str::to_owned),
                synchronization: "submission-complete".to_owned(),
                scope: "same-renderer-device".to_owned(),
                format: "rgba8unorm".to_owned(),
                usage: "copy-dst|copy-src|texture-binding".to_owned(),
                layout: "backend-managed-copy-dst".to_owned(),
                reason: value.texture_reason.map(str::to_owned),
            },
            dma_buf: NativeDmaBufCapability {
                supported: value.dmabuf_supported,
                reason: value.dmabuf_reason.map(str::to_owned),
            },
            encoder_surface: NativeEncoderSurfaceCapability {
                supported: false,
                reason: encoder_surface_reason(value.backend).to_owned(),
            },
            media_formats: vec![
                media_format("rgba8unorm", true, "single-texture", &["rgba8unorm"], None),
                media_format(
                    "nv12-planes",
                    value.nv12_planes_supported,
                    "separate-textures",
                    &["r8unorm-y", "rg8unorm-uv"],
                    (!value.nv12_planes_supported)
                        .then_some("adapter lacks writable R8/RG8 storage texture support"),
                ),
                media_format(
                    "p010-planes",
                    value.p010_planes_supported,
                    "separate-textures",
                    &["r16unorm-y10-msb", "rg16unorm-uv10-msb"],
                    (!value.p010_planes_supported)
                        .then_some("adapter lacks wgpu TEXTURE_FORMAT_16BIT_NORM storage support"),
                ),
            ],
        }
    }
}

fn media_format(
    format: &str,
    supported: bool,
    storage: &str,
    plane_formats: &[&str],
    reason: Option<&str>,
) -> NativeGpuMediaFormatCapability {
    let yuv = format != "rgba8unorm";
    NativeGpuMediaFormatCapability {
        format: format.to_owned(),
        supported,
        storage: storage.to_owned(),
        plane_formats: plane_formats
            .iter()
            .map(|value| (*value).to_owned())
            .collect(),
        reason: reason.map(str::to_owned),
        color_matrix: yuv.then(|| "bt709".to_owned()),
        color_range: yuv.then(|| "limited".to_owned()),
        chroma_siting: yuv.then(|| "centered-2x2-box".to_owned()),
    }
}

fn encoder_surface_reason(backend: &str) -> &'static str {
    match backend {
        "vulkan" => {
            "wgpu 29 cannot safely create and track writable DRM-modifier multi-planar images with exportable memory and external synchronization"
        }
        "metal" => {
            "wgpu-managed Metal textures are not IOSurface/CVPixelBuffer-backed encoder surfaces"
        }
        "dx12" => {
            "wgpu-managed D3D12 textures are not allocated as shared encoder resources with shared fences"
        }
        _ => "the active backend has no encoder-native surface export implementation",
    }
}

#[napi]
pub struct NativeGpuFrameLease {
    frame: Option<GpuFrame>,
    width: u32,
    height: u32,
    backend: String,
    handle_type: String,
}

impl NativeGpuFrameLease {
    pub fn new(frame: GpuFrame) -> Self {
        let width = frame.width;
        let height = frame.height;
        let backend = frame.backend().to_owned();
        let handle_type = frame.handle_type().unwrap_or("unsupported").to_owned();
        Self {
            frame: Some(frame),
            width,
            height,
            backend,
            handle_type,
        }
    }

    fn live_frame(&self) -> napi::Result<&GpuFrame> {
        self.frame
            .as_ref()
            .ok_or_else(|| napi::Error::from_reason("GPU frame lease has been released".to_owned()))
    }
}

#[napi]
impl NativeGpuFrameLease {
    #[napi(getter)]
    pub fn width(&self) -> u32 {
        self.width
    }

    #[napi(getter)]
    pub fn height(&self) -> u32 {
        self.height
    }

    #[napi(getter)]
    pub fn format(&self) -> &'static str {
        "rgba8unorm"
    }

    #[napi(getter)]
    pub fn backend(&self) -> String {
        self.backend.clone()
    }

    #[napi(getter)]
    pub fn handle_type(&self) -> String {
        self.handle_type.clone()
    }

    #[napi(getter)]
    pub fn released(&self) -> bool {
        self.frame.is_none()
    }

    #[napi(getter)]
    pub fn completed(&self) -> bool {
        true
    }

    #[napi(getter)]
    pub fn scope(&self) -> &'static str {
        "same-renderer-device"
    }

    #[napi(getter)]
    pub fn usage(&self) -> &'static str {
        "copy-dst|copy-src|texture-binding"
    }

    #[napi(getter)]
    pub fn layout(&self) -> &'static str {
        "backend-managed-copy-dst"
    }

    #[napi]
    pub fn native_handle(&self) -> napi::Result<BigInt> {
        let handle = self
            .live_frame()?
            .native_handle()
            .map_err(|error| napi::Error::from_reason(error.to_string()))?;
        Ok(BigInt {
            sign_bit: false,
            words: vec![handle],
        })
    }

    #[napi]
    pub fn export_dma_buf(&self) -> napi::Result<()> {
        let frame = self.live_frame()?;
        let reason = if frame.backend() == "vulkan" && cfg!(target_os = "linux") {
            "wgpu-managed Vulkan textures are not allocated with exportable DMA-BUF external-memory flags"
        } else {
            "DMA-BUF export requires the Linux Vulkan backend"
        };
        Err(napi::Error::from_reason(format!(
            "DMA-BUF export is unsupported: {reason}"
        )))
    }

    #[napi]
    pub fn release(&mut self) {
        self.frame.take();
    }
}
