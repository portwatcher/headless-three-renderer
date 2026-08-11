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
pub struct NativeGpuOutputCapabilities {
    pub backend: String,
    pub texture: NativeGpuTextureCapability,
    pub dma_buf: NativeDmaBufCapability,
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
        }
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
