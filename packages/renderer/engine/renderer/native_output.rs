use super::*;

#[derive(Clone, Debug)]
pub struct GpuOutputCapabilities {
    pub backend: &'static str,
    pub texture_supported: bool,
    pub texture_handle_type: Option<&'static str>,
    pub texture_reason: Option<&'static str>,
    pub dmabuf_supported: bool,
    pub dmabuf_reason: Option<&'static str>,
    pub encoder_prerequisites_ready: bool,
    pub encoder_prerequisites_detail: String,
    pub nv12_planes_supported: bool,
    pub p010_planes_supported: bool,
    pub i420_planes_supported: bool,
}

pub struct GpuFrame {
    texture: wgpu::Texture,
    backend: wgpu::Backend,
    pub width: u32,
    pub height: u32,
}

impl GpuRenderer {
    pub fn gpu_output_capabilities(&self) -> GpuOutputCapabilities {
        let handle_type = native_handle_type(self.backend);
        let texture_reason = handle_type.is_none().then_some(
            "the active wgpu backend does not expose a native texture handle on this platform",
        );
        let dmabuf_reason = if cfg!(target_os = "linux") && self.backend == wgpu::Backend::Vulkan {
            "wgpu-managed Vulkan textures are not allocated with exportable DMA-BUF external-memory flags"
        } else {
            "DMA-BUF export requires the Linux Vulkan backend"
        };
        GpuOutputCapabilities {
            backend: backend_name(self.backend),
            texture_supported: handle_type.is_some(),
            texture_handle_type: handle_type,
            texture_reason,
            dmabuf_supported: false,
            dmabuf_reason: Some(dmabuf_reason),
            encoder_prerequisites_ready: self.encoder_prerequisites.ready,
            encoder_prerequisites_detail: self.encoder_prerequisites.detail.clone(),
            nv12_planes_supported: self.media_nv12_planes_supported,
            p010_planes_supported: self.media_p010_planes_supported,
            i420_planes_supported: self.media_i420_planes_supported,
        }
    }
}

impl GpuFrame {
    pub(super) fn new(
        texture: wgpu::Texture,
        backend: wgpu::Backend,
        width: u32,
        height: u32,
    ) -> Self {
        Self {
            texture,
            backend,
            width,
            height,
        }
    }

    pub fn backend(&self) -> &'static str {
        backend_name(self.backend)
    }

    pub fn handle_type(&self) -> Option<&'static str> {
        native_handle_type(self.backend)
    }

    /// Returns a borrowed native resource handle. The lease that owns this
    /// `GpuFrame` must remain alive, and callers must never destroy the handle.
    pub fn native_handle(&self) -> Result<u64> {
        native_handle(&self.texture, self.backend).ok_or_else(|| {
            anyhow!(
                "native GPU texture handles are unsupported for the active {} backend",
                self.backend()
            )
        })
    }
}

pub(super) fn backend_name(backend: wgpu::Backend) -> &'static str {
    match backend {
        wgpu::Backend::Vulkan => "vulkan",
        wgpu::Backend::Metal => "metal",
        wgpu::Backend::Dx12 => "dx12",
        wgpu::Backend::Gl => "gles",
        wgpu::Backend::BrowserWebGpu => "browser-webgpu",
        wgpu::Backend::Noop => "noop",
    }
}

pub(super) fn native_handle_type(backend: wgpu::Backend) -> Option<&'static str> {
    match backend {
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        wgpu::Backend::Metal => Some("metal-texture"),
        #[cfg(any(target_os = "linux", target_os = "windows", target_os = "android"))]
        wgpu::Backend::Vulkan => Some("vulkan-image"),
        #[cfg(windows)]
        wgpu::Backend::Dx12 => Some("d3d12-resource"),
        _ => None,
    }
}

pub(super) fn native_handle(texture: &wgpu::Texture, backend: wgpu::Backend) -> Option<u64> {
    match backend {
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        wgpu::Backend::Metal => {
            // SAFETY: The returned pointer is borrowed and the GpuFrame lease
            // owns the wgpu texture for the entire documented validity period.
            let hal = unsafe { texture.as_hal::<wgpu::hal::api::Metal>() }?;
            let pointer = hal.raw_handle() as *const _ as *const ();
            Some(pointer as usize as u64)
        }
        #[cfg(any(target_os = "linux", target_os = "windows", target_os = "android"))]
        wgpu::Backend::Vulkan => {
            use ash::vk::Handle;
            // SAFETY: The image is only borrowed; callers are explicitly
            // prohibited from destroying it while the lease is live.
            let hal = unsafe { texture.as_hal::<wgpu::hal::api::Vulkan>() }?;
            Some(unsafe { hal.raw_handle() }.as_raw())
        }
        #[cfg(windows)]
        wgpu::Backend::Dx12 => {
            use windows_core::Interface;
            // SAFETY: The COM resource remains owned by the live frame lease.
            let hal = unsafe { texture.as_hal::<wgpu::hal::api::Dx12>() }?;
            Some(unsafe { hal.raw_resource() }.as_raw() as usize as u64)
        }
        _ => None,
    }
}
