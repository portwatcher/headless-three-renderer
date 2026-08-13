use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use anyhow::{Result, bail};

use super::GpuRenderer;
use super::media_pool::{GpuFramePoolOptions, MediaOutputFormat};
use crate::util::align_to;

#[derive(Clone)]
pub(super) struct MediaFrameResources {
    pub(super) rgba: wgpu::Texture,
    pub(super) y: Option<wgpu::Texture>,
    pub(super) uv: Option<wgpu::Texture>,
    pub(super) u: Option<wgpu::Texture>,
    pub(super) v: Option<wgpu::Texture>,
    pub(super) packed_i420: Option<wgpu::Buffer>,
    pub(super) i420_readback: Option<wgpu::Buffer>,
    pub(super) readback_in_flight: Arc<AtomicBool>,
    pub(super) width: u32,
    pub(super) height: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct I420Layout {
    pub byte_length: usize,
    pub gpu_readback_bytes: usize,
    pub strides: [u32; 3],
    pub offsets: [u32; 3],
}

pub(super) fn validate_options(renderer: &GpuRenderer, options: GpuFramePoolOptions) -> Result<()> {
    if options.width == 0 || options.height == 0 {
        bail!("pool width and height must be positive")
    }
    if options.width > 8192 || options.height > 8192 {
        bail!("pool dimensions must not exceed 8192")
    }
    if options.capacity == 0 || options.capacity > 64 {
        bail!("pool capacity must be between 1 and 64")
    }
    if options.format != MediaOutputFormat::Rgba8
        && (!options.width.is_multiple_of(2) || !options.height.is_multiple_of(2))
    {
        bail!("{} requires even width and height", options.format.name())
    }
    if options.format == MediaOutputFormat::Nv12 && !renderer.media_nv12_planes_supported {
        bail!("nv12-planes is unsupported: adapter lacks writable R8/RG8 storage textures")
    }
    if options.format == MediaOutputFormat::P010 && !renderer.media_p010_planes_supported {
        bail!("p010-planes is unsupported: adapter lacks 16-bit normalized storage textures")
    }
    if options.format == MediaOutputFormat::I420 && !renderer.media_i420_planes_supported {
        bail!("i420-planes is unsupported: adapter lacks writable R8 storage textures")
    }
    if options.format == MediaOutputFormat::I420 {
        validate_i420_buffer_limits(renderer, options)?;
    }
    Ok(())
}

fn validate_i420_buffer_limits(renderer: &GpuRenderer, options: GpuFramePoolOptions) -> Result<()> {
    let bytes = i420_layout(options.width, options.height).gpu_readback_bytes as u64;
    if bytes > renderer.device.limits().max_buffer_size {
        bail!(
            "packed I420 output requires {bytes} bytes, above device max_buffer_size {}",
            renderer.device.limits().max_buffer_size
        )
    }
    if bytes > renderer.device.limits().max_storage_buffer_binding_size {
        bail!(
            "packed I420 output requires {bytes} bytes, above max_storage_buffer_binding_size {}",
            renderer.device.limits().max_storage_buffer_binding_size
        )
    }
    Ok(())
}

pub(super) fn create_resources(
    device: &wgpu::Device,
    options: GpuFramePoolOptions,
) -> MediaFrameResources {
    let rgba = create_texture(
        device,
        options.width,
        options.height,
        wgpu::TextureFormat::Rgba8Unorm,
        "RGBA output",
        false,
    );
    let (y, uv, u, v) = create_yuv_planes(device, options);
    let layout = (options.format == MediaOutputFormat::I420)
        .then(|| i420_layout(options.width, options.height));
    let packed_i420 = layout.map(|layout| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("headless-three packed I420 GPU buffer"),
            size: layout.gpu_readback_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    });
    let i420_readback = layout.map(|layout| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("headless-three packed I420 readback buffer"),
            size: layout.gpu_readback_bytes as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        })
    });
    MediaFrameResources {
        rgba,
        y,
        uv,
        u,
        v,
        packed_i420,
        i420_readback,
        readback_in_flight: Arc::new(AtomicBool::new(false)),
        width: options.width,
        height: options.height,
    }
}

fn create_yuv_planes(
    device: &wgpu::Device,
    options: GpuFramePoolOptions,
) -> (
    Option<wgpu::Texture>,
    Option<wgpu::Texture>,
    Option<wgpu::Texture>,
    Option<wgpu::Texture>,
) {
    match options.format {
        MediaOutputFormat::Rgba8 => (None, None, None, None),
        MediaOutputFormat::Nv12 => (
            plane(
                device,
                options.width,
                options.height,
                wgpu::TextureFormat::R8Unorm,
                "NV12 Y plane",
            ),
            plane(
                device,
                options.width / 2,
                options.height / 2,
                wgpu::TextureFormat::Rg8Unorm,
                "NV12 UV plane",
            ),
            None,
            None,
        ),
        MediaOutputFormat::P010 => (
            plane(
                device,
                options.width,
                options.height,
                wgpu::TextureFormat::R16Unorm,
                "P010 Y plane",
            ),
            plane(
                device,
                options.width / 2,
                options.height / 2,
                wgpu::TextureFormat::Rg16Unorm,
                "P010 UV plane",
            ),
            None,
            None,
        ),
        MediaOutputFormat::I420 => (
            plane(
                device,
                options.width,
                options.height,
                wgpu::TextureFormat::R8Unorm,
                "I420 Y plane",
            ),
            None,
            plane(
                device,
                options.width / 2,
                options.height / 2,
                wgpu::TextureFormat::R8Unorm,
                "I420 U plane",
            ),
            plane(
                device,
                options.width / 2,
                options.height / 2,
                wgpu::TextureFormat::R8Unorm,
                "I420 V plane",
            ),
        ),
    }
}

fn plane(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    label: &'static str,
) -> Option<wgpu::Texture> {
    Some(create_texture(device, width, height, format, label, true))
}

pub(super) fn i420_layout(width: u32, height: u32) -> I420Layout {
    let y_length = width * height;
    let chroma_length = (width / 2) * (height / 2);
    let byte_length = (y_length + 2 * chroma_length) as usize;
    I420Layout {
        byte_length,
        gpu_readback_bytes: align_to(byte_length as u32, 4) as usize,
        strides: [width, width / 2, width / 2],
        offsets: [0, y_length, y_length + chroma_length],
    }
}

fn create_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    label: &'static str,
    storage: bool,
) -> wgpu::Texture {
    let mut usage = wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::TEXTURE_BINDING;
    usage |= if storage {
        wgpu::TextureUsages::STORAGE_BINDING
    } else {
        wgpu::TextureUsages::COPY_DST
    };
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage,
        view_formats: &[],
    })
}
