use std::sync::{Arc, Mutex};

use anyhow::{Result, anyhow, bail};

use super::media_conversion::MediaConverter;
use super::native_output::{backend_name, native_handle, native_handle_type};
use super::{Camera, GpuRenderer, RenderScene};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MediaOutputFormat {
    Rgba8,
    Nv12,
    P010,
}

impl MediaOutputFormat {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "rgba8" | "rgba8unorm" => Ok(Self::Rgba8),
            "nv12-planes" => Ok(Self::Nv12),
            "p010-planes" => Ok(Self::P010),
            _ => bail!("unsupported GPU media output format '{value}'"),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Rgba8 => "rgba8unorm",
            Self::Nv12 => "nv12-planes",
            Self::P010 => "p010-planes",
        }
    }

    fn plane_formats(self) -> &'static [wgpu::TextureFormat] {
        match self {
            Self::Rgba8 => &[wgpu::TextureFormat::Rgba8Unorm],
            Self::Nv12 => &[wgpu::TextureFormat::R8Unorm, wgpu::TextureFormat::Rg8Unorm],
            Self::P010 => &[
                wgpu::TextureFormat::R16Unorm,
                wgpu::TextureFormat::Rg16Unorm,
            ],
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OverflowPolicy {
    Error,
    DropNewest,
}

impl OverflowPolicy {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "error" => Ok(Self::Error),
            "drop-newest" => Ok(Self::DropNewest),
            _ => bail!("overflow must be 'error' or 'drop-newest'"),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct GpuFramePoolOptions {
    pub width: u32,
    pub height: u32,
    pub capacity: u32,
    pub format: MediaOutputFormat,
    pub overflow: OverflowPolicy,
}

#[derive(Clone, Debug, Default)]
pub struct GpuFramePoolStats {
    pub capacity: u32,
    pub available: u32,
    pub in_flight: u32,
    pub peak_in_flight: u32,
    pub submitted: u64,
    pub completed: u64,
    pub dropped: u64,
    pub rejected: u64,
    pub reused: u64,
    pub allocations: u64,
    pub retired: u32,
    pub closed: bool,
}

#[derive(Clone)]
pub(crate) struct MediaFrameResources {
    pub(super) rgba: wgpu::Texture,
    pub(super) y: Option<wgpu::Texture>,
    pub(super) uv: Option<wgpu::Texture>,
    pub(super) width: u32,
    pub(super) height: u32,
}

struct PoolSlot {
    resources: MediaFrameResources,
    leased: bool,
    uses: u64,
    retired: bool,
}

struct PoolInner {
    slots: Vec<PoolSlot>,
    stats: GpuFramePoolStats,
}

struct PoolState {
    options: GpuFramePoolOptions,
    inner: Mutex<PoolInner>,
}

pub struct GpuFramePool {
    renderer: Arc<GpuRenderer>,
    state: Arc<PoolState>,
    converter: MediaConverter,
}

pub struct MediaFrame {
    pub(super) renderer: Arc<GpuRenderer>,
    state: Arc<PoolState>,
    slot: Option<usize>,
    resources: MediaFrameResources,
    backend: wgpu::Backend,
    format: MediaOutputFormat,
    sequence: u64,
    pub(super) external_use_pending: bool,
    pub(super) readback_state: bool,
}

pub struct FrameReservation {
    state: Arc<PoolState>,
    slot: Option<usize>,
    resources: MediaFrameResources,
    sequence: u64,
}

pub struct PlaneReadback {
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub bytes_per_row: u32,
    pub format: &'static str,
}

impl GpuFramePool {
    pub fn new(renderer: Arc<GpuRenderer>, options: GpuFramePoolOptions) -> Result<Arc<Self>> {
        validate_options(&renderer, options)?;
        let converter = MediaConverter::new(&renderer.device, options.format)?;
        let mut slots = Vec::with_capacity(options.capacity as usize);
        for _ in 0..options.capacity {
            slots.push(PoolSlot {
                resources: create_resources(&renderer.device, options),
                leased: false,
                uses: 0,
                retired: false,
            });
        }
        let stats = GpuFramePoolStats {
            capacity: options.capacity,
            available: options.capacity,
            allocations: options.capacity as u64,
            ..Default::default()
        };
        Ok(Arc::new(Self {
            renderer,
            state: Arc::new(PoolState {
                options,
                inner: Mutex::new(PoolInner { slots, stats }),
            }),
            converter,
        }))
    }

    pub fn render(&self, scene: &RenderScene, camera: &Camera) -> Result<Option<MediaFrame>> {
        let Some(reservation) = self.reserve()? else {
            return Ok(None);
        };
        self.render_reserved(reservation, scene, camera).map(Some)
    }

    pub fn render_reserved(
        &self,
        mut reservation: FrameReservation,
        scene: &RenderScene,
        camera: &Camera,
    ) -> Result<MediaFrame> {
        if !Arc::ptr_eq(&reservation.state, &self.state) {
            bail!("GPU frame reservation belongs to another pool")
        }
        let slot_index = reservation.slot.take().expect("live frame reservation");
        let resources = reservation.resources.clone();
        let mut camera = camera.clone();
        camera.width = Some(self.state.options.width);
        camera.height = Some(self.state.options.height);
        let result = self
            .renderer
            .render_gpu_frame_into(scene, &camera, &resources.rgba)
            .and_then(|()| {
                self.converter
                    .convert(&self.renderer.device, &self.renderer.queue, &resources)
            });
        if let Err(error) = result {
            self.state.release(slot_index, false);
            return Err(error);
        }
        self.state.completed();
        Ok(MediaFrame {
            renderer: Arc::clone(&self.renderer),
            state: Arc::clone(&self.state),
            slot: Some(slot_index),
            resources,
            backend: self.renderer.backend,
            format: self.state.options.format,
            sequence: reservation.sequence,
            external_use_pending: false,
            readback_state: false,
        })
    }

    pub fn reserve(&self) -> Result<Option<FrameReservation>> {
        let mut inner = self.state.lock();
        if inner.stats.closed {
            bail!("GPU frame pool is closed")
        }
        let Some(index) = inner
            .slots
            .iter()
            .position(|slot| !slot.leased && !slot.retired)
        else {
            match self.state.options.overflow {
                OverflowPolicy::DropNewest => {
                    inner.stats.dropped += 1;
                    return Ok(None);
                }
                OverflowPolicy::Error => {
                    inner.stats.rejected += 1;
                    bail!("GPU frame pool is exhausted")
                }
            }
        };
        let sequence = inner.stats.submitted + 1;
        let (resources, reused) = {
            let slot = &mut inner.slots[index];
            slot.leased = true;
            slot.uses += 1;
            (slot.resources.clone(), slot.uses > 1)
        };
        if reused {
            inner.stats.reused += 1;
        }
        inner.stats.submitted += 1;
        inner.stats.available -= 1;
        inner.stats.in_flight += 1;
        inner.stats.peak_in_flight = inner.stats.peak_in_flight.max(inner.stats.in_flight);
        Ok(Some(FrameReservation {
            state: Arc::clone(&self.state),
            slot: Some(index),
            resources,
            sequence,
        }))
    }

    pub fn stats(&self) -> GpuFramePoolStats {
        self.state.lock().stats.clone()
    }

    pub fn close(&self) {
        self.state.lock().stats.closed = true;
    }

    pub fn options(&self) -> GpuFramePoolOptions {
        self.state.options
    }
}

impl Drop for GpuFramePool {
    fn drop(&mut self) {
        self.close();
    }
}

impl PoolState {
    fn lock(&self) -> std::sync::MutexGuard<'_, PoolInner> {
        self.inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn completed(&self) {
        self.lock().stats.completed += 1;
    }

    fn release(&self, index: usize, count_reuse: bool) {
        let mut inner = self.lock();
        if !inner.slots[index].leased {
            return;
        }
        inner.slots[index].leased = false;
        inner.stats.in_flight -= 1;
        inner.stats.available += 1;
        if count_reuse {
            // Reuse is counted on the next acquisition, not release.
        }
    }

    fn retire(&self, index: usize) {
        let mut inner = self.lock();
        if !inner.slots[index].leased {
            return;
        }
        inner.slots[index].leased = false;
        inner.slots[index].retired = true;
        inner.stats.in_flight -= 1;
        inner.stats.retired += 1;
    }
}

impl Drop for FrameReservation {
    fn drop(&mut self) {
        if let Some(index) = self.slot.take() {
            self.state.release(index, false);
        }
    }
}

impl MediaFrame {
    pub fn width(&self) -> u32 {
        self.resources.width
    }
    pub fn height(&self) -> u32 {
        self.resources.height
    }
    pub fn format(&self) -> &'static str {
        self.format.name()
    }
    pub fn backend(&self) -> &'static str {
        backend_name(self.backend)
    }
    pub fn handle_type(&self) -> Option<&'static str> {
        native_handle_type(self.backend)
    }
    pub fn sequence(&self) -> u64 {
        self.sequence
    }
    pub fn released(&self) -> bool {
        self.slot.is_none()
    }
    pub fn plane_count(&self) -> usize {
        self.format.plane_formats().len()
    }

    pub fn plane_format(&self, index: usize) -> Result<&'static str> {
        match (self.format, index) {
            (MediaOutputFormat::Rgba8, 0) => Ok("rgba8unorm"),
            (MediaOutputFormat::Nv12, 0) => Ok("r8unorm-y"),
            (MediaOutputFormat::Nv12, 1) => Ok("rg8unorm-uv"),
            (MediaOutputFormat::P010, 0) => Ok("r16unorm-y10-msb"),
            (MediaOutputFormat::P010, 1) => Ok("rg16unorm-uv10-msb"),
            _ => bail!("media plane index {index} is out of range"),
        }
    }

    pub fn plane_dimensions(&self, index: usize) -> Result<(u32, u32)> {
        if index >= self.plane_count() {
            bail!("media plane index {index} is out of range")
        }
        Ok(if index == 0 || self.format == MediaOutputFormat::Rgba8 {
            (self.width(), self.height())
        } else {
            (self.width() / 2, self.height() / 2)
        })
    }

    pub fn plane_bytes_per_row(&self, index: usize) -> Result<u32> {
        let (width, _) = self.plane_dimensions(index)?;
        let bytes = match (self.format, index) {
            (MediaOutputFormat::Rgba8, 0) => 4,
            (MediaOutputFormat::Nv12, 0) => 1,
            (MediaOutputFormat::Nv12, 1) => 2,
            (MediaOutputFormat::P010, 0) => 2,
            (MediaOutputFormat::P010, 1) => 4,
            _ => unreachable!(),
        };
        Ok(width * bytes)
    }

    pub fn plane_handle(&mut self, index: usize) -> Result<u64> {
        self.ensure_live()?;
        let texture = self.plane_texture(index)?;
        let handle = native_handle(texture, self.backend).ok_or_else(|| {
            anyhow!(
                "native GPU texture handles are unsupported for {}",
                self.backend()
            )
        })?;
        self.external_use_pending = true;
        Ok(handle)
    }

    pub fn plane_texture(&self, index: usize) -> Result<&wgpu::Texture> {
        self.ensure_live()?;
        match (self.format, index) {
            (MediaOutputFormat::Rgba8, 0) => Ok(&self.resources.rgba),
            (_, 0) => self
                .resources
                .y
                .as_ref()
                .ok_or_else(|| anyhow!("missing Y plane")),
            (_, 1) => self
                .resources
                .uv
                .as_ref()
                .ok_or_else(|| anyhow!("missing UV plane")),
            _ => bail!("media plane index {index} is out of range"),
        }
    }

    pub fn expected_state(&self, index: usize) -> Result<&'static str> {
        self.plane_dimensions(index)?;
        if self.readback_state {
            return Ok(match self.backend {
                wgpu::Backend::Vulkan => "VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL",
                wgpu::Backend::Dx12 => "D3D12_RESOURCE_STATE_COPY_SOURCE",
                wgpu::Backend::Metal => {
                    "metal-resource-state-untracked; same-command-queue ordering required"
                }
                _ => "wgpu-backend-managed-copy-source",
            });
        }
        Ok(match (self.backend, self.format) {
            (wgpu::Backend::Vulkan, MediaOutputFormat::Rgba8) => {
                "VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL"
            }
            (wgpu::Backend::Vulkan, _) => "VK_IMAGE_LAYOUT_GENERAL",
            (wgpu::Backend::Dx12, MediaOutputFormat::Rgba8) => "D3D12_RESOURCE_STATE_COPY_DEST",
            (wgpu::Backend::Dx12, _) => "D3D12_RESOURCE_STATE_UNORDERED_ACCESS",
            (wgpu::Backend::Metal, _) => {
                "metal-resource-state-untracked; same-command-queue ordering required"
            }
            _ => "wgpu-backend-managed",
        })
    }

    pub fn complete_external_use(&mut self) -> Result<()> {
        self.ensure_live()?;
        self.external_use_pending = false;
        Ok(())
    }

    pub fn release(&mut self) -> Result<()> {
        if self.external_use_pending {
            bail!(
                "external GPU use is pending; finish external work, restore every plane to requiredStateOnRelease, then call completeExternalUse()"
            )
        }
        if let Some(index) = self.slot.take() {
            self.state.release(index, true);
        }
        Ok(())
    }

    fn ensure_live(&self) -> Result<()> {
        if self.released() {
            bail!("GPU media frame lease has been released")
        }
        Ok(())
    }
}

impl Drop for MediaFrame {
    fn drop(&mut self) {
        if let Some(index) = self.slot.take() {
            if self.external_use_pending {
                self.state.retire(index);
            } else {
                self.state.release(index, true);
            }
        }
    }
}

fn validate_options(renderer: &GpuRenderer, options: GpuFramePoolOptions) -> Result<()> {
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
    Ok(())
}

fn create_resources(device: &wgpu::Device, options: GpuFramePoolOptions) -> MediaFrameResources {
    let rgba = create_texture(
        device,
        options.width,
        options.height,
        wgpu::TextureFormat::Rgba8Unorm,
        "RGBA output",
        false,
    );
    let (y, uv) = match options.format {
        MediaOutputFormat::Rgba8 => (None, None),
        MediaOutputFormat::Nv12 => (
            Some(create_texture(
                device,
                options.width,
                options.height,
                wgpu::TextureFormat::R8Unorm,
                "NV12 Y plane",
                true,
            )),
            Some(create_texture(
                device,
                options.width / 2,
                options.height / 2,
                wgpu::TextureFormat::Rg8Unorm,
                "NV12 UV plane",
                true,
            )),
        ),
        MediaOutputFormat::P010 => (
            Some(create_texture(
                device,
                options.width,
                options.height,
                wgpu::TextureFormat::R16Unorm,
                "P010 Y plane",
                true,
            )),
            Some(create_texture(
                device,
                options.width / 2,
                options.height / 2,
                wgpu::TextureFormat::Rg16Unorm,
                "P010 UV plane",
                true,
            )),
        ),
    };
    MediaFrameResources {
        rgba,
        y,
        uv,
        width: options.width,
        height: options.height,
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
    if storage {
        usage |= wgpu::TextureUsages::STORAGE_BINDING;
    } else {
        usage |= wgpu::TextureUsages::COPY_DST;
    }
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
