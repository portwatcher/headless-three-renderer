use std::sync::Arc;

use napi::bindgen_prelude::{AsyncTask, BigInt, Buffer};
use napi::{Env, Task};
use napi_derive::napi;

use crate::renderer::{
    FrameReservation, GpuFramePool, GpuFramePoolOptions, GpuFramePoolStats, MediaFrame,
    MediaOutputFormat, OverflowPolicy, PlaneReadback, PlaneReadbackJob,
};
use crate::types::{Camera, RenderScene};

#[napi(object)]
pub struct NativeGpuFramePoolOptions {
    pub width: u32,
    pub height: u32,
    pub capacity: Option<u32>,
    pub format: Option<String>,
    pub overflow: Option<String>,
}

#[napi(object)]
pub struct NativeGpuFramePoolStats {
    pub capacity: u32,
    pub available: u32,
    pub in_flight: u32,
    pub peak_in_flight: u32,
    pub submitted: i64,
    pub completed: i64,
    pub dropped: i64,
    pub rejected: i64,
    pub reused: i64,
    pub allocations: i64,
    pub retired: u32,
    pub closed: bool,
}

impl From<GpuFramePoolStats> for NativeGpuFramePoolStats {
    fn from(value: GpuFramePoolStats) -> Self {
        Self {
            capacity: value.capacity,
            available: value.available,
            in_flight: value.in_flight,
            peak_in_flight: value.peak_in_flight,
            submitted: value.submitted as i64,
            completed: value.completed as i64,
            dropped: value.dropped as i64,
            rejected: value.rejected as i64,
            reused: value.reused as i64,
            allocations: value.allocations as i64,
            retired: value.retired,
            closed: value.closed,
        }
    }
}

#[napi(object)]
#[derive(Clone)]
pub struct NativeGpuPlaneInfo {
    pub index: u32,
    pub format: String,
    pub width: u32,
    pub height: u32,
    pub bytes_per_row: u32,
    pub row_semantics: String,
    pub expected_state_before_use: String,
    pub required_state_on_release: String,
}

#[napi(object)]
pub struct NativeGpuPlaneData {
    pub index: u32,
    pub format: String,
    pub width: u32,
    pub height: u32,
    pub bytes_per_row: u32,
    pub data: Buffer,
}

#[napi]
pub struct NativeGpuMediaFrameLease {
    frame: Option<MediaFrame>,
    width: u32,
    height: u32,
    format: String,
    backend: String,
    handle_type: String,
    sequence: i64,
}

impl NativeGpuMediaFrameLease {
    fn new(frame: MediaFrame) -> napi::Result<Self> {
        Ok(Self {
            width: frame.width(),
            height: frame.height(),
            format: frame.format().to_owned(),
            backend: frame.backend().to_owned(),
            handle_type: frame.handle_type().unwrap_or("unsupported").to_owned(),
            sequence: frame.sequence() as i64,
            frame: Some(frame),
        })
    }

    fn live_frame(&self) -> napi::Result<&MediaFrame> {
        self.frame.as_ref().ok_or_else(|| {
            napi::Error::from_reason("GPU media frame lease has been released".to_owned())
        })
    }
}

#[napi]
impl NativeGpuMediaFrameLease {
    #[napi(getter)]
    pub fn width(&self) -> u32 {
        self.width
    }
    #[napi(getter)]
    pub fn height(&self) -> u32 {
        self.height
    }
    #[napi(getter)]
    pub fn format(&self) -> String {
        self.format.clone()
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
    pub fn sequence(&self) -> i64 {
        self.sequence
    }
    #[napi(getter)]
    pub fn released(&self) -> bool {
        self.frame.is_none()
    }
    #[napi(getter)]
    pub fn ready(&self) -> bool {
        self.frame.is_some()
    }

    #[napi]
    pub fn plane_info(&self, index: u32) -> napi::Result<NativeGpuPlaneInfo> {
        let frame = self.live_frame()?;
        let index_usize = index as usize;
        let (width, height) = frame.plane_dimensions(index_usize).map_err(to_napi_error)?;
        let state = frame
            .expected_state(index_usize)
            .map_err(to_napi_error)?
            .to_owned();
        Ok(NativeGpuPlaneInfo {
            index,
            format: frame
                .plane_format(index_usize)
                .map_err(to_napi_error)?
                .to_owned(),
            width,
            height,
            bytes_per_row: frame
                .plane_bytes_per_row(index_usize)
                .map_err(to_napi_error)?,
            row_semantics: "tightly-packed-logical-row; native texture pitch is backend-managed"
                .to_owned(),
            expected_state_before_use: state.clone(),
            required_state_on_release: state,
        })
    }

    #[napi]
    pub fn plane_handle(&mut self, index: u32) -> napi::Result<BigInt> {
        let handle = self
            .frame
            .as_mut()
            .ok_or_else(|| napi::Error::from_reason("GPU media frame lease has been released"))?
            .plane_handle(index as usize)
            .map_err(to_napi_error)?;
        Ok(BigInt {
            sign_bit: false,
            words: vec![handle],
        })
    }

    #[napi]
    pub fn read_planes(&mut self) -> napi::Result<AsyncTask<ReadbackTask>> {
        let frame = self.frame.as_mut().ok_or_else(|| {
            napi::Error::from_reason("GPU media frame lease has been released".to_owned())
        })?;
        let job = frame.begin_plane_readback().map_err(to_napi_error)?;
        Ok(AsyncTask::new(ReadbackTask { job: Some(job) }))
    }

    #[napi]
    pub fn export_dma_buf(&self) -> napi::Result<()> {
        self.live_frame()?;
        Err(napi::Error::from_reason(
            "DMA-BUF export is unsupported: separate plane textures are not an encoder-native DRM-modifier multi-planar surface, and wgpu 29 does not expose safe external-memory synchronization"
                .to_owned(),
        ))
    }

    #[napi]
    pub fn complete_external_use(&mut self) -> napi::Result<()> {
        self.frame
            .as_mut()
            .ok_or_else(|| napi::Error::from_reason("GPU media frame lease has been released"))?
            .complete_external_use()
            .map_err(to_napi_error)
    }

    #[napi]
    pub fn release(&mut self) -> napi::Result<()> {
        if let Some(frame) = self.frame.as_mut() {
            frame.release().map_err(to_napi_error)?;
        }
        self.frame.take();
        Ok(())
    }
}

#[napi]
pub struct NativeGpuFrameReservation {
    reservation: Option<FrameReservation>,
}

#[napi]
pub struct NativeGpuFramePool {
    pool: Arc<GpuFramePool>,
}

impl NativeGpuFramePool {
    pub fn new(pool: Arc<GpuFramePool>) -> Self {
        Self { pool }
    }
}

#[napi]
impl NativeGpuFramePool {
    #[napi]
    pub fn reserve(&self) -> napi::Result<Option<NativeGpuFrameReservation>> {
        self.pool
            .reserve()
            .map(|reservation| {
                reservation.map(|reservation| NativeGpuFrameReservation {
                    reservation: Some(reservation),
                })
            })
            .map_err(to_napi_error)
    }

    #[napi]
    pub fn render_async(
        &self,
        reservation: &mut NativeGpuFrameReservation,
        scene: RenderScene,
        camera: Camera,
    ) -> napi::Result<AsyncTask<RenderMediaTask>> {
        let reservation = reservation.reservation.take().ok_or_else(|| {
            napi::Error::from_reason("GPU frame reservation has already been consumed".to_owned())
        })?;
        Ok(AsyncTask::new(RenderMediaTask {
            pool: Arc::clone(&self.pool),
            reservation: Some(reservation),
            scene: Some(scene),
            camera: Some(camera),
        }))
    }

    #[napi]
    pub fn stats(&self) -> NativeGpuFramePoolStats {
        self.pool.stats().into()
    }

    #[napi]
    pub fn close(&self) {
        self.pool.close();
    }
}

pub struct RenderMediaTask {
    pool: Arc<GpuFramePool>,
    reservation: Option<FrameReservation>,
    scene: Option<RenderScene>,
    camera: Option<Camera>,
}

impl Task for RenderMediaTask {
    type Output = MediaFrame;
    type JsValue = NativeGpuMediaFrameLease;

    fn compute(&mut self) -> napi::Result<Self::Output> {
        self.pool
            .render_reserved(
                self.reservation.take().expect("frame reservation"),
                self.scene.as_ref().expect("render scene"),
                self.camera.as_ref().expect("render camera"),
            )
            .map_err(to_napi_error)
    }

    fn resolve(&mut self, _env: Env, output: Self::Output) -> napi::Result<Self::JsValue> {
        NativeGpuMediaFrameLease::new(output)
    }
}

pub struct ReadbackTask {
    job: Option<PlaneReadbackJob>,
}

impl Task for ReadbackTask {
    type Output = Vec<PlaneReadback>;
    type JsValue = Vec<NativeGpuPlaneData>;

    fn compute(&mut self) -> napi::Result<Self::Output> {
        self.job
            .take()
            .expect("readback job")
            .complete()
            .map_err(to_napi_error)
    }

    fn resolve(&mut self, _env: Env, output: Self::Output) -> napi::Result<Self::JsValue> {
        Ok(output
            .into_iter()
            .enumerate()
            .map(|(index, plane)| NativeGpuPlaneData {
                index: index as u32,
                format: plane.format.to_owned(),
                width: plane.width,
                height: plane.height,
                bytes_per_row: plane.bytes_per_row,
                data: Buffer::from(plane.data),
            })
            .collect())
    }
}

pub fn parse_pool_options(value: NativeGpuFramePoolOptions) -> napi::Result<GpuFramePoolOptions> {
    Ok(GpuFramePoolOptions {
        width: value.width,
        height: value.height,
        capacity: value.capacity.unwrap_or(3),
        format: MediaOutputFormat::parse(value.format.as_deref().unwrap_or("rgba8unorm"))
            .map_err(to_napi_error)?,
        overflow: OverflowPolicy::parse(value.overflow.as_deref().unwrap_or("error"))
            .map_err(to_napi_error)?,
    })
}

fn to_napi_error(error: anyhow::Error) -> napi::Error {
    napi::Error::from_reason(error.to_string())
}
