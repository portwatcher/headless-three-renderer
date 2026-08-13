use std::sync::Arc;

use napi::Env;
use napi::bindgen_prelude::{BigInt, Buffer, JsValue, Object};
use napi_derive::napi;

use crate::renderer::{
    FrameReservation, GpuFramePool, GpuFramePoolOptions, GpuFramePoolStats, MediaFrame,
    MediaOutputFormat, OverflowPolicy, PlaneReadback,
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

#[napi(object)]
pub struct NativeCpuI420Frame {
    pub width: u32,
    pub height: u32,
    pub data: Buffer,
    pub format: String,
    pub color_matrix: String,
    pub color_range: String,
    pub chroma_siting: String,
    pub strides: Vec<u32>,
    pub offsets: Vec<u32>,
    pub byte_length: u32,
    pub gpu_readback_bytes: u32,
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

    #[napi(ts_return_type = "Promise<NativeGpuPlaneData[]>")]
    pub fn read_planes(&mut self, env: Env) -> napi::Result<Object<'static>> {
        let frame = self.frame.as_mut().ok_or_else(|| {
            napi::Error::from_reason("GPU media frame lease has been released".to_owned())
        })?;
        let job = frame.begin_plane_readback().map_err(to_napi_error)?;
        let (deferred, promise) = env.create_deferred()?;
        frame
            .worker()
            .schedule(move || {
                let result = job
                    .complete()
                    .map(convert_plane_readbacks)
                    .map_err(to_napi_error);
                deferred.resolve(move |_env| result);
            })
            .map_err(to_napi_error)?;
        Ok(Object::from_raw(env.raw(), promise.raw()))
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
impl NativeGpuFrameReservation {
    #[napi]
    pub fn cancel(&mut self) {
        self.reservation.take();
    }
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

    #[napi(ts_return_type = "Promise<NativeGpuMediaFrameLease>")]
    pub fn render_async(
        &self,
        env: Env,
        reservation: &mut NativeGpuFrameReservation,
        scene: RenderScene,
        camera: Camera,
    ) -> napi::Result<Object<'static>> {
        let reservation = reservation.reservation.take().ok_or_else(|| {
            napi::Error::from_reason("GPU frame reservation has already been consumed".to_owned())
        })?;
        let (deferred, promise) = env.create_deferred()?;
        let pool = Arc::clone(&self.pool);
        self.pool
            .worker()
            .schedule(move || {
                let result = pool
                    .render_reserved(reservation, &scene, &camera)
                    .map_err(to_napi_error)
                    .and_then(NativeGpuMediaFrameLease::new);
                deferred.resolve(move |_env| result);
            })
            .map_err(to_napi_error)?;
        Ok(Object::from_raw(env.raw(), promise.raw()))
    }

    #[napi(ts_return_type = "Promise<NativeCpuI420Frame>")]
    pub fn render_i420_async(
        &self,
        env: Env,
        reservation: &mut NativeGpuFrameReservation,
        scene: RenderScene,
        camera: Camera,
        target: Option<Buffer>,
    ) -> napi::Result<Object<'static>> {
        let reservation = reservation.reservation.take().ok_or_else(|| {
            napi::Error::from_reason("GPU frame reservation has already been consumed".to_owned())
        })?;
        let layout = self.pool.i420_layout().map_err(to_napi_error)?;
        let mut data = target.unwrap_or_else(|| Buffer::from(vec![0; layout.byte_length]));
        if data.len() != layout.byte_length {
            return Err(napi::Error::from_reason(format!(
                "packed I420 target must contain exactly {} bytes, received {}",
                layout.byte_length,
                data.len()
            )));
        }
        let width = self.pool.options().width;
        let height = self.pool.options().height;
        let (deferred, promise) = env.create_deferred()?;
        let pool = Arc::clone(&self.pool);
        self.pool
            .worker()
            .schedule(move || {
                let result = pool
                    .render_i420_reserved(reservation, &scene, &camera, &mut data)
                    .map(|layout| NativeCpuI420Frame {
                        width,
                        height,
                        data,
                        format: "I420".to_owned(),
                        color_matrix: "bt601".to_owned(),
                        color_range: "limited".to_owned(),
                        chroma_siting: "centered-2x2-box".to_owned(),
                        strides: layout.strides.to_vec(),
                        offsets: layout.offsets.to_vec(),
                        byte_length: layout.byte_length as u32,
                        gpu_readback_bytes: layout.gpu_readback_bytes as u32,
                    })
                    .map_err(to_napi_error);
                deferred.resolve(move |_env| result);
            })
            .map_err(to_napi_error)?;
        Ok(Object::from_raw(env.raw(), promise.raw()))
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

fn convert_plane_readbacks(output: Vec<PlaneReadback>) -> Vec<NativeGpuPlaneData> {
    output
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
        .collect()
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
