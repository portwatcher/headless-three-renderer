use std::sync::mpsc;

use anyhow::{Context, Result};

use super::{MediaFrame, PlaneReadback};
use crate::util::align_to;

pub struct PlaneReadbackJob {
    device: wgpu::Device,
    pending: Vec<PendingPlane>,
}

struct PendingPlane {
    buffer: wgpu::Buffer,
    width: u32,
    height: u32,
    bytes_per_row: u32,
    padded_bytes_per_row: u32,
    format: &'static str,
}

impl MediaFrame {
    /// Enqueues a diagnostic plane readback before returning. Queue ordering
    /// makes it safe for the caller to release the lease immediately after
    /// this method returns; only the wait and buffer mapping happen later.
    pub fn begin_plane_readback(&mut self) -> Result<PlaneReadbackJob> {
        self.plane_texture(0)?;
        if self.external_use_pending {
            anyhow::bail!("cannot read back planes while external GPU use is pending")
        }
        let device = self.renderer.device.clone();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("headless-three media plane readback encoder"),
        });
        let mut pending = Vec::with_capacity(self.plane_count());
        for index in 0..self.plane_count() {
            let texture = self.plane_texture(index)?;
            let (width, height) = self.plane_dimensions(index)?;
            let bytes_per_row = self.plane_bytes_per_row(index)?;
            let padded_bytes_per_row = align_to(bytes_per_row, wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("headless-three media plane readback buffer"),
                size: padded_bytes_per_row as u64 * height as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &buffer,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(padded_bytes_per_row),
                        rows_per_image: Some(height),
                    },
                },
                wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
            );
            pending.push(PendingPlane {
                buffer,
                width,
                height,
                bytes_per_row,
                padded_bytes_per_row,
                format: self.plane_format(index)?,
            });
        }
        self.renderer.queue.submit([encoder.finish()]);
        self.readback_state = true;
        Ok(PlaneReadbackJob { device, pending })
    }
}

impl PlaneReadbackJob {
    pub fn complete(self) -> Result<Vec<PlaneReadback>> {
        let mut receivers = Vec::with_capacity(self.pending.len());
        for plane in &self.pending {
            let (sender, receiver) = mpsc::channel();
            plane
                .buffer
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |result| {
                    let _ = sender.send(result);
                });
            receivers.push(receiver);
        }
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .context("failed while waiting for media plane readback")?;
        let mut output = Vec::with_capacity(self.pending.len());
        for (plane, receiver) in self.pending.into_iter().zip(receivers) {
            receiver
                .recv()
                .context("media plane readback callback was not delivered")?
                .context("failed to map media plane readback buffer")?;
            let mapped = plane.buffer.slice(..).get_mapped_range();
            let mut data = vec![0; plane.bytes_per_row as usize * plane.height as usize];
            for row in 0..plane.height as usize {
                let source = row * plane.padded_bytes_per_row as usize;
                let target = row * plane.bytes_per_row as usize;
                data[target..target + plane.bytes_per_row as usize]
                    .copy_from_slice(&mapped[source..source + plane.bytes_per_row as usize]);
            }
            drop(mapped);
            plane.buffer.unmap();
            output.push(PlaneReadback {
                data,
                width: plane.width,
                height: plane.height,
                bytes_per_row: plane.bytes_per_row,
                format: plane.format,
            });
        }
        Ok(output)
    }
}
