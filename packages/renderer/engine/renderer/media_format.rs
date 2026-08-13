use anyhow::{Result, bail};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MediaOutputFormat {
    Rgba8,
    Nv12,
    P010,
    I420,
}

impl MediaOutputFormat {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "rgba8" | "rgba8unorm" => Ok(Self::Rgba8),
            "nv12-planes" => Ok(Self::Nv12),
            "p010-planes" => Ok(Self::P010),
            "i420-planes" => Ok(Self::I420),
            _ => bail!("unsupported GPU media output format '{value}'"),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Rgba8 => "rgba8unorm",
            Self::Nv12 => "nv12-planes",
            Self::P010 => "p010-planes",
            Self::I420 => "i420-planes",
        }
    }

    pub(super) fn plane_formats(self) -> &'static [wgpu::TextureFormat] {
        match self {
            Self::Rgba8 => &[wgpu::TextureFormat::Rgba8Unorm],
            Self::Nv12 => &[wgpu::TextureFormat::R8Unorm, wgpu::TextureFormat::Rg8Unorm],
            Self::P010 => &[
                wgpu::TextureFormat::R16Unorm,
                wgpu::TextureFormat::Rg16Unorm,
            ],
            Self::I420 => &[
                wgpu::TextureFormat::R8Unorm,
                wgpu::TextureFormat::R8Unorm,
                wgpu::TextureFormat::R8Unorm,
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
