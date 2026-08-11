use super::*;

impl GpuRenderer {
    pub fn new() -> Result<Self> {
        pollster::block_on(Self::new_async())
    }
}
