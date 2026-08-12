use std::sync::Arc;

use super::*;

#[test]
fn dropping_a_frame_with_pending_external_use_retires_its_slot() {
    let renderer = match GpuRenderer::new() {
        Ok(renderer) => Arc::new(renderer),
        Err(error) => {
            eprintln!("skipping GPU pool retirement test: {error}");
            return;
        }
    };
    if !renderer.gpu_output_capabilities().texture_supported {
        return;
    }
    let pool = GpuFramePool::new(
        renderer,
        GpuFramePoolOptions {
            width: 2,
            height: 2,
            capacity: 1,
            format: MediaOutputFormat::Rgba8,
            overflow: OverflowPolicy::Error,
        },
    )
    .expect("create one-slot GPU pool");
    let mut frame = pool
        .render(&RenderScene::default(), &Camera::default())
        .expect("render frame")
        .expect("error overflow never drops");
    assert_ne!(frame.plane_handle(0).expect("borrow plane handle"), 0);

    drop(frame);

    let stats = pool.stats();
    assert_eq!(stats.retired, 1);
    assert_eq!(stats.available, 0);
    assert_eq!(stats.in_flight, 0);
    let error = pool
        .render(&RenderScene::default(), &Camera::default())
        .err()
        .expect("retired slot must never be reacquired");
    assert!(error.to_string().contains("pool is exhausted"));
}
