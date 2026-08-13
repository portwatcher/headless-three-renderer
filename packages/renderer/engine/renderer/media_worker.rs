use std::sync::mpsc::{self, SyncSender, TrySendError};

use anyhow::{Context, Result, anyhow};

type Job = Box<dyn FnOnce() + Send + 'static>;
const MEDIA_WORK_QUEUE_CAPACITY: usize = 64;

/// Runs blocking GPU completion work outside Node's shared libuv worker pool.
///
/// The public frame reservations bound render jobs to the pool capacity. Plane
/// readback permits are separately bounded to one per leased slot, so the
/// renderer-wide channel is also bounded so many independent pools cannot
/// accumulate unbounded scene payloads behind the single safe submitter.
pub(crate) struct MediaWorker {
    sender: SyncSender<Job>,
}

impl MediaWorker {
    pub(crate) fn new() -> Result<Self> {
        let (sender, receiver) = mpsc::sync_channel::<Job>(MEDIA_WORK_QUEUE_CAPACITY);
        std::thread::Builder::new()
            .name("headless-three-media".to_owned())
            .spawn(move || {
                while let Ok(job) = receiver.recv() {
                    job();
                }
            })
            .context("failed to start the GPU media worker")?;
        Ok(Self { sender })
    }

    pub(crate) fn schedule(&self, job: impl FnOnce() + Send + 'static) -> Result<()> {
        match self.sender.try_send(Box::new(job)) {
            Ok(()) => Ok(()),
            Err(TrySendError::Full(_)) => Err(anyhow!(
                "GPU media worker queue is full ({MEDIA_WORK_QUEUE_CAPACITY} pending jobs)"
            )),
            Err(TrySendError::Disconnected(_)) => Err(anyhow!("GPU media worker is unavailable")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renderer_media_queue_is_bounded() {
        let worker = MediaWorker::new().expect("start media worker");
        let (started_tx, started_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        worker
            .schedule(move || {
                started_tx.send(()).expect("signal worker start");
                release_rx.recv().expect("release blocked worker");
            })
            .expect("schedule blocking job");
        started_rx.recv().expect("worker started");
        for _ in 0..MEDIA_WORK_QUEUE_CAPACITY {
            worker.schedule(|| {}).expect("fill bounded queue");
        }
        let error = worker
            .schedule(|| {})
            .expect_err("queue must reject overflow");
        assert!(error.to_string().contains("queue is full"));
        release_tx.send(()).expect("release worker");
    }
}
