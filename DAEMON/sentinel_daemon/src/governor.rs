// src/governor.rs
//
// Resource governor — throttles embedding work when CPU is over the threshold.

use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
use std::time::Duration;
use sysinfo::{System, RefreshKind, CpuRefreshKind};
use tokio::time::interval;
use tracing::{debug, info};

pub struct Governor {
    threshold: f32,
    should_yield: Arc<AtomicBool>,
}

impl Governor {
    pub fn new(threshold_pct: f32) -> Self {
        Self {
            threshold: threshold_pct,
            should_yield: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Spawn the background sampling loop.
    pub fn start(&self) {
        let flag = self.should_yield.clone();
        let threshold = self.threshold;

        tokio::spawn(async move {
            let mut sys = System::new_with_specifics(
                RefreshKind::new().with_cpu(CpuRefreshKind::everything()),
            );
            let mut tick = interval(Duration::from_secs(2));

            info!(threshold, "Resource Governor started");

            loop {
                tick.tick().await;

                sys.refresh_cpu_usage();
                let usage = sys.global_cpu_info().cpu_usage();

                let yield_now = usage >= threshold;
                flag.store(yield_now, Ordering::Relaxed);

                debug!(
                    cpu_usage = usage,
                    threshold,
                    yielding = yield_now,
                    "Governor check"
                );
            }
        });
    }

    /// Returns a shared flag that embedding workers can check.
    pub fn pause_flag(&self) -> Arc<AtomicBool> {
        self.should_yield.clone()
    }

    /// Synchronous check (optional, for compatibility).
    pub fn should_yield(&self) -> bool {
        self.should_yield.load(Ordering::Relaxed)
    }
}

