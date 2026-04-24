// src/governor/mod.rs
//
// Resource Governor
//
// Periodically samples the daemon's own CPU usage (via sysinfo) and exposes
// a simple `should_yield()` predicate.  The embedder worker polls this
// before processing each task.
//
// Design:
//   • A background task refreshes CPU stats every 2 seconds.
//   • `should_yield()` is a non-blocking read of an AtomicBool.
//   • We monitor SYSTEM-WIDE CPU, not just our own process.  If the user
//     starts compiling or gaming, we yield — even if our own usage is low.

use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use sysinfo::{System, RefreshKind, CpuRefreshKind};
use tokio::time::{interval, Duration};
use tracing::debug;

#[derive(Clone)]
pub struct ResourceGovernor {
    should_yield: Arc<AtomicBool>,
}

impl ResourceGovernor {
    /// Spawn the background sampling loop and return a handle.
    pub fn new(cpu_threshold_pct: f32) -> Self {
        let flag = Arc::new(AtomicBool::new(false));
        let flag_c = flag.clone();

        tokio::spawn(async move {
            let mut sys = System::new_with_specifics(
                RefreshKind::new().with_cpu(CpuRefreshKind::everything()),
            );
            let mut tick = interval(Duration::from_secs(2));

            loop {
                tick.tick().await;

                sys.refresh_cpu_all();
                let usage = sys.global_cpu_usage();

                let yield_now = usage >= cpu_threshold_pct;
                flag_c.store(yield_now, Ordering::Relaxed);

                debug!(
                    cpu_usage = usage,
                    threshold = cpu_threshold_pct,
                    yielding  = yield_now,
                    "Governor tick"
                );
            }
        });

        Self { should_yield: flag }
    }

    /// Returns `true` if the embedder should pause and wait.
    #[inline]
    pub fn should_yield(&self) -> bool {
        self.should_yield.load(Ordering::Relaxed)
    }
}
