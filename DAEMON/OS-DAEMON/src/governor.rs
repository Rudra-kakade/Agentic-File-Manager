// src/governor.rs
//
// Resource governor — throttles embedding work when CPU is over the threshold.

use sysinfo::System;
use tracing::trace;

pub struct ResourceGovernor {
    threshold: f32,
}

impl ResourceGovernor {
    pub fn new(threshold_pct: f32) -> Self {
        Self { threshold: threshold_pct }
    }

    /// Returns `true` if the system is too busy and callers should yield.
    pub fn should_yield(&self) -> bool {
        let mut sys = System::new();
        sys.refresh_cpu_usage();
        // sysinfo needs a short delay between refreshes for accurate readings
        std::thread::sleep(std::time::Duration::from_millis(200));
        sys.refresh_cpu_usage();
        let cpu = sys.global_cpu_info().cpu_usage();
        let over = cpu > self.threshold;
        trace!(cpu = cpu, threshold = self.threshold, over, "Governor check");
        over
    }
}
