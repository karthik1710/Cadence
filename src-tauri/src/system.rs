//! Host-hardware facts the pipeline has to adapt to, and the memory profile
//! derived from them.
//!
//! Cadence's reference machine has 32 GB of RAM, but the
//! app is expected to run on 16 GB laptops too. RAM is the constraint that
//! actually decides which LLM can be loaded, so it is measured at runtime
//! rather than assumed: see [`MemoryProfile::detect`].

use serde::{Deserialize, Serialize};

/// One measurement of the host's memory and CPU.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct HostInfo {
    pub total_ram_mb: u64,
    pub available_ram_mb: u64,
    pub logical_cores: usize,
}

impl HostInfo {
    pub fn detect() -> Self {
        let (total, available) = physical_memory_mb();
        Self {
            total_ram_mb: total,
            available_ram_mb: available,
            logical_cores: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4),
        }
    }
}

/// Which class of machine we are on, and therefore how much of the memory
/// budget the LLM stage may spend.
///
/// The tiers are deliberately coarse. The hard limit is 15 GB
/// of LLM process RAM with 10 GB preferred, but that budget assumes the
/// 32 GB reference laptop. On a 16 GB machine, Windows plus the app plus the
/// page cache leave roughly 9-11 GB usable, and the LLM must also coexist with
/// whatever the OS has already paged in — so the budget scales with the box
/// instead of being a constant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MemoryProfile {
    /// Under 12 GB total. Smallest viable models, shortest context.
    Low,
    /// 12-24 GB total — the 16 GB laptop case.
    Balanced,
    /// 24 GB and up — the i9-13950HX / 32 GB reference machine.
    High,
}

impl MemoryProfile {
    pub fn detect() -> Self {
        Self::for_total_ram_mb(HostInfo::detect().total_ram_mb)
    }

    pub fn for_total_ram_mb(total_ram_mb: u64) -> Self {
        // Thresholds sit below the nominal sizes (24 GB rather than 32, 12 GB
        // rather than 16) because Windows reports *usable* physical memory:
        // a 16 GB laptop with shared iGPU memory commonly reports ~15.7 GB,
        // and a 32 GB one ~31.7 GB.
        if total_ram_mb >= 24_000 {
            MemoryProfile::High
        } else if total_ram_mb >= 12_000 {
            MemoryProfile::Balanced
        } else {
            MemoryProfile::Low
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            MemoryProfile::Low => "low",
            MemoryProfile::Balanced => "balanced",
            MemoryProfile::High => "high",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "low" => Some(MemoryProfile::Low),
            "balanced" => Some(MemoryProfile::Balanced),
            "high" => Some(MemoryProfile::High),
            _ => None,
        }
    }

    /// Default LLM model id for this tier (see `models_manager::catalog`).
    ///
    /// Qwen3-8B Q4_K_M needs roughly 5 GB of weights plus KV cache; that is
    /// comfortable on 32 GB and uncomfortably close to the ceiling on 16 GB
    /// once Windows, the webview, and the OS file cache holding a 60-minute
    /// WAV are accounted for. The 4B at the same quantization is ~2.5 GB and
    /// leaves real headroom. Neither is a permanent selection — nothing is settled
    /// before benchmarks — and both are user-overridable in Settings.
    pub fn default_llm_model(self) -> &'static str {
        match self {
            MemoryProfile::High => "qwen3-8b-q4",
            MemoryProfile::Balanced => "qwen3-4b-q4",
            MemoryProfile::Low => "qwen3-4b-q4",
        }
    }

    /// Default STT model id for this tier.
    pub fn default_stt_model(self) -> &'static str {
        match self {
            MemoryProfile::High => "whisper-small",
            MemoryProfile::Balanced => "whisper-small",
            // `base` costs accuracy, but `small` plus a 4B LLM plus Windows on
            // an 8 GB machine is how you get the OS swapping mid-meeting.
            MemoryProfile::Low => "whisper-base",
        }
    }

    /// llama.cpp context window. Bigger contexts cost KV-cache RAM roughly
    /// linearly, and MoM generation is chunked precisely so it never
    /// needs a huge one.
    pub fn default_llm_context(self) -> usize {
        match self {
            MemoryProfile::High => 8192,
            MemoryProfile::Balanced => 4096,
            MemoryProfile::Low => 2048,
        }
    }

    /// Advisory ceiling on LLM process RAM, in MB. Enforced as a
    /// pre-flight check in `pipeline::llm`, not as a hard allocator limit —
    /// llama.cpp does not take a memory cap, so the useful thing to do is
    /// refuse to start a model that clearly will not fit, and say why.
    pub fn llm_ram_budget_mb(self) -> u64 {
        match self {
            // The budget prefers <= 10 GB even where 15 GB is allowed.
            MemoryProfile::High => 10_000,
            MemoryProfile::Balanced => 6_000,
            MemoryProfile::Low => 3_500,
        }
    }
}

/// Physical memory in MB as `(total, available)`.
#[cfg(windows)]
fn physical_memory_mb() -> (u64, u64) {
    #[repr(C)]
    struct MemoryStatusEx {
        length: u32,
        memory_load: u32,
        total_phys: u64,
        avail_phys: u64,
        total_page_file: u64,
        avail_page_file: u64,
        total_virtual: u64,
        avail_virtual: u64,
        avail_extended_virtual: u64,
    }

    #[link(name = "kernel32")]
    extern "system" {
        fn GlobalMemoryStatusEx(buffer: *mut MemoryStatusEx) -> i32;
    }

    let mut status = MemoryStatusEx {
        length: std::mem::size_of::<MemoryStatusEx>() as u32,
        memory_load: 0,
        total_phys: 0,
        avail_phys: 0,
        total_page_file: 0,
        avail_page_file: 0,
        total_virtual: 0,
        avail_virtual: 0,
        avail_extended_virtual: 0,
    };

    // Safe: `status` is a correctly sized, correctly initialized
    // MEMORYSTATUSEX and the call only writes into it.
    let ok = unsafe { GlobalMemoryStatusEx(&mut status) } != 0;
    if !ok {
        // Fall back to the reference machine's spec rather than to zero: a
        // zero here would silently demote every machine to the Low profile.
        return (32_768, 16_384);
    }
    (
        status.total_phys / (1024 * 1024),
        status.avail_phys / (1024 * 1024),
    )
}

#[cfg(not(windows))]
fn physical_memory_mb() -> (u64, u64) {
    // Cadence ships on Windows only; this branch exists so the crate's
    // unit tests can run anywhere.
    (32_768, 16_384)
}

/// Peak working set of a spawned child process, in MB — the "peak RAM"
/// measurement for whisper.cpp / sherpa-onnx / llama.cpp. Returns `None` if
/// the query fails.
#[cfg(windows)]
pub fn peak_working_set_mb(child: &std::process::Child) -> Option<u64> {
    use std::os::windows::io::AsRawHandle;

    #[repr(C)]
    #[derive(Default)]
    struct ProcessMemoryCounters {
        cb: u32,
        page_fault_count: u32,
        peak_working_set_size: usize,
        working_set_size: usize,
        quota_peak_paged_pool_usage: usize,
        quota_paged_pool_usage: usize,
        quota_peak_non_paged_pool_usage: usize,
        quota_non_paged_pool_usage: usize,
        pagefile_usage: usize,
        peak_pagefile_usage: usize,
    }

    #[link(name = "kernel32")]
    extern "system" {
        fn K32GetProcessMemoryInfo(
            process: *mut std::ffi::c_void,
            counters: *mut ProcessMemoryCounters,
            cb: u32,
        ) -> i32;
    }

    let mut counters = ProcessMemoryCounters {
        cb: std::mem::size_of::<ProcessMemoryCounters>() as u32,
        ..Default::default()
    };
    let handle = child.as_raw_handle() as *mut std::ffi::c_void;
    // Safe: `handle` is owned by `child` and outlives this call; `counters` is
    // a correctly sized PROCESS_MEMORY_COUNTERS.
    let ok = unsafe { K32GetProcessMemoryInfo(handle, &mut counters, counters.cb) } != 0;
    if !ok {
        return None;
    }
    Some((counters.peak_working_set_size as u64) / (1024 * 1024))
}

#[cfg(not(windows))]
pub fn peak_working_set_mb(_child: &std::process::Child) -> Option<u64> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_ram_sizes_to_profiles() {
        // A 32 GB machine reports a little under 32768 MB.
        assert_eq!(MemoryProfile::for_total_ram_mb(31_700), MemoryProfile::High);
        // A 16 GB laptop, the case this tiering exists for.
        assert_eq!(
            MemoryProfile::for_total_ram_mb(15_700),
            MemoryProfile::Balanced
        );
        assert_eq!(MemoryProfile::for_total_ram_mb(8_100), MemoryProfile::Low);
    }

    #[test]
    fn a_16gb_machine_gets_the_smaller_llm_and_context() {
        let p = MemoryProfile::for_total_ram_mb(15_700);
        assert_eq!(p.default_llm_model(), "qwen3-4b-q4");
        assert_eq!(p.default_llm_context(), 4096);
        assert!(p.llm_ram_budget_mb() < MemoryProfile::High.llm_ram_budget_mb());
    }

    #[test]
    fn every_profile_stays_within_the_15gb_hard_limit() {
        for p in [
            MemoryProfile::Low,
            MemoryProfile::Balanced,
            MemoryProfile::High,
        ] {
            assert!(p.llm_ram_budget_mb() <= 15_000, "15 GB hard limit");
        }
    }

    #[test]
    fn profile_names_round_trip() {
        for p in [
            MemoryProfile::Low,
            MemoryProfile::Balanced,
            MemoryProfile::High,
        ] {
            assert_eq!(MemoryProfile::parse(p.as_str()), Some(p));
        }
        assert_eq!(MemoryProfile::parse("nonsense"), None);
    }

    #[test]
    fn detects_a_plausible_host() {
        let info = HostInfo::detect();
        assert!(info.total_ram_mb > 0);
        assert!(info.logical_cores >= 1);
    }
}
