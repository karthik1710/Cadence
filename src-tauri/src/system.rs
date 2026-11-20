//! Host-hardware facts, and the one model set every machine runs.
//!
//! Cadence used to tier machines by RAM and pick a smaller LLM on smaller
//! boxes. It no longer does: the minimum supported machine is 16 GB, and every
//! machine gets the same models, sized to fit that minimum. One set means a
//! meeting is summarised by the same model on a 16 GB laptop as on a 32 GB
//! one, and there is exactly one configuration to benchmark and support.
//!
//! RAM is still measured at runtime — not to choose anything, but so a machine
//! below the minimum can be told so plainly (see [`HostInfo::meets_minimum`]).

use serde::{Deserialize, Serialize};

/// The smallest machine Cadence supports, as Windows reports it. Deliberately
/// below 16384: Windows reports *usable* physical memory, and a 16 GB laptop
/// with iGPU-reserved memory commonly shows ~15.7 GB. A threshold of exactly
/// 16 GB would reject the very machines it exists to admit.
pub const MIN_TOTAL_RAM_MB: u64 = 15_000;

/// Catalog id of the one language model (see `models_manager::catalog`).
/// Qwen3-8B at Q4_K_M: ~4.8 GB of weights, multilingual, and the best MoM
/// quality that still fits [`LLM_RAM_BUDGET_MB`] on a 16 GB machine.
pub const LLM_MODEL_ID: &str = "qwen3-8b-q4";

/// Catalog id of the one speech-to-text model.
pub const STT_MODEL_ID: &str = "whisper-small";

/// llama.cpp context window. MoM generation is chunked precisely so it never
/// needs a bigger one, and the KV cache it costs is quantized to q8_0 (see
/// `pipeline::llm::server`), which keeps 8K affordable on 16 GB.
pub const LLM_CONTEXT_TOKENS: usize = 8192;

/// Advisory ceiling on LLM process RAM, in MB, sized for the 16 GB minimum.
/// A 16 GB machine leaves roughly 9-11 GB once Windows, the webview and the
/// page cache holding the meeting WAV are accounted for; 7 GB covers the 8B
/// model's ~6 GB estimate at 8K context with room for compute buffers to grow
/// during generation, and leaves the rest of the machine usable meanwhile.
///
/// Enforced as a pre-flight check in `pipeline::llm`, not as an allocator
/// limit — llama.cpp does not take a memory cap, so the useful thing to do is
/// check before starting and say why.
pub const LLM_RAM_BUDGET_MB: u64 = 7_000;

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

    /// Whether this machine meets the supported minimum. Advisory only: a
    /// machine below it is warned, never blocked from recording — a meeting
    /// captured to disk can always be processed later.
    pub fn meets_minimum(&self) -> bool {
        self.total_ram_mb >= MIN_TOTAL_RAM_MB
    }

    /// The processor's core layout, for choosing thread counts.
    ///
    /// The real topology is used only when it describes *this* `HostInfo` —
    /// same logical-processor count as the machine the process is running
    /// on. A hand-built `HostInfo` (a test, a hypothetical machine) gets
    /// [`CpuTopology::uniform`], so it resolves thread counts exactly as it did
    /// before hybrid CPUs were taken into account, whatever the machine
    /// running the test happens to be. Unit tests never consult the real
    /// topology at all, for the same reason.
    pub fn topology(&self) -> CpuTopology {
        #[cfg(not(test))]
        if let Some(detected) = CpuTopology::detect() {
            if detected.logical == self.logical_cores {
                return detected;
            }
        }
        CpuTopology::uniform(self.logical_cores)
    }
}

/// How the host's logical processors are spread over physical cores, and
/// which of those cores are the fast ones.
///
/// It matters because Intel's hybrid parts (12th gen onwards: P-cores plus
/// E-cores) run ggml's and onnxruntime's thread pools at the pace of the
/// slowest core in them. On an i7-1260P (4 P + 8 E, 16 logical) llama.cpp
/// generation on a ~3.2K-token prompt measured 4.6 tokens/s on 8 threads and
/// 3.0 on 14, and sherpa-onnx diarization took ~31-35 s at 16 threads against
/// ~16-22 s at 2-8. See `settings::default_llm_threads` and
/// docs/benchmarks.md.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CpuTopology {
    pub logical: usize,
    pub physical: usize,
    /// Physical cores of the highest efficiency class: the P-cores on a hybrid
    /// CPU, every core on a conventional one.
    pub performance_physical: usize,
    /// Logical processors on those cores, SMT siblings included.
    pub performance_logical: usize,
}

/// One physical core as Windows reports it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CoreInfo {
    /// Windows' `EfficiencyClass`: higher is faster. Every core reports 0 on a
    /// CPU without heterogeneous cores.
    pub efficiency_class: u8,
    /// Logical processors on this core (2 with SMT, 1 without).
    pub logical: usize,
}

impl CpuTopology {
    /// A CPU with no distinction between cores — what a machine looks like
    /// when nothing better is known. Thread selection treats it exactly as it
    /// treated every machine before hybrid CPUs were considered.
    pub fn uniform(logical: usize) -> Self {
        let logical = logical.max(1);
        Self {
            logical,
            physical: logical,
            performance_physical: logical,
            performance_logical: logical,
        }
    }

    /// Whether some cores are slower than others (P-cores + E-cores).
    pub fn is_hybrid(&self) -> bool {
        self.performance_physical < self.physical
    }

    /// Builds the topology from a per-core list. `None` for an empty list,
    /// which the caller should treat as "unknown", never as zero cores.
    pub fn from_cores(cores: &[CoreInfo]) -> Option<Self> {
        let top_class = cores.iter().map(|c| c.efficiency_class).max()?;
        let fast = cores.iter().filter(|c| c.efficiency_class == top_class);
        let logical: usize = cores.iter().map(|c| c.logical.max(1)).sum();
        Some(Self {
            logical,
            physical: cores.len(),
            performance_physical: fast.clone().count(),
            performance_logical: fast.map(|c| c.logical.max(1)).sum(),
        })
    }

    /// The running machine's topology, read once and cached. `None` when the
    /// query fails or the platform has no such query.
    pub fn detect() -> Option<Self> {
        static DETECTED: std::sync::OnceLock<Option<CpuTopology>> = std::sync::OnceLock::new();
        *DETECTED.get_or_init(|| Self::from_cores(&query_cores()?))
    }
}

/// `RelationProcessorCore` in `LOGICAL_PROCESSOR_RELATIONSHIP`.
const RELATION_PROCESSOR_CORE: u32 = 0;

/// Walks the variable-length `SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX` records
/// `GetLogicalProcessorInformationEx(RelationProcessorCore, ..)` fills in, one
/// per physical core. Kept separate from the FFI call so the layout handling
/// is testable on any platform.
///
/// Layout (x64): `Relationship: u32` at 0, `Size: u32` at 4, then
/// `PROCESSOR_RELATIONSHIP` at 8 — `Flags: u8` at 8, `EfficiencyClass: u8` at
/// 9, 20 reserved bytes, `GroupCount: u16` at 30, and `GroupMask[GroupCount]`
/// of 16-byte `GROUP_AFFINITY` (`Mask: usize` first) from 32. The logical
/// processors on a core are the set bits of its group masks.
pub fn parse_core_records(buffer: &[u8]) -> Option<Vec<CoreInfo>> {
    const HEADER: usize = 8;
    const EFFICIENCY_CLASS: usize = HEADER + 1;
    const GROUP_COUNT: usize = HEADER + 22;
    const GROUP_MASKS: usize = HEADER + 24;
    const GROUP_AFFINITY_SIZE: usize = 16;

    let u32_at = |b: &[u8], at: usize| -> Option<u32> {
        Some(u32::from_le_bytes(b.get(at..at + 4)?.try_into().ok()?))
    };
    let mut cores = Vec::new();
    let mut offset = 0;
    while offset < buffer.len() {
        let record = &buffer[offset..];
        let relationship = u32_at(record, 0)?;
        let size = u32_at(record, 4)? as usize;
        if size < GROUP_MASKS || size > record.len() {
            return None;
        }
        if relationship == RELATION_PROCESSOR_CORE {
            let group_count =
                u16::from_le_bytes(record.get(GROUP_COUNT..GROUP_COUNT + 2)?.try_into().ok()?);
            let mut logical = 0usize;
            for g in 0..group_count as usize {
                let at = GROUP_MASKS + g * GROUP_AFFINITY_SIZE;
                let mask = u64::from_le_bytes(record.get(at..at + 8)?.try_into().ok()?);
                logical += mask.count_ones() as usize;
            }
            cores.push(CoreInfo {
                efficiency_class: record[EFFICIENCY_CLASS],
                logical: logical.max(1),
            });
        }
        offset += size;
    }
    if cores.is_empty() {
        None
    } else {
        Some(cores)
    }
}

#[cfg(windows)]
fn query_cores() -> Option<Vec<CoreInfo>> {
    #[link(name = "kernel32")]
    extern "system" {
        fn GetLogicalProcessorInformationEx(
            relationship: u32,
            buffer: *mut u8,
            returned_length: *mut u32,
        ) -> i32;
    }

    // First call sizes the buffer (it fails with ERROR_INSUFFICIENT_BUFFER and
    // writes the length needed); the second fills it.
    let mut length: u32 = 0;
    // Safe: a null buffer with length 0 is the documented sizing call.
    unsafe {
        GetLogicalProcessorInformationEx(RELATION_PROCESSOR_CORE, std::ptr::null_mut(), &mut length)
    };
    if length == 0 {
        return None;
    }
    // u64 backing store so the records are 8-byte aligned, as the API expects.
    let mut storage = vec![0u64; (length as usize).div_ceil(8)];
    let buffer = storage.as_mut_ptr() as *mut u8;
    // Safe: `buffer` points to at least `length` writable bytes that live
    // until the slice below is done with.
    let ok = unsafe {
        GetLogicalProcessorInformationEx(RELATION_PROCESSOR_CORE, buffer, &mut length)
    } != 0;
    if !ok {
        return None;
    }
    // Safe: the call wrote `length` bytes into `storage`, which holds at
    // least that many.
    let bytes = unsafe { std::slice::from_raw_parts(buffer as *const u8, length as usize) };
    parse_core_records(bytes)
}

#[cfg(not(windows))]
fn query_cores() -> Option<Vec<CoreInfo>> {
    None
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
        // zero here would warn every machine that it is below the minimum.
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

/// Stops a console window from flashing open behind Cadence — a windowed GUI
/// app with no console of its own — every time it shells out to curl, tar,
/// whisper-cli, llama-server, or sherpa-onnx. Call before `spawn`/`output`.
#[cfg(windows)]
pub fn hide_console(command: &mut std::process::Command) {
    use std::os::windows::process::CommandExt;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    command.creation_flags(CREATE_NO_WINDOW);
}

#[cfg(not(windows))]
pub fn hide_console(_command: &mut std::process::Command) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_16gb_machine_meets_the_minimum_and_an_8gb_one_does_not() {
        let host = |total_ram_mb| HostInfo {
            total_ram_mb,
            available_ram_mb: 4_000,
            logical_cores: 8,
        };
        // What a 16 GB laptop with shared iGPU memory actually reports.
        assert!(host(15_700).meets_minimum());
        assert!(host(31_700).meets_minimum());
        assert!(!host(8_100).meets_minimum());
    }

    #[test]
    fn the_llm_budget_stays_within_the_hard_limit_and_the_minimum_machine() {
        assert!(LLM_RAM_BUDGET_MB <= 15_000, "15 GB hard limit");
        // Leave Windows and the app at least half the smallest machine.
        assert!(LLM_RAM_BUDGET_MB <= MIN_TOTAL_RAM_MB / 2);
    }

    #[test]
    fn detects_a_plausible_host() {
        let info = HostInfo::detect();
        assert!(info.total_ram_mb > 0);
        assert!(info.logical_cores >= 1);
    }

    /// One `SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX` core record, laid out the
    /// way Windows writes it on x64.
    fn core_record(efficiency_class: u8, mask: u64) -> Vec<u8> {
        let mut r = vec![0u8; 48];
        r[0..4].copy_from_slice(&RELATION_PROCESSOR_CORE.to_le_bytes());
        r[4..8].copy_from_slice(&48u32.to_le_bytes());
        r[8] = if mask.count_ones() > 1 { 1 } else { 0 }; // LTP_PC_SMT
        r[9] = efficiency_class;
        r[30..32].copy_from_slice(&1u16.to_le_bytes());
        r[32..40].copy_from_slice(&mask.to_le_bytes());
        r
    }

    #[test]
    fn parses_an_i7_1260p_as_four_p_cores_and_eight_e_cores() {
        // 4 SMT P-cores on logical 0-7 (class 1), 8 E-cores on 8-15 (class 0).
        let mut buffer = Vec::new();
        for p in 0..4 {
            buffer.extend(core_record(1, 0b11 << (2 * p)));
        }
        for e in 8..16 {
            buffer.extend(core_record(0, 1 << e));
        }
        let cores = parse_core_records(&buffer).unwrap();
        assert_eq!(cores.len(), 12);
        let cpu = CpuTopology::from_cores(&cores).unwrap();
        assert_eq!(
            cpu,
            CpuTopology {
                logical: 16,
                physical: 12,
                performance_physical: 4,
                performance_logical: 8,
            }
        );
        assert!(cpu.is_hybrid());
    }

    #[test]
    fn a_conventional_cpu_is_not_hybrid() {
        // 8 SMT cores, all efficiency class 0.
        let cores: Vec<CoreInfo> = (0..8)
            .map(|_| CoreInfo { efficiency_class: 0, logical: 2 })
            .collect();
        let cpu = CpuTopology::from_cores(&cores).unwrap();
        assert_eq!(cpu.logical, 16);
        assert_eq!(cpu.physical, 8);
        assert_eq!(cpu.performance_physical, 8);
        assert!(!cpu.is_hybrid());
        assert!(!CpuTopology::uniform(16).is_hybrid());
    }

    #[test]
    fn a_malformed_buffer_is_unknown_not_zero_cores() {
        assert_eq!(parse_core_records(&[]), None);
        // A record claiming to be longer than the buffer.
        let mut truncated = core_record(1, 0b11);
        truncated[4..8].copy_from_slice(&4096u32.to_le_bytes());
        assert_eq!(parse_core_records(&truncated), None);
        assert_eq!(CpuTopology::from_cores(&[]), None);
    }

    #[test]
    fn a_hand_built_host_is_treated_as_uniform() {
        let host = HostInfo {
            total_ram_mb: 16_000,
            available_ram_mb: 8_000,
            logical_cores: 16,
        };
        assert_eq!(host.topology(), CpuTopology::uniform(16));
    }

    #[test]
    fn the_running_machine_topology_is_consistent_when_known() {
        if let Some(cpu) = CpuTopology::detect() {
            assert!(cpu.performance_physical >= 1);
            assert!(cpu.performance_physical <= cpu.physical);
            assert!(cpu.physical <= cpu.logical);
            assert!(cpu.performance_logical <= cpu.logical);
            eprintln!("detected CPU topology: {cpu:?}");
        }
    }
}
