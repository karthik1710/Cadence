//! Persisted user settings (master prompt §27) and the machine-aware defaults
//! behind them.
//!
//! Everything here is a *preference*: each field is optional, and leaving it
//! unset means "let the machine decide" via [`crate::system::MemoryProfile`].
//! That is what lets the same build behave sensibly on the 32 GB reference
//! laptop and on a 16 GB one without the user configuring anything.
//!
//! Stored at `<app_data>/config/settings.json`.

use crate::system::{HostInfo, MemoryProfile};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    /// Windows input device name; `None` means the system default (§9).
    pub input_device: Option<String>,
    /// "auto" (detect from RAM), or a forced "low" / "balanced" / "high".
    pub profile: String,
    /// Catalog ids. `None` means the profile's default.
    pub stt_model: Option<String>,
    pub llm_model: Option<String>,
    /// `None` means auto-scale to the core count, capped per stage.
    pub stt_threads: Option<usize>,
    pub diarization_threads: Option<usize>,
    pub llm_threads: Option<usize>,
    pub cluster_threshold: f32,
    /// llama.cpp context window; `None` means the profile's default.
    pub llm_context_tokens: Option<usize>,
    /// Output language for generated minutes (§8 — the transcript itself is
    /// never translated, only the MoM).
    pub mom_language: String,
    /// Lets a user turn off the LLM stage entirely — useful on a machine that
    /// cannot host one, and the transcript is still produced.
    pub generate_mom: bool,
    /// §34: off by default, because deeper logging touches meeting content.
    pub debug_logging: bool,
    // --- Audio front-end (§10, §11). Each step is a judgement call about
    // someone else's room and microphone, so each is individually toggleable.
    /// Subtract the microphone's constant electrical offset.
    pub preprocess_remove_dc: bool,
    /// High-pass corner in Hz — removes HVAC rumble, desk thumps and handling
    /// noise from below the bottom of a human voice. `0` disables it.
    pub preprocess_high_pass_hz: f32,
    /// Normalise loudness before transcription. The single biggest accuracy
    /// win on quiet laptop microphones, which is why it defaults to on.
    pub preprocess_normalize: bool,
    pub preprocess_target_rms_dbfs: f32,
    /// Run the optional GTCRN speech denoiser when it is installed. Safe to
    /// leave on: with the model absent the stage logs a skip and does nothing.
    pub denoise_enabled: bool,
    /// Measure how much of the recording is speech (§11's VAD stage). Also
    /// a no-op unless the optional Silero VAD model is installed.
    pub vad_enabled: bool,
    /// Cut the silence out of the audio sent to STT. Off by default — see
    /// `pipeline::vad` for why the timestamp mapping, though exact, is not yet
    /// trusted on real meeting audio.
    pub vad_trim_silence: bool,
    /// Padding kept around each speech segment when trimming, so a cut never
    /// clips a word onset.
    pub vad_speech_pad_ms: u32,
    /// UI presentation only — "light" or "dark". Anything else is treated as
    /// dark by the frontend (fail-safe default). Not parsed/validated here
    /// because, unlike `profile`, it drives no pipeline decision.
    pub theme: String,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            input_device: None,
            profile: "auto".to_string(),
            stt_model: None,
            llm_model: None,
            stt_threads: None,
            diarization_threads: None,
            llm_threads: None,
            cluster_threshold: DEFAULT_CLUSTER_THRESHOLD,
            llm_context_tokens: None,
            mom_language: "English".to_string(),
            generate_mom: true,
            debug_logging: false,
            preprocess_remove_dc: true,
            preprocess_high_pass_hz: DEFAULT_HIGH_PASS_HZ,
            preprocess_normalize: true,
            preprocess_target_rms_dbfs: DEFAULT_TARGET_RMS_DBFS,
            denoise_enabled: true,
            vad_enabled: true,
            vad_trim_silence: false,
            vad_speech_pad_ms: DEFAULT_VAD_PAD_MS,
            theme: "dark".to_string(),
        }
    }
}

/// Provisional. On the one 57s sample available, sherpa-onnx's own default
/// (0.5) shattered a single dominant voice into 8 speaker ids, while 0.9
/// absorbed the second voice into the first. See docs/models.md — this needs
/// re-testing against labelled 3/5/10-speaker recordings (Phase 3).
pub const DEFAULT_CLUSTER_THRESHOLD: f32 = 0.8;

/// A human voice bottoms out around 85 Hz, so everything below this corner is
/// room noise: air conditioning, projector fans, desk thumps, mic handling.
pub const DEFAULT_HIGH_PASS_HZ: f32 = 80.0;

/// Whisper was trained on normalised audio, and -20 dBFS RMS is the level
/// broadcast speech sits at. Peak limiting keeps the gain from ever clipping.
pub const DEFAULT_TARGET_RMS_DBFS: f32 = -20.0;

/// Silence kept around each speech segment when trimming, so a cut never
/// clips the start of a word.
pub const DEFAULT_VAD_PAD_MS: u32 = 200;

/// whisper.cpp keeps gaining out to at least 24 threads on the i9-13950HX;
/// sherpa-onnx peaks around 8 and gets slower past it. See docs/performance.md.
pub const MAX_STT_THREADS: usize = 24;
pub const MAX_DIARIZATION_THREADS: usize = 8;
/// llama.cpp on CPU stops scaling around the physical-core count; past that,
/// hyperthread contention costs more than it gains.
pub const MAX_LLM_THREADS: usize = 16;

impl Settings {
    /// The profile actually in force: the user's override if they set one,
    /// otherwise whatever the machine's RAM implies.
    pub fn effective_profile(&self) -> MemoryProfile {
        MemoryProfile::parse(&self.profile).unwrap_or_else(MemoryProfile::detect)
    }

    pub fn effective_stt_model(&self) -> String {
        self.stt_model
            .clone()
            .unwrap_or_else(|| self.effective_profile().default_stt_model().to_string())
    }

    pub fn effective_llm_model(&self) -> String {
        self.llm_model
            .clone()
            .unwrap_or_else(|| self.effective_profile().default_llm_model().to_string())
    }

    pub fn effective_llm_context(&self) -> usize {
        self.llm_context_tokens
            .unwrap_or_else(|| self.effective_profile().default_llm_context())
            .clamp(1024, 32_768)
    }

    pub fn effective_stt_threads(&self, host: &HostInfo) -> usize {
        self.stt_threads
            .unwrap_or_else(|| host.logical_cores.min(MAX_STT_THREADS))
            .max(1)
    }

    pub fn effective_diarization_threads(&self, host: &HostInfo) -> usize {
        self.diarization_threads
            .unwrap_or_else(|| host.logical_cores.min(MAX_DIARIZATION_THREADS))
            .max(1)
    }

    pub fn effective_llm_threads(&self, host: &HostInfo) -> usize {
        self.llm_threads
            .unwrap_or_else(|| {
                // Leave a couple of threads for the UI and the OS on small
                // machines, where the core count is small enough to notice.
                let cores = host.logical_cores;
                if cores <= 4 {
                    cores.max(1)
                } else {
                    (cores - 2).min(MAX_LLM_THREADS)
                }
            })
            .max(1)
    }
}

fn settings_path(data_dir: &Path) -> PathBuf {
    data_dir.join("config").join("settings.json")
}

/// Reads settings, falling back to defaults for a missing or corrupt file —
/// a broken settings file must never stop someone recording a meeting (§33).
pub fn load(data_dir: &Path) -> Settings {
    match std::fs::read_to_string(settings_path(data_dir)) {
        Ok(raw) => serde_json::from_str(&raw).unwrap_or_else(|e| {
            eprintln!("[settings] settings.json is unreadable ({e}); using defaults");
            Settings::default()
        }),
        Err(_) => Settings::default(),
    }
}

pub fn save(data_dir: &Path, settings: &Settings) -> Result<(), String> {
    let path = settings_path(data_dir);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let json = serde_json::to_string_pretty(settings).map_err(|e| e.to_string())?;
    std::fs::write(&path, json).map_err(|e| format!("could not write {}: {e}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn host(cores: usize) -> HostInfo {
        HostInfo {
            total_ram_mb: 16_000,
            available_ram_mb: 9_000,
            logical_cores: cores,
        }
    }

    #[test]
    fn defaults_defer_to_the_machine() {
        let s = Settings::default();
        assert_eq!(s.profile, "auto");
        assert!(s.stt_model.is_none());
        assert!(s.llm_model.is_none());
    }

    #[test]
    fn an_explicit_profile_overrides_detection() {
        let s = Settings {
            profile: "low".to_string(),
            ..Settings::default()
        };
        assert_eq!(s.effective_profile(), MemoryProfile::Low);
        assert_eq!(s.effective_stt_model(), "whisper-base");
    }

    #[test]
    fn explicit_model_choices_win_over_the_profile() {
        let s = Settings {
            profile: "low".to_string(),
            stt_model: Some("whisper-medium".to_string()),
            llm_model: Some("qwen3-8b-q4".to_string()),
            ..Settings::default()
        };
        assert_eq!(s.effective_stt_model(), "whisper-medium");
        assert_eq!(s.effective_llm_model(), "qwen3-8b-q4");
    }

    #[test]
    fn thread_counts_scale_with_cores_and_stay_capped() {
        let s = Settings::default();
        assert_eq!(s.effective_stt_threads(&host(32)), MAX_STT_THREADS);
        assert_eq!(s.effective_diarization_threads(&host(32)), MAX_DIARIZATION_THREADS);
        // A modest 16 GB laptop: 8 logical cores.
        assert_eq!(s.effective_stt_threads(&host(8)), 8);
        assert_eq!(s.effective_llm_threads(&host(8)), 6);
        // And a very small machine still gets at least one thread per stage.
        assert_eq!(s.effective_llm_threads(&host(2)), 2);
        assert_eq!(s.effective_stt_threads(&host(1)), 1);
    }

    #[test]
    fn context_is_clamped_to_something_a_local_model_can_serve() {
        let s = Settings {
            llm_context_tokens: Some(10),
            ..Settings::default()
        };
        assert_eq!(s.effective_llm_context(), 1024);
    }

    #[test]
    fn theme_defaults_to_dark() {
        let s = Settings::default();
        assert_eq!(s.theme, "dark");
    }

    #[test]
    fn theme_round_trips_through_disk() {
        let dir = std::env::temp_dir().join("cadence-settings-theme-test");
        let _ = std::fs::remove_dir_all(&dir);
        let mut s = Settings::default();
        s.theme = "light".to_string();
        save(&dir, &s).unwrap();
        let loaded = load(&dir);
        assert_eq!(loaded.theme, "light");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn round_trips_through_disk() {
        let dir = std::env::temp_dir().join("cadence-settings-test");
        let _ = std::fs::remove_dir_all(&dir);
        let mut s = Settings::default();
        s.mom_language = "Tamil".to_string();
        s.llm_threads = Some(4);
        save(&dir, &s).unwrap();
        let loaded = load(&dir);
        assert_eq!(loaded.mom_language, "Tamil");
        assert_eq!(loaded.llm_threads, Some(4));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_audio_front_end_defaults_to_the_safe_choice() {
        let s = Settings::default();
        assert!(s.preprocess_remove_dc);
        assert!(s.preprocess_normalize);
        assert_eq!(s.preprocess_high_pass_hz, DEFAULT_HIGH_PASS_HZ);
        assert_eq!(s.preprocess_target_rms_dbfs, DEFAULT_TARGET_RMS_DBFS);
        // Both optional models no-op when absent, so the flags can default on.
        assert!(s.denoise_enabled);
        assert!(s.vad_enabled);
        // Trimming is the one that can move timestamps, so it stays off.
        assert!(!s.vad_trim_silence);
        assert_eq!(s.vad_speech_pad_ms, DEFAULT_VAD_PAD_MS);
    }

    #[test]
    fn a_settings_file_written_before_the_audio_options_existed_still_loads() {
        let dir = std::env::temp_dir().join("cadence-settings-legacy");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("config")).unwrap();
        std::fs::write(
            dir.join("config").join("settings.json"),
            r#"{"profile":"high","mom_language":"Tamil","generate_mom":false}"#,
        )
        .unwrap();
        let loaded = load(&dir);
        assert_eq!(loaded.profile, "high");
        assert_eq!(loaded.mom_language, "Tamil");
        assert!(!loaded.generate_mom);
        // The new fields fall back to their defaults rather than failing the
        // parse and discarding everything the user had configured.
        assert!(loaded.preprocess_normalize);
        assert_eq!(loaded.preprocess_high_pass_hz, DEFAULT_HIGH_PASS_HZ);
        assert!(!loaded.vad_trim_silence);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn audio_options_round_trip_through_disk() {
        let dir = std::env::temp_dir().join("cadence-settings-audio");
        let _ = std::fs::remove_dir_all(&dir);
        let s = Settings {
            preprocess_high_pass_hz: 0.0,
            preprocess_normalize: false,
            denoise_enabled: false,
            vad_trim_silence: true,
            vad_speech_pad_ms: 350,
            ..Settings::default()
        };
        save(&dir, &s).unwrap();
        let loaded = load(&dir);
        assert_eq!(loaded.preprocess_high_pass_hz, 0.0);
        assert!(!loaded.preprocess_normalize);
        assert!(!loaded.denoise_enabled);
        assert!(loaded.vad_trim_silence);
        assert_eq!(loaded.vad_speech_pad_ms, 350);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_corrupt_settings_file_falls_back_to_defaults() {
        let dir = std::env::temp_dir().join("cadence-settings-corrupt");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("config")).unwrap();
        std::fs::write(dir.join("config").join("settings.json"), "{not json").unwrap();
        assert_eq!(load(&dir).profile, "auto");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
