//! Persisted user settings.
//!
//! Everything here is a *preference* about how a meeting is processed — never
//! which models process it. Every machine runs the same model set (see
//! [`crate::system`]), so there is no model, profile or context-size choice to
//! store. Thread counts still scale with the machine, because that costs
//! nothing in consistency: the same model on more cores gives the same output
//! sooner.
//!
//! Settings files written before the single model set still load: serde
//! ignores the retired `profile` / `stt_model` / `llm_model` /
//! `llm_context_tokens` keys rather than failing the whole parse.
//!
//! Stored at `<app_data>/config/settings.json`.

use crate::pipeline::{enhance, redact};
use crate::system::{self, CpuTopology, HostInfo};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    /// Windows input device name; `None` means the system default.
    pub input_device: Option<String>,
    /// `None` means auto-scale to the core count, capped per stage.
    pub stt_threads: Option<usize>,
    pub diarization_threads: Option<usize>,
    pub llm_threads: Option<usize>,
    pub cluster_threshold: f32,
    /// How STT is told to handle language. `"auto"`, `"code-switched"`,
    /// or an explicit ISO code — see [`crate::pipeline::stt::SttLanguage`].
    pub stt_language: String,
    /// Output language for generated minutes (the transcript itself is
    /// never translated, only the MoM).
    pub mom_language: String,
    /// Lets a user turn off the LLM stage entirely — useful on a machine that
    /// cannot host one, and the transcript is still produced.
    pub generate_mom: bool,
    /// off by default, because deeper logging touches meeting content.
    pub debug_logging: bool,
    // --- Audio front-end. Each step is a judgement call about
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
    /// Measure how much of the recording is speech (the VAD stage). Also
    /// a no-op unless the optional Silero VAD model is installed.
    pub vad_enabled: bool,
    /// Cut the silence out of the audio sent to STT. Off by default — see
    /// `pipeline::vad` for why the timestamp mapping, though exact, is not yet
    /// trusted on real meeting audio.
    pub vad_trim_silence: bool,
    /// Padding kept around each speech segment when trimming, so a cut never
    /// clips a word onset.
    pub vad_speech_pad_ms: u32,
    /// Even out the distance between the person nearest the microphone and the
    /// one at the far end of the table. On by default: it is O(n), costs
    /// nothing measurable, and lifting the far end above the noise floor is the
    /// cheapest accuracy win left in the front end.
    pub preprocess_compress: bool,
    pub preprocess_compress_threshold_dbfs: f32,
    pub preprocess_compress_ratio: f32,
    pub preprocess_compress_knee_db: f32,
    pub preprocess_compress_attack_ms: f32,
    pub preprocess_compress_release_ms: f32,
    /// Model-free spectral-subtraction denoising: the fallback noise handling.
    /// On by default, but it only ever runs when the GTCRN denoiser does not —
    /// GTCRN is part of the installed model set and supersedes it, and the
    /// pipeline never runs both. So on a complete install this costs nothing;
    /// it is what keeps a room's noise handled when the GTCRN model is missing
    /// or fails mid-run, at the price of one STFT over the recording.
    pub preprocess_noise_reduction: bool,
    pub preprocess_noise_reduction_strength: f32,
    /// "off" | "flag" | "mask" | "redact". `flag` is the only honest default
    /// for a lexicon that cannot tell a quotation from an insult: it
    /// highlights without rewriting anyone's words.
    pub redaction_mode: String,
    /// Also catch obfuscated and misspelled forms. Off by default, because
    /// fuzzy matching is where a profanity filter starts flagging surnames.
    pub redaction_fuzzy: bool,
    /// Work out what kind of meeting this was, and what it was about, from the
    /// transcript. Pure text analysis, no model, so it can default on.
    pub classify_meeting: bool,
    /// UI presentation only — "light" or "dark". Anything else is treated as
    /// dark by the frontend (fail-safe default). Not parsed/validated here
    /// because it drives no pipeline decision.
    pub theme: String,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            input_device: None,
            stt_threads: None,
            diarization_threads: None,
            llm_threads: None,
            cluster_threshold: DEFAULT_CLUSTER_THRESHOLD,
            stt_language: DEFAULT_STT_LANGUAGE.to_string(),
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
            preprocess_compress: true,
            preprocess_compress_threshold_dbfs: DEFAULT_COMPRESS_THRESHOLD_DBFS,
            preprocess_compress_ratio: DEFAULT_COMPRESS_RATIO,
            preprocess_compress_knee_db: DEFAULT_COMPRESS_KNEE_DB,
            preprocess_compress_attack_ms: DEFAULT_COMPRESS_ATTACK_MS,
            preprocess_compress_release_ms: DEFAULT_COMPRESS_RELEASE_MS,
            preprocess_noise_reduction: true,
            preprocess_noise_reduction_strength: DEFAULT_NOISE_REDUCTION_STRENGTH,
            redaction_mode: DEFAULT_REDACTION_MODE.to_string(),
            redaction_fuzzy: false,
            classify_meeting: true,
            theme: "dark".to_string(),
        }
    }
}

/// Provisional, and now known to conflict across the only two samples tested.
/// On a 57s noisy single-dominant-voice sample, sherpa-onnx's own default
/// (0.5) shattered that one voice into 8 speaker ids, so Cadence shipped 0.8
/// instead — which then collapsed a clean 26s two-speaker sample into a
/// single id (0.55 and above all do). 0.5 is back because a full collapse
/// erases per-speaker attribution entirely, the worse failure of the two,
/// and it matches sherpa-onnx's own tuning. See docs/benchmarks.md — this
/// still needs labelled 3/5/10-speaker recordings (Phase 3) to settle for
/// real; until then, the Settings screen lets a user raise it back up if a
/// specific recording over-splits.
pub const DEFAULT_CLUSTER_THRESHOLD: f32 = 0.5;

/// These meetings are English + Tamil in the same sentence,
/// so the whole-file single detection whisper.cpp's `auto` performs is the
/// wrong default *for this app*: it locks to whichever language opens the
/// recording and degrades the other for the rest of it. A user with
/// monolingual meetings can still pick `auto`, which is cheaper.
pub const DEFAULT_STT_LANGUAGE: &str = "code-switched";

/// A human voice bottoms out around 85 Hz, so everything below this corner is
/// room noise: air conditioning, projector fans, desk thumps, mic handling.
pub const DEFAULT_HIGH_PASS_HZ: f32 = 80.0;

/// Whisper was trained on normalised audio, and -20 dBFS RMS is the level
/// broadcast speech sits at. Peak limiting keeps the gain from ever clipping.
pub const DEFAULT_TARGET_RMS_DBFS: f32 = -20.0;

/// Silence kept around each speech segment when trimming, so a cut never
/// clips the start of a word.
pub const DEFAULT_VAD_PAD_MS: u32 = 200;

/// A deliberately gentle compression curve. This is an ASR front end, not a
/// broadcast chain: the goal is to lift the far end of the meeting table above
/// the noise floor, not to flatten the meeting. These mirror
/// `enhance::CompressorOptions::default` and exist so the Settings screen has a
/// named number to reset to.
pub const DEFAULT_COMPRESS_THRESHOLD_DBFS: f32 = -24.0;
pub const DEFAULT_COMPRESS_RATIO: f32 = 2.5;
pub const DEFAULT_COMPRESS_KNEE_DB: f32 = 6.0;
pub const DEFAULT_COMPRESS_ATTACK_MS: f32 = 5.0;
pub const DEFAULT_COMPRESS_RELEASE_MS: f32 = 120.0;

/// Halfway. Spectral subtraction's failure mode is "musical noise" — warbling
/// artefacts a user has no way to attribute to a setting — so the middle of the
/// range is the value that is hardest to regret.
pub const DEFAULT_NOISE_REDUCTION_STRENGTH: f32 = 0.5;

/// Detect and highlight, never rewrite. A lexicon cannot tell someone quoting a
/// complaint from someone making one, so the shipped default shows the finding
/// and leaves the words alone.
pub const DEFAULT_REDACTION_MODE: &str = "flag";

/// whisper.cpp keeps gaining out to at least 24 threads on the i9-13950HX;
/// sherpa-onnx peaks around 8 and gets slower past it. See docs/benchmarks.md.
pub const MAX_STT_THREADS: usize = 24;
pub const MAX_DIARIZATION_THREADS: usize = 8;
/// llama.cpp on CPU stops scaling around the physical-core count; past that,
/// hyperthread contention costs more than it gains.
pub const MAX_LLM_THREADS: usize = 16;

impl Settings {
    /// The STT model's catalog id. Not a setting — every machine runs the same
    /// one — but kept as an accessor so the pipeline reads it in one place.
    pub fn effective_stt_model(&self) -> String {
        system::STT_MODEL_ID.to_string()
    }

    /// The LLM's catalog id; see [`Self::effective_stt_model`].
    pub fn effective_llm_model(&self) -> String {
        system::LLM_MODEL_ID.to_string()
    }

    pub fn effective_llm_context(&self) -> usize {
        system::LLM_CONTEXT_TOKENS
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

    /// The compressor the front end should run, or `None` when the user has
    /// turned the stage off. Returning an `Option` rather than a flag beside an
    /// options struct makes a disabled stage unrepresentable downstream: there
    /// is nothing left to accidentally run.
    ///
    /// The knobs are clamped here rather than in the UI, because a settings
    /// file is a text file someone can edit: a ratio below 1.0 would be an
    /// expander, and a negative knee is not a shape at all.
    pub fn effective_compressor(&self) -> Option<enhance::CompressorOptions> {
        if !self.preprocess_compress {
            return None;
        }
        Some(enhance::CompressorOptions {
            threshold_dbfs: self.preprocess_compress_threshold_dbfs.clamp(-60.0, 0.0),
            ratio: self.preprocess_compress_ratio.clamp(1.0, 20.0),
            knee_db: self.preprocess_compress_knee_db.clamp(0.0, 24.0),
            attack_ms: self.preprocess_compress_attack_ms.clamp(0.1, 500.0),
            release_ms: self.preprocess_compress_release_ms.clamp(1.0, 2_000.0),
            // Not user-exposed: with makeup off the stage quietly costs
            // loudness, and the normalise stage after it would then be
            // correcting a level the user never asked to change.
            auto_makeup: true,
        })
    }

    /// The model-free denoiser's settings, or `None` when it is off. On by
    /// default as the fallback for GTCRN; the pipeline drops it whenever GTCRN
    /// runs — see the field's own comment.
    pub fn effective_noise_reduction(&self) -> Option<enhance::NoiseReductionOptions> {
        if !self.preprocess_noise_reduction {
            return None;
        }
        let strength = self.preprocess_noise_reduction_strength.clamp(0.0, 1.0);
        if strength <= 0.0 {
            // Enabled at zero strength is a no-op that still costs a full STFT
            // over the whole meeting. Treat it as off.
            return None;
        }
        Some(enhance::NoiseReductionOptions { strength })
    }

    /// An unrecognised mode falls back to `flag`, not to `off`: a typo in a
    /// hand-edited settings file should cost the user a decoration they did not
    /// ask for, never silently disable a filter they did.
    pub fn effective_redaction_mode(&self) -> redact::RedactionMode {
        redact::RedactionMode::parse(&self.redaction_mode).unwrap_or(redact::RedactionMode::Flag)
    }

    pub fn effective_llm_threads(&self, host: &HostInfo) -> usize {
        self.llm_threads
            .unwrap_or_else(|| default_llm_threads(&host.topology()))
            .max(1)
    }
}

/// llama.cpp's default thread count for a CPU, before any user override.
///
/// On a hybrid CPU this is the P-cores' logical-processor count (SMT
/// siblings included, E-cores excluded): llama.cpp's threads meet at a
/// barrier every layer, so an E-core in the pool sets the pace for all of
/// them. On an i7-1260P (4 P + 8 E, 16 logical), with a MoM-shaped ~3.2K-token
/// prompt through `llama-server` (Qwen3-4B Q4_K_M, the pipeline's own flags),
/// 8 threads measured 12.3-13.3 tokens/s prefill and 4.6 generation, against
/// 10.8 / 3.0 on 14 — the old `logical - 2` rule's choice — and a noisy
/// 4.8-8.0 / 3.0-4.1 on 4. On the i9-13950HX (8 P + 16 E) this gives 16, the
/// same as the old rule, so the reference machine is unchanged. Floored at 4
/// so a part with only one or two P-cores (i3/i5 U-series) does not prefill an
/// 8K-token prompt on two threads; that floor is unmeasured.
///
/// Every other CPU keeps the rule it always had: leave a couple of threads
/// for the UI and the OS, where the core count is small enough to notice.
pub fn default_llm_threads(cpu: &CpuTopology) -> usize {
    let cores = cpu.logical.max(1);
    if cpu.is_hybrid() {
        return cpu
            .performance_logical
            .max(4)
            .min(cores)
            .min(MAX_LLM_THREADS);
    }
    if cores <= 4 {
        cores
    } else {
        (cores - 2).min(MAX_LLM_THREADS)
    }
}

fn settings_path(data_dir: &Path) -> PathBuf {
    data_dir.join("config").join("settings.json")
}

/// Reads settings, falling back to defaults for a missing or corrupt file —
/// a broken settings file must never stop someone recording a meeting.
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
    fn every_machine_resolves_the_same_models() {
        let s = Settings::default();
        assert_eq!(s.effective_stt_model(), system::STT_MODEL_ID);
        assert_eq!(s.effective_llm_model(), system::LLM_MODEL_ID);
        assert_eq!(s.effective_llm_context(), system::LLM_CONTEXT_TOKENS);
    }

    #[test]
    fn a_settings_file_from_the_memory_profile_era_still_loads() {
        let dir = std::env::temp_dir().join("cadence-settings-legacy-profile");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("config")).unwrap();
        std::fs::write(
            dir.join("config").join("settings.json"),
            r#"{"profile":"high","llm_model":"qwen3-4b-q4","stt_model":null,"llm_context_tokens":4096,"mom_language":"Tamil","llm_threads":6,"theme":"light"}"#,
        )
        .unwrap();
        let loaded = load(&dir);
        // The retired keys are ignored, not an error that discards the file...
        assert_eq!(loaded.mom_language, "Tamil");
        assert_eq!(loaded.llm_threads, Some(6));
        assert_eq!(loaded.theme, "light");
        // ...and cannot steer the pipeline back to a model that is gone.
        assert_eq!(loaded.effective_llm_model(), system::LLM_MODEL_ID);
        assert_eq!(loaded.effective_stt_model(), system::STT_MODEL_ID);
        assert_eq!(loaded.effective_llm_context(), system::LLM_CONTEXT_TOKENS);
        // Saving it again drops them for good.
        save(&dir, &loaded).unwrap();
        let raw = std::fs::read_to_string(dir.join("config").join("settings.json")).unwrap();
        assert!(!raw.contains("profile"));
        assert!(!raw.contains("llm_model"));
        let _ = std::fs::remove_dir_all(&dir);
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

    /// i7-1260P: 4 SMT P-cores + 8 E-cores = 16 logical.
    fn i7_1260p() -> CpuTopology {
        CpuTopology {
            logical: 16,
            physical: 12,
            performance_physical: 4,
            performance_logical: 8,
        }
    }

    /// i9-13950HX, the docs' reference machine: 8 SMT P-cores + 16 E-cores.
    fn i9_13950hx() -> CpuTopology {
        CpuTopology {
            logical: 32,
            physical: 24,
            performance_physical: 8,
            performance_logical: 16,
        }
    }

    #[test]
    fn a_conventional_cpu_keeps_the_old_llm_rule() {
        // Exactly what `logical - 2, capped` gave before hybrids were handled.
        for (cores, expected) in [(1, 1), (2, 2), (4, 4), (8, 6), (16, 14), (32, 16)] {
            assert_eq!(default_llm_threads(&CpuTopology::uniform(cores)), expected);
        }
        // A detected non-hybrid CPU (SMT, every core the same class) too.
        let ryzen = CpuTopology {
            logical: 16,
            physical: 8,
            performance_physical: 8,
            performance_logical: 16,
        };
        assert!(!ryzen.is_hybrid());
        assert_eq!(default_llm_threads(&ryzen), 14);
    }

    #[test]
    fn a_hybrid_cpu_runs_the_llm_on_its_p_cores_only() {
        assert_eq!(default_llm_threads(&i7_1260p()), 8);
        // The reference machine resolves exactly as it did under the old rule.
        assert_eq!(default_llm_threads(&i9_13950hx()), 16);
        assert_eq!(default_llm_threads(&CpuTopology::uniform(32)), 16);
        // One SMT-less P-core is floored rather than prefilling on one thread...
        let lakefield = CpuTopology {
            logical: 5,
            physical: 5,
            performance_physical: 1,
            performance_logical: 1,
        };
        assert_eq!(default_llm_threads(&lakefield), 4);
        // ...but never past the machine itself.
        let tiny = CpuTopology {
            logical: 3,
            physical: 3,
            performance_physical: 1,
            performance_logical: 1,
        };
        assert_eq!(default_llm_threads(&tiny), 3);
    }

    #[test]
    fn a_user_thread_override_beats_the_topology() {
        let s = Settings {
            llm_threads: Some(12),
            ..Settings::default()
        };
        assert_eq!(s.effective_llm_threads(&host(16)), 12);
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
    fn the_compressor_and_the_fallback_denoiser_both_default_on() {
        let s = Settings::default();
        // The compressor is O(n) and lifts the far end of the table. The
        // model-free denoiser is the fallback when GTCRN is missing or fails;
        // the pipeline drops it whenever GTCRN runs, so it never runs twice.
        assert!(s.preprocess_compress);
        assert!(s.preprocess_noise_reduction);
        let compressor = s.effective_compressor().unwrap();
        assert_eq!(compressor.threshold_dbfs, DEFAULT_COMPRESS_THRESHOLD_DBFS);
        assert_eq!(compressor.ratio, DEFAULT_COMPRESS_RATIO);
        assert_eq!(compressor.knee_db, DEFAULT_COMPRESS_KNEE_DB);
        assert_eq!(compressor.attack_ms, DEFAULT_COMPRESS_ATTACK_MS);
        assert_eq!(compressor.release_ms, DEFAULT_COMPRESS_RELEASE_MS);
        assert!(compressor.auto_makeup);
        assert_eq!(
            s.effective_noise_reduction().unwrap().strength,
            DEFAULT_NOISE_REDUCTION_STRENGTH
        );
        // And the shipped curve is exactly enhance's own gentle default.
        assert_eq!(compressor, enhance::CompressorOptions::default());
    }

    #[test]
    fn a_disabled_stage_resolves_to_nothing_to_run() {
        let s = Settings {
            preprocess_compress: false,
            preprocess_noise_reduction: true,
            preprocess_noise_reduction_strength: 0.8,
            ..Settings::default()
        };
        assert!(s.effective_compressor().is_none());
        assert_eq!(s.effective_noise_reduction().unwrap().strength, 0.8);

        // Enabled at zero strength is a no-op that would still cost a full
        // STFT, so it resolves to off rather than to a wasted pass.
        let pointless = Settings {
            preprocess_noise_reduction: true,
            preprocess_noise_reduction_strength: 0.0,
            ..Settings::default()
        };
        assert!(pointless.effective_noise_reduction().is_none());
    }

    #[test]
    fn a_hand_edited_settings_file_cannot_produce_a_nonsense_curve() {
        let s = Settings {
            preprocess_compress_ratio: 0.2,
            preprocess_compress_knee_db: -5.0,
            preprocess_compress_attack_ms: 0.0,
            preprocess_compress_threshold_dbfs: 12.0,
            preprocess_noise_reduction: true,
            preprocess_noise_reduction_strength: 4.0,
            ..Settings::default()
        };
        let compressor = s.effective_compressor().unwrap();
        assert!(compressor.ratio >= 1.0, "a ratio below 1 is an expander");
        assert!(compressor.knee_db >= 0.0);
        assert!(compressor.attack_ms > 0.0);
        assert!(compressor.threshold_dbfs <= 0.0);
        assert_eq!(s.effective_noise_reduction().unwrap().strength, 1.0);
    }

    #[test]
    fn redaction_defaults_to_flagging_and_never_falls_back_to_off() {
        let s = Settings::default();
        assert_eq!(s.redaction_mode, DEFAULT_REDACTION_MODE);
        assert!(!s.redaction_fuzzy);
        assert!(s.classify_meeting);
        assert_eq!(s.effective_redaction_mode(), redact::RedactionMode::Flag);

        for (raw, expected) in [
            ("off", redact::RedactionMode::Off),
            ("MASK", redact::RedactionMode::Mask),
            (" redact ", redact::RedactionMode::Redact),
        ] {
            let s = Settings {
                redaction_mode: raw.to_string(),
                ..Settings::default()
            };
            assert_eq!(s.effective_redaction_mode(), expected, "parsing {raw:?}");
        }

        // A typo must not silently disable the filter the user asked for.
        let typo = Settings {
            redaction_mode: "redct".to_string(),
            ..Settings::default()
        };
        assert_eq!(typo.effective_redaction_mode(), redact::RedactionMode::Flag);
    }

    #[test]
    fn a_settings_file_written_before_the_enhancement_options_existed_still_loads() {
        let dir = std::env::temp_dir().join("cadence-settings-legacy-enhance");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("config")).unwrap();
        std::fs::write(
            dir.join("config").join("settings.json"),
            r#"{"profile":"high","stt_language":"ta","vad_speech_pad_ms":350,"theme":"light"}"#,
        )
        .unwrap();
        let loaded = load(&dir);
        // What the user had configured survives.
        assert_eq!(loaded.stt_language, "ta");
        assert_eq!(loaded.vad_speech_pad_ms, 350);
        assert_eq!(loaded.theme, "light");
        // And the fields that did not exist when it was written take their
        // defaults rather than failing the parse and discarding all of it.
        assert!(loaded.preprocess_compress);
        assert_eq!(
            loaded.preprocess_compress_threshold_dbfs,
            DEFAULT_COMPRESS_THRESHOLD_DBFS
        );
        assert_eq!(loaded.preprocess_compress_ratio, DEFAULT_COMPRESS_RATIO);
        assert!(loaded.preprocess_noise_reduction);
        assert_eq!(
            loaded.preprocess_noise_reduction_strength,
            DEFAULT_NOISE_REDUCTION_STRENGTH
        );
        assert_eq!(loaded.redaction_mode, DEFAULT_REDACTION_MODE);
        assert!(!loaded.redaction_fuzzy);
        assert!(loaded.classify_meeting);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_enhancement_options_round_trip_through_disk() {
        let dir = std::env::temp_dir().join("cadence-settings-enhance");
        let _ = std::fs::remove_dir_all(&dir);
        let s = Settings {
            preprocess_compress: false,
            preprocess_compress_ratio: 4.0,
            preprocess_noise_reduction: true,
            preprocess_noise_reduction_strength: 0.75,
            redaction_mode: "mask".to_string(),
            redaction_fuzzy: true,
            classify_meeting: false,
            ..Settings::default()
        };
        save(&dir, &s).unwrap();
        let loaded = load(&dir);
        assert!(!loaded.preprocess_compress);
        assert_eq!(loaded.preprocess_compress_ratio, 4.0);
        assert!(loaded.preprocess_noise_reduction);
        assert_eq!(loaded.preprocess_noise_reduction_strength, 0.75);
        assert_eq!(
            loaded.effective_redaction_mode(),
            redact::RedactionMode::Mask
        );
        assert!(loaded.redaction_fuzzy);
        assert!(!loaded.classify_meeting);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn stt_language_defaults_to_code_switched() {
        // this app's meetings mix English and Tamil, so whisper's
        // one-detection-per-file `auto` is the wrong default here.
        assert_eq!(Settings::default().stt_language, "code-switched");
        assert_eq!(DEFAULT_STT_LANGUAGE, "code-switched");
    }

    #[test]
    fn a_settings_file_written_before_stt_language_existed_still_loads() {
        let dir = std::env::temp_dir().join("cadence-settings-stt-language");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("config")).unwrap();
        std::fs::write(
            dir.join("config").join("settings.json"),
            r#"{"profile":"high","mom_language":"Tamil","cluster_threshold":0.7}"#,
        )
        .unwrap();
        let loaded = load(&dir);
        assert_eq!(loaded.mom_language, "Tamil");
        assert_eq!(loaded.cluster_threshold, 0.7);
        assert_eq!(loaded.stt_language, DEFAULT_STT_LANGUAGE);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_forced_stt_language_round_trips_through_disk() {
        let dir = std::env::temp_dir().join("cadence-settings-forced-language");
        let _ = std::fs::remove_dir_all(&dir);
        let s = Settings {
            stt_language: "ta".to_string(),
            ..Settings::default()
        };
        save(&dir, &s).unwrap();
        assert_eq!(load(&dir).stt_language, "ta");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_corrupt_settings_file_falls_back_to_defaults() {
        let dir = std::env::temp_dir().join("cadence-settings-corrupt");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("config")).unwrap();
        std::fs::write(dir.join("config").join("settings.json"), "{not json").unwrap();
        assert_eq!(load(&dir).theme, "dark");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
