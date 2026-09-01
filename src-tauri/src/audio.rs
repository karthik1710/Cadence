use crate::pipeline::audio_profile::{self, amplitude_to_dbfs, AudioProfile, SILENCE_DBFS};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::SampleFormat;
use serde::Serialize;
use std::fs::File;
use std::io::BufWriter;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;
use tauri::{AppHandle, Manager, State};

/// A sample this close to full scale was limited by the ADC.
const CLIP_THRESHOLD: f32 = 0.999;

/// Default length of the pre-meeting microphone check.
const MIC_TEST_SECONDS: u32 = 3;
const MAX_MIC_TEST_SECONDS: u32 = 30;

#[derive(Serialize, Clone)]
pub struct AudioDeviceInfo {
    pub name: String,
    pub is_default: bool,
}

/// Live recording quality, sampled by the UI while a meeting is being
/// recorded — the point being that a level problem is worth knowing about
/// *during* the meeting, when it can still be fixed.
#[derive(Serialize, Clone, Debug, PartialEq)]
pub struct MicMeter {
    /// The legacy 0.0..1.0 linear peak, kept so `get_mic_level` and anything
    /// drawing a simple bar keep working unchanged.
    pub peak: f32,
    pub peak_dbfs: f32,
    pub rms_dbfs: f32,
    /// Loudest sample seen since the recording started.
    pub session_peak_dbfs: f32,
    pub noise_floor_dbfs: f32,
    pub estimated_snr_db: f32,
    pub clipped_samples: u64,
    pub clipped_percent: f32,
    pub recording: bool,
}

impl Default for MicMeter {
    fn default() -> Self {
        Self {
            peak: 0.0,
            peak_dbfs: SILENCE_DBFS,
            rms_dbfs: SILENCE_DBFS,
            session_peak_dbfs: SILENCE_DBFS,
            noise_floor_dbfs: SILENCE_DBFS,
            estimated_snr_db: 0.0,
            clipped_samples: 0,
            clipped_percent: 0.0,
            recording: false,
        }
    }
}

/// How far above the current floor a block may sit and still count as room
/// tone rather than speech (~12 dB). Speech runs far louder than this, so it
/// can never pull the floor up with it.
const NOISE_FLOOR_RISE_GATE: f32 = 4.0;

/// Lock-free recording meter shared between the audio callback and the UI.
///
/// The callback is a real-time thread: it must not allocate, lock, or touch
/// the filesystem, so every field here is an atomic and every value is a plain
/// `f32` bit pattern. There is exactly one writer (the callback), which is why
/// the read-modify-write of the noise floor needs no compare-and-swap.
#[derive(Debug)]
pub struct LevelMeter {
    peak_bits: AtomicU32,
    rms_bits: AtomicU32,
    session_peak_bits: AtomicU32,
    /// `NaN` until the first block arrives, so the floor starts at the real
    /// room level instead of decaying towards it from an invented one.
    noise_floor_bits: AtomicU32,
    clipped: AtomicU64,
    observed: AtomicU64,
}

impl Default for LevelMeter {
    fn default() -> Self {
        Self {
            peak_bits: AtomicU32::new(0),
            rms_bits: AtomicU32::new(0),
            session_peak_bits: AtomicU32::new(0),
            noise_floor_bits: AtomicU32::new(f32::NAN.to_bits()),
            clipped: AtomicU64::new(0),
            observed: AtomicU64::new(0),
        }
    }
}

impl LevelMeter {
    /// Called from the audio callback, once per block. Pure arithmetic over an
    /// iterator: no allocation, no locking, no I/O.
    pub fn observe<I: Iterator<Item = f32>>(&self, samples: I) {
        let mut peak = 0f32;
        let mut sum_sq = 0f64;
        let mut count = 0u64;
        let mut clipped = 0u64;
        for sample in samples {
            let magnitude = sample.abs();
            if magnitude > peak {
                peak = magnitude;
            }
            sum_sq += (sample as f64) * (sample as f64);
            count += 1;
            if magnitude >= CLIP_THRESHOLD {
                clipped += 1;
            }
        }
        if count == 0 {
            return;
        }

        let rms = (sum_sq / count as f64).sqrt() as f32;
        self.peak_bits.store(peak.to_bits(), Ordering::Relaxed);
        self.rms_bits.store(rms.to_bits(), Ordering::Relaxed);
        let session_peak = f32::from_bits(self.session_peak_bits.load(Ordering::Relaxed));
        if peak > session_peak {
            self.session_peak_bits
                .store(peak.to_bits(), Ordering::Relaxed);
        }
        self.clipped.fetch_add(clipped, Ordering::Relaxed);
        self.observed.fetch_add(count, Ordering::Relaxed);

        // Asymmetric one-pole tracker: it follows a drop in level within a
        // few blocks (that is the room going quiet, which is exactly what the
        // floor should read) and rises over minutes, so even an uninterrupted
        // monologue never drags the floor up to speech level and hides a
        // genuinely noisy room.
        let floor = f32::from_bits(self.noise_floor_bits.load(Ordering::Relaxed));
        let next = if !floor.is_finite() {
            rms
        } else if rms < floor {
            // Falls fast: a drop in level IS the room going quiet, which is
            // exactly what the floor should read.
            floor * 0.9 + rms * 0.1
        } else if rms < floor * NOISE_FLOOR_RISE_GATE {
            // Rises only while the block is still plausibly room tone. Speech
            // sits far above that gate, so an uninterrupted monologue never
            // drags the floor up to speech level and hides a noisy room.
            floor * 0.999 + rms * 0.001
        } else {
            floor
        };
        self.noise_floor_bits
            .store(next.to_bits(), Ordering::Relaxed);
    }

    /// The 0.0..1.0 linear block peak the original `get_mic_level` exposed.
    pub fn peak_linear(&self) -> f32 {
        f32::from_bits(self.peak_bits.load(Ordering::Relaxed)).clamp(0.0, 1.0)
    }

    pub fn snapshot(&self) -> MicMeter {
        let peak = self.peak_linear();
        let rms = f32::from_bits(self.rms_bits.load(Ordering::Relaxed));
        let session_peak = f32::from_bits(self.session_peak_bits.load(Ordering::Relaxed));
        let floor = f32::from_bits(self.noise_floor_bits.load(Ordering::Relaxed));
        let floor_dbfs = if floor.is_finite() {
            amplitude_to_dbfs(floor)
        } else {
            SILENCE_DBFS
        };
        let clipped = self.clipped.load(Ordering::Relaxed);
        let observed = self.observed.load(Ordering::Relaxed);

        MicMeter {
            peak,
            peak_dbfs: amplitude_to_dbfs(peak),
            rms_dbfs: amplitude_to_dbfs(rms),
            session_peak_dbfs: amplitude_to_dbfs(session_peak),
            noise_floor_dbfs: floor_dbfs,
            estimated_snr_db: (amplitude_to_dbfs(rms) - floor_dbfs).max(0.0),
            clipped_samples: clipped,
            clipped_percent: if observed == 0 {
                0.0
            } else {
                (clipped as f64 / observed as f64 * 100.0) as f32
            },
            recording: true,
        }
    }
}

#[derive(Serialize, Clone)]
pub struct RecordingSummary {
    pub file_path: String,
    pub sample_rate: u32,
    pub channels: u16,
}

/// Handle to a recording running on its own dedicated OS thread.
///
/// cpal's Windows (WASAPI) backend is built on COM, which is
/// apartment-threaded: a stream must be created, driven, and dropped on the
/// same thread it was created on. Tauri command handlers do not guarantee
/// that, so the stream itself never leaves `recording_thread` — this handle
/// only holds channels to control it and an atomic for the live level meter.
pub struct ActiveRecording {
    stop_tx: mpsc::Sender<()>,
    done_rx: mpsc::Receiver<Result<(), String>>,
    meter: Arc<LevelMeter>,
    error: Arc<Mutex<Option<String>>>,
    file_path: PathBuf,
    sample_rate: u32,
    channels: u16,
    _join: thread::JoinHandle<()>,
}

pub type RecordingState = Mutex<Option<ActiveRecording>>;

#[tauri::command]
pub fn list_input_devices() -> Result<Vec<AudioDeviceInfo>, String> {
    let host = cpal::default_host();
    let default_name = host.default_input_device().and_then(|d| d.name().ok());

    let devices = host.input_devices().map_err(|e| e.to_string())?;
    let mut result = Vec::new();
    for device in devices {
        let name = device.name().map_err(|e| e.to_string())?;
        let is_default = default_name.as_deref() == Some(name.as_str());
        result.push(AudioDeviceInfo { name, is_default });
    }
    Ok(result)
}

fn find_device(name: Option<&str>) -> Result<cpal::Device, String> {
    let host = cpal::default_host();
    match name {
        None => host
            .default_input_device()
            .ok_or_else(|| "No default microphone available".to_string()),
        Some(target) => {
            let mut devices = host.input_devices().map_err(|e| e.to_string())?;
            devices
                .find(|d| d.name().map(|n| n == target).unwrap_or(false))
                .ok_or_else(|| format!("Microphone '{target}' not found or was disconnected"))
        }
    }
}

/// Unsigned 16-bit capture is centred on 32768; shifting to signed here keeps
/// the metering and the WAV writer working on the same numbers.
fn u16_to_unit(sample: u16) -> f32 {
    (sample as i32 - 32_768) as f32 / 32_768.0
}

fn write_f32(data: &[f32], writer: &Arc<Mutex<hound::WavWriter<BufWriter<File>>>>, meter: &Arc<LevelMeter>) {
    meter.observe(data.iter().copied());
    if let Ok(mut w) = writer.lock() {
        for &sample in data {
            let clamped = sample.clamp(-1.0, 1.0);
            let value = (clamped * i16::MAX as f32) as i16;
            let _ = w.write_sample(value);
        }
    }
}

fn write_i16(data: &[i16], writer: &Arc<Mutex<hound::WavWriter<BufWriter<File>>>>, meter: &Arc<LevelMeter>) {
    meter.observe(data.iter().map(|&s| s as f32 / i16::MAX as f32));
    if let Ok(mut w) = writer.lock() {
        for &sample in data {
            let _ = w.write_sample(sample);
        }
    }
}

fn write_u16(data: &[u16], writer: &Arc<Mutex<hound::WavWriter<BufWriter<File>>>>, meter: &Arc<LevelMeter>) {
    meter.observe(data.iter().map(|&s| u16_to_unit(s)));
    if let Ok(mut w) = writer.lock() {
        for &sample in data {
            let shifted = sample as i32 - (u16::MAX as i32 / 2) - 1;
            let _ = w.write_sample(shifted as i16);
        }
    }
}

/// Runs entirely on its own thread for the lifetime of one recording.
/// Sends `Ok((sample_rate, channels))` on `ready_tx` once the stream is
/// live, then blocks until `stop_rx` fires, finalizes the WAV file, and
/// reports the result on `done_tx`.
fn recording_thread(
    device_name: Option<String>,
    file_path: PathBuf,
    meter: Arc<LevelMeter>,
    error: Arc<Mutex<Option<String>>>,
    ready_tx: mpsc::Sender<Result<(u32, u16), String>>,
    stop_rx: mpsc::Receiver<()>,
    done_tx: mpsc::Sender<Result<(), String>>,
) {
    // The setup steps below (device enumeration, WASAPI stream activation)
    // are exactly where this has been observed to hang/fail, so trace each
    // one to stderr (captured by `tauri dev`'s console) and catch panics
    // that would otherwise just show up as a silent timeout to the caller.
    let ready_tx_panic = ready_tx.clone();
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        recording_thread_inner(
            device_name,
            file_path,
            meter,
            error,
            ready_tx,
            stop_rx,
            done_tx,
        );
    }));
    if let Err(payload) = outcome {
        let msg = payload
            .downcast_ref::<&str>()
            .map(|s| s.to_string())
            .or_else(|| payload.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "Unknown panic in recording thread".to_string());
        eprintln!("[audio] recording thread panicked: {msg}");
        let _ = ready_tx_panic.send(Err(format!("Recording thread panicked: {msg}")));
    }
}

fn recording_thread_inner(
    device_name: Option<String>,
    file_path: PathBuf,
    meter: Arc<LevelMeter>,
    error: Arc<Mutex<Option<String>>>,
    ready_tx: mpsc::Sender<Result<(u32, u16), String>>,
    stop_rx: mpsc::Receiver<()>,
    done_tx: mpsc::Sender<Result<(), String>>,
) {
    eprintln!("[audio] finding device (requested: {device_name:?})...");
    let setup = (|| -> Result<(cpal::Device, cpal::StreamConfig, SampleFormat), String> {
        let device = find_device(device_name.as_deref())?;
        eprintln!("[audio] device found: {:?}", device.name());
        eprintln!("[audio] querying default input config...");
        let config = device.default_input_config().map_err(|e| e.to_string())?;
        eprintln!("[audio] config: {config:?}");
        let sample_format = config.sample_format();
        Ok((device, config.into(), sample_format))
    })();

    let (device, stream_config, sample_format) = match setup {
        Ok(v) => v,
        Err(e) => {
            eprintln!("[audio] setup failed: {e}");
            let _ = ready_tx.send(Err(e));
            return;
        }
    };

    let sample_rate = stream_config.sample_rate.0;
    let channels = stream_config.channels;

    let spec = hound::WavSpec {
        channels,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    eprintln!("[audio] creating WAV writer at {file_path:?} ({spec:?})");
    let writer = match hound::WavWriter::create(&file_path, spec) {
        Ok(w) => Arc::new(Mutex::new(w)),
        Err(e) => {
            eprintln!("[audio] WAV writer creation failed: {e}");
            let _ = ready_tx.send(Err(e.to_string()));
            return;
        }
    };

    let stream_writer = writer.clone();
    let stream_meter = meter.clone();
    let err_flag = error.clone();
    let err_fn = move |err: cpal::StreamError| {
        eprintln!("[audio] input stream error: {err}");
        if let Ok(mut guard) = err_flag.lock() {
            *guard = Some(err.to_string());
        }
    };

    eprintln!("[audio] building input stream (format: {sample_format:?})...");
    let stream_result = match sample_format {
        SampleFormat::F32 => device.build_input_stream(
            &stream_config,
            move |data: &[f32], _| write_f32(data, &stream_writer, &stream_meter),
            err_fn,
            None,
        ),
        SampleFormat::I16 => device.build_input_stream(
            &stream_config,
            move |data: &[i16], _| write_i16(data, &stream_writer, &stream_meter),
            err_fn,
            None,
        ),
        SampleFormat::U16 => device.build_input_stream(
            &stream_config,
            move |data: &[u16], _| write_u16(data, &stream_writer, &stream_meter),
            err_fn,
            None,
        ),
        other => {
            eprintln!("[audio] unsupported sample format: {other:?}");
            let _ = ready_tx.send(Err(format!("Unsupported microphone sample format: {other:?}")));
            return;
        }
    };

    let stream = match stream_result {
        Ok(s) => {
            eprintln!("[audio] input stream built");
            s
        }
        Err(e) => {
            eprintln!("[audio] build_input_stream failed: {e}");
            let _ = ready_tx.send(Err(e.to_string()));
            return;
        }
    };

    eprintln!("[audio] calling stream.play()...");
    if let Err(e) = stream.play() {
        eprintln!("[audio] stream.play() failed: {e}");
        let _ = ready_tx.send(Err(e.to_string()));
        return;
    }
    eprintln!("[audio] stream playing, sending ready");

    if ready_tx.send(Ok((sample_rate, channels))).is_err() {
        // Nobody is listening any more (e.g. start_recording timed out) —
        // still fall through so the stream gets cleaned up below.
        eprintln!("[audio] ready receiver already gone (caller timed out?)");
    }

    // Block here — this thread's only job now is to keep the stream (and
    // its COM apartment, on Windows) alive until told to stop.
    let _ = stop_rx.recv();
    eprintln!("[audio] stop signal received, tearing down stream");

    drop(stream);

    let finalize_result = Arc::try_unwrap(writer)
        .map_err(|_| "Recording buffer still in use".to_string())
        .and_then(|m| m.into_inner().map_err(|e| e.to_string()))
        .and_then(|w| w.finalize().map_err(|e| e.to_string()));

    let _ = done_tx.send(finalize_result);
}

#[tauri::command]
pub fn start_recording(
    app: AppHandle,
    state: State<RecordingState>,
    device_name: Option<String>,
    meeting_id: String,
) -> Result<RecordingSummary, String> {
    let mut guard = state.lock().map_err(|e| e.to_string())?;
    if guard.is_some() {
        return Err("A recording is already in progress".to_string());
    }

    let data_dir = app
        .path()
        .app_data_dir()
        .map_err(|e| e.to_string())?
        .join("meetings")
        .join(&meeting_id);
    std::fs::create_dir_all(&data_dir).map_err(|e| e.to_string())?;
    let file_path = data_dir.join("original.wav");

    let meter = Arc::new(LevelMeter::default());
    let error = Arc::new(Mutex::new(None));
    let (ready_tx, ready_rx) = mpsc::channel();
    let (stop_tx, stop_rx) = mpsc::channel();
    let (done_tx, done_rx) = mpsc::channel();

    let thread_meter = meter.clone();
    let thread_error = error.clone();
    let thread_path = file_path.clone();
    let join = thread::spawn(move || {
        recording_thread(
            device_name,
            thread_path,
            thread_meter,
            thread_error,
            ready_tx,
            stop_rx,
            done_tx,
        );
    });

    let (sample_rate, channels) = ready_rx
        .recv_timeout(Duration::from_secs(5))
        .map_err(|_| "Timed out starting the microphone".to_string())??;

    *guard = Some(ActiveRecording {
        stop_tx,
        done_rx,
        meter,
        error,
        file_path: file_path.clone(),
        sample_rate,
        channels,
        _join: join,
    });

    Ok(RecordingSummary {
        file_path: file_path.to_string_lossy().to_string(),
        sample_rate,
        channels,
    })
}

#[tauri::command]
pub fn stop_recording(state: State<RecordingState>) -> Result<RecordingSummary, String> {
    let mut guard = state.lock().map_err(|e| e.to_string())?;
    let active = guard.take().ok_or_else(|| "No recording in progress".to_string())?;

    let _ = active.stop_tx.send(());
    let finalize = active
        .done_rx
        .recv_timeout(Duration::from_secs(10))
        .map_err(|_| "Timed out stopping the recording".to_string())?;
    finalize?;

    Ok(RecordingSummary {
        file_path: active.file_path.to_string_lossy().to_string(),
        sample_rate: active.sample_rate,
        channels: active.channels,
    })
}

#[tauri::command]
pub fn get_mic_level(state: State<RecordingState>) -> f32 {
    let guard = match state.lock() {
        Ok(g) => g,
        Err(_) => return 0.0,
    };
    guard.as_ref().map(|a| a.meter.peak_linear()).unwrap_or(0.0)
}

/// The full live meter: peak and RMS in dBFS, a rolling noise-floor estimate,
/// and the clipping count for the session. `recording` is false (and every
/// level reads as digital silence) when nothing is being recorded.
#[tauri::command]
pub fn get_mic_meter(state: State<RecordingState>) -> MicMeter {
    let guard = match state.lock() {
        Ok(g) => g,
        Err(_) => return MicMeter::default(),
    };
    guard
        .as_ref()
        .map(|a| a.meter.snapshot())
        .unwrap_or_default()
}

/// Records a short sample from a device and profiles it, so a user can find a
/// quiet or noisy microphone *before* the meeting rather than after it.
///
/// Runs the stream on its own thread for the same COM reason
/// `recording_thread` does, and never touches any meeting's files.
#[tauri::command]
pub fn test_microphone(
    device_name: Option<String>,
    seconds: Option<u32>,
) -> Result<AudioProfile, String> {
    let seconds = seconds.unwrap_or(MIC_TEST_SECONDS).clamp(1, MAX_MIC_TEST_SECONDS);
    let (samples, sample_rate, channels) =
        record_sample(device_name, Duration::from_secs(seconds as u64))?;
    if samples.is_empty() {
        return Err(
            "The microphone produced no audio. Check that it is connected and that Windows lets Cadence use it."
                .to_string(),
        );
    }
    Ok(audio_profile::analyze(&samples, sample_rate, channels))
}

/// Captures `duration` of audio from `device_name`, normalised to -1.0..1.0.
pub fn record_sample(
    device_name: Option<String>,
    duration: Duration,
) -> Result<(Vec<f32>, u32, u16), String> {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        // Same panic guard as the recording thread: a WASAPI failure inside
        // cpal must reach the caller as a message, not as a silent timeout.
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            capture_thread(device_name, duration)
        }))
        .unwrap_or_else(|_| Err("The microphone driver crashed during the test".to_string()));
        let _ = tx.send(outcome);
    });
    rx.recv_timeout(duration + Duration::from_secs(10))
        .map_err(|_| "Timed out reading from the microphone".to_string())?
}

/// The stream is created, played and dropped entirely inside this function, so
/// it never crosses a thread boundary (see `ActiveRecording`).
fn capture_thread(
    device_name: Option<String>,
    duration: Duration,
) -> Result<(Vec<f32>, u32, u16), String> {
    let device = find_device(device_name.as_deref())?;
    let config = device.default_input_config().map_err(|e| e.to_string())?;
    let sample_format = config.sample_format();
    let stream_config: cpal::StreamConfig = config.into();
    let sample_rate = stream_config.sample_rate.0;
    let channels = stream_config.channels;

    // A short test, so the whole sample is buffered. The lock here mirrors the
    // WAV writer's in the recording path; nothing else contends for it.
    let buffer = Arc::new(Mutex::new(Vec::<f32>::with_capacity(
        (sample_rate as usize * channels as usize * duration.as_secs() as usize).min(1 << 24),
    )));
    let sink = buffer.clone();
    let err_fn = |err: cpal::StreamError| eprintln!("[audio] microphone test stream error: {err}");

    let stream = match sample_format {
        SampleFormat::F32 => device.build_input_stream(
            &stream_config,
            move |data: &[f32], _| push(&sink, data.iter().copied()),
            err_fn,
            None,
        ),
        SampleFormat::I16 => device.build_input_stream(
            &stream_config,
            move |data: &[i16], _| push(&sink, data.iter().map(|&s| s as f32 / i16::MAX as f32)),
            err_fn,
            None,
        ),
        SampleFormat::U16 => device.build_input_stream(
            &stream_config,
            move |data: &[u16], _| push(&sink, data.iter().map(|&s| u16_to_unit(s))),
            err_fn,
            None,
        ),
        other => {
            return Err(format!("Unsupported microphone sample format: {other:?}"));
        }
    }
    .map_err(|e| e.to_string())?;

    stream.play().map_err(|e| e.to_string())?;
    thread::sleep(duration);
    drop(stream);

    let samples = Arc::try_unwrap(buffer)
        .map_err(|_| "Microphone test buffer still in use".to_string())?
        .into_inner()
        .map_err(|e| e.to_string())?;
    Ok((samples, sample_rate, channels))
}

fn push<I: Iterator<Item = f32>>(sink: &Arc<Mutex<Vec<f32>>>, samples: I) {
    if let Ok(mut buffer) = sink.lock() {
        buffer.extend(samples);
    }
}

/// Surfaces a stream-level error (e.g. the device was unplugged mid-recording)
/// without tearing down the recording session — the UI can show a warning
/// and let the user decide whether to stop.
#[tauri::command]
pub fn take_recording_error(state: State<RecordingState>) -> Option<String> {
    let guard = state.lock().ok()?;
    let active = guard.as_ref()?;
    let mut err_guard = active.error.lock().ok()?;
    err_guard.take()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_fresh_meter_reads_as_silence_not_as_a_missing_value() {
        let meter = LevelMeter::default();
        let snapshot = meter.snapshot();
        assert_eq!(snapshot.peak, 0.0);
        assert_eq!(snapshot.peak_dbfs, SILENCE_DBFS);
        assert_eq!(snapshot.noise_floor_dbfs, SILENCE_DBFS);
        assert_eq!(snapshot.clipped_samples, 0);
        assert_eq!(snapshot.clipped_percent, 0.0);
        assert!(snapshot.peak_dbfs.is_finite());
    }

    #[test]
    fn peak_and_rms_are_reported_in_dbfs() {
        let meter = LevelMeter::default();
        // A square wave at half scale: peak and RMS are both 0.5 => -6 dBFS.
        let block: Vec<f32> = (0..1_000)
            .map(|i| if i % 2 == 0 { 0.5 } else { -0.5 })
            .collect();
        meter.observe(block.iter().copied());
        let snapshot = meter.snapshot();
        assert!((snapshot.peak - 0.5).abs() < 1e-6);
        assert!((snapshot.peak_dbfs + 6.02).abs() < 0.05);
        assert!((snapshot.rms_dbfs + 6.02).abs() < 0.05);
        assert!(snapshot.recording);
    }

    #[test]
    fn the_legacy_linear_peak_survives_the_upgrade() {
        // get_mic_level's contract: 0.0..1.0, the peak of the last block.
        let meter = LevelMeter::default();
        meter.observe([0.25f32, -0.75, 0.1].into_iter());
        assert!((meter.peak_linear() - 0.75).abs() < 1e-6);
        meter.observe([0.1f32].into_iter());
        assert!((meter.peak_linear() - 0.1).abs() < 1e-6);
        // Out-of-range input from a misbehaving driver still clamps.
        meter.observe([4.0f32].into_iter());
        assert_eq!(meter.peak_linear(), 1.0);
    }

    #[test]
    fn clipped_samples_accumulate_over_the_session() {
        let meter = LevelMeter::default();
        meter.observe([1.0f32, 0.1, -1.0, 0.1].into_iter());
        meter.observe([0.1f32, 0.1, 0.1, 1.0].into_iter());
        let snapshot = meter.snapshot();
        assert_eq!(snapshot.clipped_samples, 3);
        assert!((snapshot.clipped_percent - 37.5).abs() < 0.01);
    }

    #[test]
    fn the_session_peak_holds_after_the_signal_drops() {
        let meter = LevelMeter::default();
        meter.observe([0.8f32].into_iter());
        meter.observe([0.01f32].into_iter());
        let snapshot = meter.snapshot();
        assert!((snapshot.session_peak_dbfs - amplitude_to_dbfs(0.8)).abs() < 1e-3);
        assert!(snapshot.peak_dbfs < snapshot.session_peak_dbfs);
    }

    #[test]
    fn the_noise_floor_follows_a_quiet_room_and_resists_speech() {
        let meter = LevelMeter::default();
        // 200 blocks of room tone settle the floor near the quiet level.
        for _ in 0..200 {
            meter.observe([0.01f32; 16].into_iter());
        }
        let quiet_floor = meter.snapshot().noise_floor_dbfs;
        assert!(
            (quiet_floor - amplitude_to_dbfs(0.01)).abs() < 1.0,
            "floor settled at {quiet_floor}"
        );

        // Then someone talks, without pausing, for a couple of thousand
        // blocks: the floor must barely move, so the SNR the UI shows still
        // reflects the room rather than the speaker.
        for _ in 0..2_000 {
            meter.observe([0.5f32; 16].into_iter());
        }
        let snapshot = meter.snapshot();
        assert!(
            snapshot.noise_floor_dbfs < quiet_floor + 6.0,
            "speech dragged the floor to {}",
            snapshot.noise_floor_dbfs
        );
        assert!(snapshot.estimated_snr_db > 20.0);
    }

    #[test]
    fn an_empty_block_leaves_the_meter_untouched() {
        let meter = LevelMeter::default();
        meter.observe([0.5f32].into_iter());
        meter.observe(std::iter::empty());
        assert!((meter.peak_linear() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn unsigned_capture_is_centred_before_it_is_measured() {
        // 32768 is silence on a u16 device, not full scale.
        assert!(u16_to_unit(32_768).abs() < 1e-6);
        assert!((u16_to_unit(65_535) - 1.0).abs() < 1e-4);
        assert!((u16_to_unit(0) + 1.0).abs() < 1e-6);
        let meter = LevelMeter::default();
        meter.observe([32_768u16, 32_768, 32_768].iter().map(|&s| u16_to_unit(s)));
        assert_eq!(meter.peak_linear(), 0.0);
    }
}
