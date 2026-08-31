use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::SampleFormat;
use serde::Serialize;
use std::fs::File;
use std::io::BufWriter;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;
use tauri::{AppHandle, Manager, State};

#[derive(Serialize, Clone)]
pub struct AudioDeviceInfo {
    pub name: String,
    pub is_default: bool,
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
    level: Arc<AtomicU32>,
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

fn update_level(peak: f32, level: &Arc<AtomicU32>) {
    let scaled = (peak.clamp(0.0, 1.0) * 1000.0) as u32;
    level.store(scaled, Ordering::Relaxed);
}

fn write_f32(data: &[f32], writer: &Arc<Mutex<hound::WavWriter<BufWriter<File>>>>, level: &Arc<AtomicU32>) {
    let peak = data.iter().map(|s| s.abs()).fold(0f32, f32::max);
    update_level(peak, level);
    if let Ok(mut w) = writer.lock() {
        for &sample in data {
            let clamped = sample.clamp(-1.0, 1.0);
            let value = (clamped * i16::MAX as f32) as i16;
            let _ = w.write_sample(value);
        }
    }
}

fn write_i16(data: &[i16], writer: &Arc<Mutex<hound::WavWriter<BufWriter<File>>>>, level: &Arc<AtomicU32>) {
    let peak = data
        .iter()
        .map(|s| (*s as f32 / i16::MAX as f32).abs())
        .fold(0f32, f32::max);
    update_level(peak, level);
    if let Ok(mut w) = writer.lock() {
        for &sample in data {
            let _ = w.write_sample(sample);
        }
    }
}

fn write_u16(data: &[u16], writer: &Arc<Mutex<hound::WavWriter<BufWriter<File>>>>, level: &Arc<AtomicU32>) {
    let peak = data
        .iter()
        .map(|s| ((*s as i32 - i16::MAX as i32) as f32 / i16::MAX as f32).abs())
        .fold(0f32, f32::max);
    update_level(peak, level);
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
    level: Arc<AtomicU32>,
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
            level,
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
    level: Arc<AtomicU32>,
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
    let stream_level = level.clone();
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
            move |data: &[f32], _| write_f32(data, &stream_writer, &stream_level),
            err_fn,
            None,
        ),
        SampleFormat::I16 => device.build_input_stream(
            &stream_config,
            move |data: &[i16], _| write_i16(data, &stream_writer, &stream_level),
            err_fn,
            None,
        ),
        SampleFormat::U16 => device.build_input_stream(
            &stream_config,
            move |data: &[u16], _| write_u16(data, &stream_writer, &stream_level),
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

    let level = Arc::new(AtomicU32::new(0));
    let error = Arc::new(Mutex::new(None));
    let (ready_tx, ready_rx) = mpsc::channel();
    let (stop_tx, stop_rx) = mpsc::channel();
    let (done_tx, done_rx) = mpsc::channel();

    let thread_level = level.clone();
    let thread_error = error.clone();
    let thread_path = file_path.clone();
    let join = thread::spawn(move || {
        recording_thread(
            device_name,
            thread_path,
            thread_level,
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
        level,
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
    guard
        .as_ref()
        .map(|a| a.level.load(Ordering::Relaxed) as f32 / 1000.0)
        .unwrap_or(0.0)
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
