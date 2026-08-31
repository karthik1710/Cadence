mod audio;
pub mod pipeline;
mod processing;

use audio::RecordingState;
use processing::ProcessingState;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .manage(RecordingState::new(None))
        .manage(ProcessingState::default())
        .invoke_handler(tauri::generate_handler![
            audio::list_input_devices,
            audio::start_recording,
            audio::stop_recording,
            audio::get_mic_level,
            audio::take_recording_error,
            processing::start_processing,
            processing::get_processing_status,
            processing::get_transcript,
            processing::check_models,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
