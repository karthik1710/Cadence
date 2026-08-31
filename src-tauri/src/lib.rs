mod audio;
mod backend;

use audio::RecordingState;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .manage(RecordingState::new(None))
        .invoke_handler(tauri::generate_handler![
            audio::list_input_devices,
            audio::start_recording,
            audio::stop_recording,
            audio::get_mic_level,
            audio::take_recording_error,
            backend::upload_and_process,
            backend::get_meeting_status,
            backend::get_transcript,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
