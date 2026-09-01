mod audio;
mod configuration;
pub mod export;
mod library;
pub mod models_manager;
pub mod pipeline;
mod processing;
pub mod settings;
pub mod storage;
pub mod system;

use audio::RecordingState;
use configuration::InstallState;
use processing::ProcessingState;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .manage(RecordingState::new(None))
        .manage(ProcessingState::default())
        .manage(InstallState::default())
        .setup(|app| {
            // A meeting that was recording or processing when the app last
            // closed is marked recoverable here, before any UI reads it (§33).
            processing::recover_on_startup(app.handle());
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            // Audio (§9, §10)
            audio::list_input_devices,
            audio::start_recording,
            audio::stop_recording,
            audio::get_mic_level,
            audio::take_recording_error,
            audio::get_mic_meter,
            audio::test_microphone,
            pipeline::audio_profile::profile_meeting_audio,
            // Meeting lifecycle and local processing (§11, §12)
            processing::create_meeting,
            processing::finalize_recording,
            processing::start_processing,
            processing::regenerate_mom,
            processing::get_processing_status,
            // Library, transcript, minutes, export (§23-§26)
            library::list_meetings,
            library::get_meeting,
            library::update_meeting,
            library::rename_speaker,
            library::delete_meeting,
            library::get_transcript,
            library::get_mom,
            library::export_meeting,
            library::reveal_meeting_folder,
            // Settings and model management (§27)
            configuration::get_host_info,
            configuration::get_settings,
            configuration::save_settings,
            configuration::check_models,
            configuration::list_models,
            configuration::install_model,
            configuration::get_install_progress,
            configuration::remove_model,
            configuration::verify_model,
            // Native window backdrop (presentation only)
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
