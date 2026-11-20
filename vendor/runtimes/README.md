# Bundled native runtimes

This folder is what a Cadence installer ships as `runtimes\` next to the app:
whisper.cpp, llama.cpp and sherpa-onnx, each in its own subfolder. Only this
README is committed. Everything else is generated, and is ignored by git.

`scripts/prepare-runtimes.ps1` fills it. It runs on its own as Tauri's
`build.beforeBundleCommand` during `npm run tauri build` and `npm run release`.
You can also run it by hand:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\prepare-runtimes.ps1
```

It downloads each runtime's pinned release archive into `vendor/cache/` and
refuses one whose SHA-256 does not match the pin. It keeps only the DLLs and
the executables Cadence launches, signs them through `scripts/sign.ps1`, and
writes `manifest.json` (the SHA-256 of every shipped file, which the app's
Verify checks) and `THIRD_PARTY_NOTICES.txt`. Every run rebuilds the folder
from scratch.

With nothing here but this README, `cargo build` and `npm run tauri dev` still
work. The app finds no bundled runtimes and uses the copies the Model Manager
downloads into `%APPDATA%\com.cadence.app\models\bin\`. After you run the
script, a debug build uses the bundled copies too: tauri-build copies this
folder into `src-tauri/target/<profile>/runtimes/`. That copy refreshes only
when the build script re-runs, so touch `src-tauri/tauri.conf.json` (or
`cargo clean`) after you fill or empty this folder. See `docs/release.md`.
