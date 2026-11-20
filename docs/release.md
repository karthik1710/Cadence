# Cadence — Releasing

How to produce a signed Windows installer, what goes into it, and how to check one before it goes out. The bundled-runtime design is in [architecture.md](architecture.md#runtime-layout). What is still unverified about endpoint-security behaviour is in [roadmap.md](roadmap.md) (risks 5 and 16).

## What a release is

A release is one NSIS installer, `src-tauri/target/release/bundle/nsis/Cadence_<version>_x64-setup.exe`. Its contents:

- the app executable, signed;
- `runtimes\`, holding the three native runtimes Cadence drives: `whisper\` (whisper.cpp b4938), `llama\` (llama.cpp b10717) and `sherpa\` (sherpa-onnx v1.13.6 with ONNX Runtime 1.27.1). Each folder has only the executables Cadence launches and the DLLs from that release. The folder also holds `manifest.json`, a SHA-256 for every shipped file, and `THIRD_PARTY_NOTICES.txt`, the upstream license texts;
- the WebView2 bootstrapper (see [WebView2](#webview2));
- an uninstaller, signed.

The installer carries **no model weights**. They are several gigabytes of data, and first-run setup downloads them after the user agrees to a list showing each file and its size. Because the runtimes ship inside the installer, a release build never downloads an executable. The only things it downloads are `.bin`, `.gguf` and `.onnx` model files.

## Producing a release

```powershell
npm run release
```

`scripts/release.mjs` sets `CADENCE_REQUIRE_SIGNING=1` and runs `tauri build`. That runs, in order:

1. `build.beforeBuildCommand`, `npm run build`: the frontend typecheck and bundle.
2. `cargo build --release` of the app.
3. `build.beforeBundleCommand`, `scripts/prepare-runtimes.ps1` (below). It fills `vendor/runtimes/`.
4. The NSIS bundler. Each file to be signed goes through `bundle.windows.signCommand`, which is `scripts/sign.ps1`. That covers the app executable and every bundled `.exe`/`.dll` that does not already carry a valid signature (Tauri checks with `signtool verify /pa`; files `prepare-runtimes.ps1` already signed pass). It then signs the uninstaller and the finished installer.

Under `CADENCE_REQUIRE_SIGNING=1`, any file that cannot be given a trusted signature fails the build. For a local, unsigned test build, run `npm run tauri build` instead: the same steps run, but `sign.ps1` warns and leaves each file unsigned.

### `scripts/prepare-runtimes.ps1`

The script pins each runtime twice: by the exact release URL the app's catalog names (`models_manager.rs`), and by the SHA-256 of that archive.

| Runtime | Archive | SHA-256 |
|---|---|---|
| whisper.cpp b4938 | `whisper-bin-x64.zip` | `c2a4b60edb11f7e11a9191ffb50929535527d4d91c9903dbe3e554583bbbc63d` |
| llama.cpp b10717 | `llama-b10717-bin-win-cpu-x64.zip` | `a8c6f95fc6b2f690ebbcabc8bba72cd0ba428353eca0a663556ab515f9203593` |
| sherpa-onnx v1.13.6 | `sherpa-onnx-v1.13.6-win-x64-shared-MD-Release-no-tts.tar.bz2` | `071d6641efd737a1f60de48c9c4cd596f78d5b0980815e8ad3798c95785d2b26` |

Each hash was taken from a fresh download on 2026-11-20. It was cross-checked file by file against the digests the app had recorded, at install time, for its own downloaded copies of the same releases (installed through the Model Manager on 2026-11-19). All 54 shipped files matched. That cross-check is two downloads of the same URL, a day apart, agreeing. It is not an independent attestation of what upstream built: none of the three projects publishes a checksum for these archives.

For each runtime, the script:

1. Uses the cached archive in `vendor/cache/` if its SHA-256 matches the pin. Otherwise it downloads the archive again with Windows' own `curl.exe` (`--disable --proto =https --proto-redir =https`). A mismatch is a hard failure: nothing is bundled.
2. Unpacks it with `%SystemRoot%\System32\tar.exe`, never a `tar` from `PATH`, because Git Bash's tar misreads `C:`.
3. Keeps every `.dll` plus only the executables Cadence launches: `whisper-cli.exe` and `whisper-server.exe`; `llama-server.exe`; and `sherpa-onnx-offline-speaker-diarization.exe`, `sherpa-onnx-offline-denoiser.exe` and `sherpa-onnx-vad.exe`. Test, example and benchmark executables are dropped. It refuses an archive that holds two different files with the same name.
4. Signs every kept file through `sign.ps1`.
5. Only then writes `manifest.json`, so the digests describe the signed bytes the installer ships.

Everything is assembled in `vendor/cache/staging/` and swapped into `vendor/runtimes/` only once complete, so a failed run never leaves a half-built runtime folder behind. The script never executes anything it downloads. It is idempotent: a second run reuses the cache and reproduces the same manifest.

Last run here (2026-11-20, no signing identity): whisper 17 files / 15.5 MB, llama 30 files / 42.8 MB, sherpa 7 files / 19.6 MB, all 54 unsigned. None of the upstream files carries a signature of its own, including `onnxruntime.dll`: sherpa-onnx builds ONNX Runtime itself, so its copy is not Microsoft's signed binary. Every bundled file therefore ends up with Cadence's signature, or with none.

To move a runtime to a new release, change its URL and SHA-256 in `prepare-runtimes.ps1` and the matching catalog entry in `models_manager.rs` together. Check that the executables Cadence launches still exist, update `src-tauri/THIRD_PARTY_NOTICES.txt` if the license set changed, and re-run `bench_cli baseline` ([benchmarks.md](benchmarks.md)). Never edit a pin just to make a mismatch pass.

## Signing identities

Nothing secret lives in the repository. `scripts/sign.ps1` reads the identity from the environment, in this order:

| Variable | Meaning |
|---|---|
| `CADENCE_SIGN_CERT_THUMBPRINT` | SHA-1 thumbprint of a code-signing certificate in `CurrentUser\My` or `LocalMachine\My`. Use it for an OV or EV certificate, including one on a hardware token that exposes the certificate there. |
| `CADENCE_SIGN_DLIB` + `CADENCE_SIGN_DLIB_METADATA` | Azure Trusted Signing (Artifact Signing): the path to `Azure.CodeSigning.Dlib.dll` and to its `metadata.json` (endpoint, account, certificate profile). The signing key stays in Azure. |
| `CADENCE_SIGN_TIMESTAMP_URL` | Optional RFC 3161 timestamp server. The default is DigiCert's for a certificate and Microsoft's for Trusted Signing. |
| `CADENCE_SIGNTOOL` | Optional full path to `signtool.exe`. The default is the newest x64 `signtool` in the Windows SDK. |
| `CADENCE_REQUIRE_SIGNING=1` | Fail instead of warning when no identity is configured, or when a signature does not chain to a trusted root. `npm run release` sets it. |

Every signature is SHA-256 and **timestamped**. A timestamp keeps the signature valid after the certificate expires, which matters most for Trusted Signing's short-lived certificates. `sign.ps1` retries a failed timestamp request before giving up. A file that already carries a valid signature is left alone, so an upstream publisher's signature would be kept. No file Cadence bundles today has one.

Some notes on the build machine:

- **Install the Windows SDK signing tools even for an unsigned build.** Because `signCommand` is configured, Tauri runs `signtool verify /pa` on every bundled `.exe`/`.dll` before deciding whether to sign it. Without `signtool.exe`, the bundle step fails with "SignTool not found".
- **Do not release with an untrusted test certificate.** A signature that does not chain to a trusted root is not `Valid`, so Tauri's check re-signs those files after `manifest.json` was written. The installed app's Verify then reports them as changed. `CADENCE_REQUIRE_SIGNING=1` refuses such a signature anyway.
- **Paths in `signCommand` resolve against `src-tauri/`.** Tauri changes into `src-tauri/` before bundling and turns any relative `signCommand` argument that exists there into an absolute path. It does that so the uninstaller's signing step, which runs in another directory, still finds it. That is why `../scripts/sign.ps1` works. `beforeBundleCommand` sets `"cwd": ".."` (the repository root), and `prepare-runtimes.ps1` resolves all its own paths from `$PSScriptRoot`, so it works from any directory.

## What signing and bundling buy, and what they do not

Before this change, first-run setup downloaded unsigned executables and DLLs from GitHub into `%APPDATA%` and then ran them. Endpoint security looks for exactly that pattern, and Microsoft Defender's attack-surface-reduction rule "Block executable files from running unless they meet a prevalence, age, or trusted list criterion" targets it. Now every executable Cadence runs arrives inside one signed installer, at install time, carrying the same publisher signature. The app downloads no code at runtime. That removes the strongest dropper-like signal, and it lets reputation systems treat the app as one signed product rather than a stream of unknown binaries.

It cannot guarantee anything about a particular security product:

- **SmartScreen reputation is earned, not bought.** An installer signed with an OV certificate still shows "Windows protected your PC" until enough people have downloaded and run builds signed with that certificate. Reputation attaches to the certificate and builds up over downloads. EV certificates and Azure Trusted Signing usually get there much faster, but neither guarantees it from the first build.
- **Antivirus heuristics can still fire.** llama.cpp and whisper.cpp builds are common enough to have triggered machine-learning false positives elsewhere, and a newly signed file has no prevalence yet. Submit a false positive to Microsoft through the [Windows Defender Security Intelligence portal](https://www.microsoft.com/en-us/wdsi/filesubmission), as a software developer, with the signed installer. Other vendors run similar submission portals.
- **Model downloads remain.** First-run setup still fetches model data files with `curl.exe` and `tar.exe` (`tar.exe` unpacks the diarization model archive). These are data, not code, but how an EDR product reacts to a desktop app spawning `curl.exe` is untested ([roadmap.md](roadmap.md) risk 16).
- **Signing proves origin, not safety.** A signature says the files came from whoever holds the key. It says nothing about the upstream builds themselves, which are pinned by hash but not independently attested (risk 5).

No signing identity has been obtained for Cadence yet. Until one is, every build is unsigned, and none of the reputation benefits above apply.

## Install location

The installer is per-user (`bundle.windows.nsis.installMode: "currentUser"`). It installs to `%LOCALAPPDATA%\Cadence` with no administrator rights and no UAC prompt, and it registers an uninstaller under the current user only. `runtimes\` sits next to the app executable, because Tauri's resource directory on Windows is the executable's directory. User data (`%APPDATA%\com.cadence.app\`: settings, models, meetings) is separate from the install folder: uninstalling or reinstalling Cadence leaves it alone, and reinstalling is how a damaged bundled runtime is repaired.

## WebView2

`webviewInstallMode` is `embedBootstrapper` (silent). The installer carries Microsoft's small WebView2 bootstrapper and runs it only if the WebView2 runtime is missing. Windows 11 ships WebView2, so there it does nothing. On a machine without it, the bootstrapper downloads the runtime from Microsoft at install time, so that one case needs a network connection during installation. The app itself never does.

## Verifying a build

Before publishing an installer:

```powershell
$setup = 'src-tauri\target\release\bundle\nsis\Cadence_0.1.0_x64-setup.exe'

# 1. The installer is signed and timestamped, by the expected publisher.
Get-AuthenticodeSignature $setup | Format-List Status, StatusMessage, SignerCertificate, TimeStamperCertificate

# 2. Every bundled runtime file is signed (run against the build folder or an installed copy).
Get-ChildItem vendor\runtimes -Recurse -Include *.exe, *.dll |
    Get-AuthenticodeSignature | Where-Object Status -ne 'Valid' | Select-Object Path, Status

# 3. Microsoft Defender's verdict on the installer (custom scan of one file).
& "$env:ProgramFiles\Windows Defender\MpCmdRun.exe" -Scan -ScanType 3 -File (Resolve-Path $setup)
```

Step 2 should print nothing. After installing on a clean machine (a fresh VM is best), open Settings → Models. The three runtimes should read "Included with Cadence", Verify on each should report every file unchanged since the build, and the setup screen should list model data only. Recording a short meeting with the network adapter disabled, after the models are installed, is still the test that matters ([roadmap.md](roadmap.md) risk 6).
