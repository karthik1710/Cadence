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

There are two ways to produce the installer:

- **In CI**, `.github/workflows/release.yml` on GitHub-hosted runners. This is the route for published releases. It signs through SignPath Foundation once the project is accepted there, and it produces an unsigned, clearly labelled build until then. See [Release builds in CI](#release-builds-in-ci).
- **Locally**, `npm run release` with a certificate you hold (below). It stays useful for a test build, or if the project ever gets its own certificate.

```powershell
npm run release
```

`scripts/release.mjs` sets `CADENCE_REQUIRE_SIGNING=1` and runs `tauri build`. That runs, in order:

1. `build.beforeBuildCommand`, `npm run build`: the frontend typecheck and bundle.
2. `cargo build --release` of the app.
3. `build.beforeBundleCommand`, `scripts/prepare-runtimes.ps1` (below). It fills `vendor/runtimes/`.
4. The NSIS bundler. Each file to be signed goes through `bundle.windows.signCommand`, which is `scripts/sign.ps1`. That covers the app executable and every bundled `.exe`/`.dll` that does not already carry a valid signature (Tauri checks with `signtool verify /pa`; files `prepare-runtimes.ps1` already signed pass). It then signs the uninstaller and the finished installer.

Under `CADENCE_REQUIRE_SIGNING=1`, any file that cannot be given a trusted signature fails the build. For a local, unsigned test build, run `npm run tauri build` instead: the same steps run, but `sign.ps1` warns and leaves each file unsigned.

One signing step does not stop the build on its own when it fails: the uninstaller's. Tauri signs it from inside `makensis` through NSIS's `!uninstfinalize`, and `makensis` ignores that command's exit code (checked with NSIS 3.11). So `release.mjs` also sets `CADENCE_SIGN_FAILURE_LOG`, `sign.ps1` appends every failure to that file, and `release.mjs` fails the release if anything is in it, even when `tauri build` itself reported success.

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
| `CADENCE_SIGN_FAILURE_LOG` | A file each signing failure is appended to. `npm run release` sets it and fails if anything was written (see above). |

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

No signing identity has been obtained for Cadence yet. Until one is, every build is unsigned, and none of the reputation benefits above apply. The plan is SignPath Foundation's free open-source signing, through the CI pipeline below. Under it, the bundled runtimes stay unsigned (SignPath Foundation's terms forbid signing upstream binaries), so only the installer, the app executable and the uninstaller carry a signature. Whether endpoint-security products treat the unsigned runtime DLLs inside a signed installer differently from fully signed ones is untested.

## Release builds in CI

`.github/workflows/release.yml` builds the per-user NSIS installer on GitHub-hosted Windows runners. It runs on a pushed `v*` tag, or by hand from Actions → Release → Run workflow, which takes two inputs. `sign` submits the build to SignPath. `release_tag` (default branch only) also publishes a GitHub prerelease under that tag, for example `v0.1.0-rc.1`. The workflow never runs for pull requests. A tag must equal `v` plus the version in `src-tauri/tauri.conf.json`, and that version must match `package.json`, or the run stops before building.

The checks for every push to `main` and every pull request are a separate workflow, `.github/workflows/ci.yml`: `npx tsc --noEmit`, `npm run build`, `cargo build --locked` and `cargo test --locked` (including the offline-guarantee suite) on `windows-latest`. Neither workflow needs models, and neither needs anything in `vendor/runtimes/` beyond the committed README, except that the release build fills it through `prepare-runtimes.ps1` as usual.

### Why SignPath Foundation, and what it allows

The maintainer is an individual without a code-signing certificate, and Azure Trusted Signing is not available to them. [SignPath Foundation](https://signpath.org) offers free code signing to open-source projects. The certificate is issued to SignPath Foundation, so Windows shows SignPath Foundation as the publisher, and signing runs through [SignPath.io](https://about.signpath.io). The conditions that shape this pipeline, from [signpath.org/terms](https://signpath.org/terms):

- **Only the project's own binaries.** "The team must only sign software artifacts built from their own source code", and "upstream OSS projects' binaries must not be signed using your subscription, but may be included in signed packages and installers". So the bundled whisper.cpp, llama.cpp and sherpa-onnx files, and NSIS's own plugin DLLs, are never sent to SignPath. They ship unsigned inside the signed installer. `prepare-runtimes.ps1` runs without a signing identity in CI, so `runtimes/manifest.json` describes those unsigned bytes and the installed app's Verify still matches. The terms add: "we reserve the right to require that signed packages only include signed program and library files in the future". That would need the upstream projects to sign their releases.
- **Verifiable builds, manual approval.** Builds must run on a trusted build system (GitHub-hosted runners), SignPath verifies each artifact's origin (repository, commit, workflow run), and "every release needs manual approval for signing".
- **Metadata restrictions.** Every signed file must have its product name restricted to the project's name, and its product version restricted to the same value across the build. Tauri stamps `Cadence` and the `tauri.conf.json` version on `Cadence.exe`, the uninstaller and the installer, and the artifact configurations enforce both.
- **Project obligations.** An OSI license with no proprietary components, MFA for every team member on GitHub and SignPath, a published [code signing policy](code-signing-policy.md) naming the team roles, a privacy statement, and the attribution "Free code signing provided by SignPath.io, certificate by SignPath Foundation".

### Why the installer is bundled twice

Tauri signs synchronously while it bundles. It calls `signCommand` once per file: the app executable (after patching a package-type marker into it), every unsigned `.exe`/`.dll` resource, NSIS's plugin DLLs, the uninstaller (from inside `makensis`, through `!uninstfinalize`) and the finished installer. SignPath signs a whole GitHub artifact asynchronously, behind a manual approval. It cannot answer a per-file call, and it cannot reach inside an NSIS installer, because it has no NSIS container format. The workflow therefore runs Tauri's bundler twice, with `src-tauri/tauri.ci.conf.json` pointing `signCommand` at `scripts/ci-sign-hook.ps1`:

| Job | Runner | What it does |
|---|---|---|
| `build` | windows-latest | `tauri build` in **collect** mode. The hook signs nothing. It copies Cadence's own binaries, the patched `Cadence.exe` and the uninstaller, under fixed names and records the SHA-256 of each. Its installer is the unsigned build and is always uploaded. |
| `sign-binaries` | ubuntu-latest | SignPath signs `Cadence.exe` and `uninstall.exe` (artifact configuration `cadence-binaries`). |
| `bundle` | windows-latest | `tauri bundle` in **apply** mode, from the same unpatched `Cadence.exe`. When Tauri asks to sign a file whose SHA-256 matches one SignPath signed, the hook puts the signed copy in its place, so the installer embeds a signed app and a signed uninstaller. |
| `sign-installer` | ubuntu-latest | SignPath signs `Cadence_<version>_x64-setup.exe` (artifact configuration `cadence-installer`). |
| `finalize` | windows-latest | Verifies the final installer and writes `SHA256SUMS.txt`. |
| `release` | ubuntu-latest | When a tag was given: a SLSA build provenance attestation (`actions/attest`) for the installer, then a GitHub prerelease with the installer and `SHA256SUMS.txt`. |

The second pass works because both of Cadence's own binaries are reproducible from the same inputs. Tauri's patch is a fixed byte substitution in the unpatched executable. `makensis` wrote a byte-identical uninstaller across runs, directories and `TEMP` folders when tested with NSIS 3.11 on 2026-11-20. Every mismatch fails the build rather than shipping an unsigned file:

- In apply mode, with `CADENCE_CI_SIGN_STRICT=1`, the hook fails if a binary's SHA-256 matches nothing SignPath signed, or if the signed copy is not validly signed, is not timestamped, or is signed by someone other than `SIGNPATH_EXPECTED_SIGNER`.
- `makensis` ignores the uninstaller hook's exit code, so the hook also appends each replacement to `applied.txt`. `scripts/ci-verify-installer.ps1` then fails unless every collected binary was replaced.
- `ci-verify-installer.ps1` also checks the installer's own signature state: unsigned in the build job, signed in `finalize` when signing ran. It then unpacks the installer with 7-Zip. Inside, `Cadence.exe` must carry the expected signature state, every runtime file must match `runtimes\manifest.json` with no extras, and `pipeline_cli`/`bench_cli` must be absent.
- Any file the hook cannot classify fails the build. That means a Tauri update started signing something new, and the hook has to be taught whether it is Cadence's or upstream's.

The SignPath API token is visible only to the two signing jobs, which run SignPath's action and no repository code. The build jobs, which run `npm ci`, Cargo build scripts and `prepare-runtimes.ps1`, never see it. Every third-party action is pinned to a full commit SHA; `.github/dependabot.yml` proposes monthly updates. Release builds use no build cache.

### Without SignPath

If the trigger does not ask for signing, or the repository variables below are not set, the signing, second bundle and SignPath jobs are skipped. `finalize` verifies the first pass's installer as **unsigned** and uploads it as the `Cadence-<version>-unsigned` artifact. On a tag, the prerelease is titled "(unsigned test build)" and its notes say so. A signing run that fails publishes nothing.

### Configuration

Repository **variables** (Settings → Secrets and variables → Actions → Variables):

| Variable | Required | Default | Meaning |
|---|---|---|---|
| `SIGNPATH_ORGANIZATION_ID` | yes | none | SignPath organization ID. Signing is enabled only when it and the project slug are set. |
| `SIGNPATH_PROJECT_SLUG` | yes | none | SignPath project slug. |
| `SIGNPATH_SIGNING_POLICY_SLUG` | no | `release-signing` | Signing policy used for both requests. |
| `SIGNPATH_BINARIES_CONFIG_SLUG` | no | `cadence-binaries` | Artifact configuration for `Cadence.exe` + `uninstall.exe`. |
| `SIGNPATH_INSTALLER_CONFIG_SLUG` | no | `cadence-installer` | Artifact configuration for the installer. |
| `SIGNPATH_EXPECTED_SIGNER` | no | `SignPath Foundation` | Substring the signer certificate's subject must contain. |
| `SIGNPATH_WAIT_SECONDS` | no | `7200` | How long each signing job waits for the manual approval. |

The **secret** `SIGNPATH_API_TOKEN` is the API token of a SignPath user with submitter rights on the project and policy. Store it as a secret of the `signpath` environment that both signing jobs use (Settings → Environments), with deployment restricted to `main` and `v*` tags, so that a workflow on any other branch cannot use it. A repository secret also works.

The artifact configurations are versioned in `.github/signpath/artifact-configurations/`. SignPath does not read them from the repository, so after changing one, paste it into the SignPath project again.

### Setting up SignPath (maintainer)

1. Enable MFA on GitHub, and later on SignPath.
2. Fill in and publish [code-signing-policy.md](code-signing-policy.md): confirm the team roles and the privacy statement, and link it from the README as "Code signing policy".
3. Publish at least one release in the form to be signed. SignPath Foundation accepts only projects that are already released. An unsigned prerelease from this workflow (run it with `release_tag`) is that form.
4. Apply at [signpath.org/apply](https://signpath.org/apply) with the repository URL, the license (MIT) and a link to the policy page. This is the maintainer's own decision and action.
5. Once accepted, in SignPath:
   - make sure the project's repository URL is `https://github.com/karthik1710/Cadence` and that the predefined trusted build system *GitHub.com* is linked to it;
   - add the two artifact configurations from `.github/signpath/artifact-configurations/` with the slugs `cadence-binaries` and `cadence-installer`;
   - use a `release-signing` policy with origin verification and manual approval by the approvers in the policy page. Restrict it to the refs releases are built from. SignPath's documentation shows branch patterns such as `main` and `release/*`, but does not say whether a tag ref matches. If tag-triggered runs are refused, release by running the workflow by hand on `main` with `release_tag` set;
   - create an API token for a user who is a submitter on that policy.
6. In GitHub, set the variables and the secret above.
7. Run the workflow by hand on `main` with `sign` ticked and no `release_tag`. Approve both signing requests in SignPath, then download the `Cadence-<version>-signed` artifact and check it as in [Verifying a build](#verifying-a-build). Then release by tag, or by hand with `release_tag`.
8. Change the status in `code-signing-policy.md` and enable the README's signing note.

Each signed release needs two approvals in SignPath, one for the binaries and one for the installer, because the installer cannot exist until the signed binaries are inside it.

### Verifying a CI release

```powershell
gh attestation verify Cadence_0.1.0-rc.1_x64-setup.exe --repo karthik1710/Cadence   # built by this repository's workflow
Get-FileHash Cadence_0.1.0-rc.1_x64-setup.exe -Algorithm SHA256                       # compare with SHA256SUMS.txt
Get-AuthenticodeSignature Cadence_0.1.0-rc.1_x64-setup.exe | Format-List Status, SignerCertificate, TimeStamperCertificate
```

A signed release reports `Valid` with a SignPath Foundation certificate and a timestamp. The checks under [Verifying a build](#verifying-a-build) apply too, except that the runtime files are expected to be unsigned in a SignPath-signed build.

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
