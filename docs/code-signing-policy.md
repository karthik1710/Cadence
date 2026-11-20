# Code signing policy

<!--
  OWNER: this page is the code signing policy SignPath Foundation requires
  (https://signpath.org/terms, "Conditions for the website / repository").
  Before applying, confirm every line marked OWNER, then link this page from
  the README and from every release page as "Code signing policy" (the
  release workflow already does the latter). When SignPath Foundation accepts
  the project, change "Status" below. Delete these comments when done.
-->

## Status

**Cadence's Windows releases are not code-signed yet.** The project intends to apply to [SignPath Foundation](https://signpath.org) for free open-source code signing. Until it is accepted, every installer on the [Releases page](https://github.com/karthik1710/Cadence/releases) is an unsigned test build, labelled as such, and ships with a SHA-256 checksum and a GitHub build provenance attestation.

<!-- OWNER: once accepted, replace the paragraph above with:
Windows releases of Cadence are code-signed. Free code signing provided by [SignPath.io](https://about.signpath.io), certificate by [SignPath Foundation](https://signpath.org).
-->

When signing is in place:

> Free code signing provided by [SignPath.io](https://about.signpath.io), certificate by [SignPath Foundation](https://signpath.org).

The certificate is issued to SignPath Foundation, so Windows shows **SignPath Foundation** as the publisher of signed Cadence files.

## What is signed

Every signed file is built from this repository's source by the [release workflow](../.github/workflows/release.yml) on GitHub-hosted runners. SignPath verifies the origin of each build (the repository, the commit and the workflow run) before it signs, and every signing request is approved by hand by an approver listed below.

| File | Signed |
|---|---|
| `Cadence_<version>_x64-setup.exe`, the installer | Yes |
| `Cadence.exe`, the application | Yes |
| `uninstall.exe`, the uninstaller | Yes |
| `runtimes\whisper\*`, `runtimes\llama\*`, `runtimes\sherpa\*`: whisper.cpp, llama.cpp and sherpa-onnx (with ONNX Runtime) | **No** |
| NSIS plugin DLLs the installer uses while it runs | **No** |
| Microsoft's WebView2 bootstrapper, run by the installer only when WebView2 is missing | Carries Microsoft's own signature |

The runtimes and NSIS plugins are upstream open-source builds. Cadence downloads them from their projects' official releases at build time, pinned by URL and SHA-256 ([scripts/prepare-runtimes.ps1](../scripts/prepare-runtimes.ps1)), and ships them unchanged. SignPath Foundation's terms allow a project to sign only binaries built from its own source code, while allowing unsigned upstream binaries inside a signed installer. Cadence follows both rules. The installed app's Verify (Settings → Models) checks every runtime file against the SHA-256 digests recorded at build time (`runtimes\manifest.json`).

The exact SignPath artifact configurations are in [.github/signpath/artifact-configurations/](../.github/signpath/artifact-configurations/).

## Team roles

<!-- OWNER: confirm these. Cadence currently has a single maintainer, which
     holds every role. If you add maintainers, list them here and give them
     the matching GitHub repository role. Every member needs multi-factor
     authentication on GitHub and on SignPath. -->

| Role | Responsibility | Members |
|---|---|---|
| Committers | May change the source code, including the build scripts and CI configuration in `scripts/` and `.github/`, without a further review | [karthik1710](https://github.com/karthik1710) (repository owner) |
| Reviewers | Review every change proposed by people who are not committers (pull requests) before it is merged | [karthik1710](https://github.com/karthik1710) |
| Approvers | Approve each signing request in SignPath, deciding whether a given release may be signed | [karthik1710](https://github.com/karthik1710) |

Changes to `scripts/`, `.github/workflows/`, `src-tauri/tauri.conf.json` and `src-tauri/tauri.ci.conf.json` decide what gets signed, so they get the same review as application code, or more.

All team members use multi-factor authentication for GitHub and for SignPath.

## Privacy policy

This program will not transfer any information to other networked systems unless specifically requested by the user or the person installing or operating it.

In detail:

- **Recording and processing happen on the user's computer.** Audio, transcripts and minutes are written only to `%APPDATA%\com.cadence.app\`. Cadence has no telemetry, analytics, crash reporting or update checks, and no cloud speech-to-text or language-model service. Its test suite enforces this structurally ([docs/architecture.md](architecture.md)).
- **Model downloads happen only on request.** After installation, Cadence downloads its AI model files only when the user starts it from the setup screen or Settings → Models, after it has listed every file and its size. The files come from GitHub (`github.com`, release downloads of whisper.cpp, sherpa-onnx and related projects) and Hugging Face (`huggingface.co`). Those hosts see the request, including the user's IP address, as they would for any download; their own privacy policies apply ([GitHub](https://docs.github.com/site-policy/privacy-policies/github-general-privacy-statement), [Hugging Face](https://huggingface.co/privacy)). Nothing about the user's meetings is sent.
- **Microsoft Edge WebView2.** Cadence's interface runs in the WebView2 runtime that Windows provides. On a machine without it, the installer runs Microsoft's WebView2 bootstrapper, which downloads the runtime from Microsoft. WebView2 is a Microsoft component governed by the [Microsoft Privacy Statement](https://privacy.microsoft.com/privacystatement). Cadence loads no remote web content into it.

## Uninstallation

Cadence installs per user, without administrator rights, to `%LOCALAPPDATA%\Cadence`. Remove it from **Settings → Apps → Installed apps** or with `%LOCALAPPDATA%\Cadence\uninstall.exe`. By default the uninstaller leaves user data (`%APPDATA%\com.cadence.app\`: settings, downloaded models, recorded meetings) in place, so that a reinstall keeps it. Tick the uninstaller's option to delete the application data, or delete that folder by hand, to remove it as well.

## Reporting a problem

If you believe a file signed for Cadence violates this policy, open an issue on this repository or contact the maintainer. Reports of signed malware or policy violations can also go to SignPath directly at `support@signpath.io`.

<!-- OWNER: add a private contact (email or GitHub security advisory link) if you prefer reports not to be public. -->
