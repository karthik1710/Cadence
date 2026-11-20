<#
.SYNOPSIS
  Fetches the native AI binaries and model weights Cadence's local pipeline
  needs (whisper.cpp, llama.cpp, sherpa-onnx + their model files) into the
  same directory the shipped app resolves at runtime (see
  src-tauri/src/pipeline/paths.rs). Dev-time convenience only — the primary
  install path is the app's first-run setup screen and Model Manager
  (src-tauri/src/models_manager.rs).

  Fetches exactly the required set, models_manager::required_ids(): the three
  runtimes, Whisper small, Qwen3-8B Q4_K_M, the two diarization models, the
  GTCRN denoiser and Silero VAD. URLs must match models_manager::catalog().

  Safe to re-run: every step is skipped if its output already exists.
#>

$ErrorActionPreference = "Stop"

$AppDataDir = Join-Path $env:APPDATA "com.cadence.app"
$ModelsDir = Join-Path $AppDataDir "models"
$BinDir = Join-Path $ModelsDir "bin"
# One folder per runtime, as models_manager::Runtime installs them: whisper.cpp
# and llama.cpp both ship ggml*.dll from different ggml versions, and Windows
# loads an exe's DLLs from its own folder first. Never copy into $BinDir itself.
$WhisperBin = Join-Path $BinDir "whisper"
$LlamaBin = Join-Path $BinDir "llama"
$SherpaBin = Join-Path $BinDir "sherpa"
$SttDir = Join-Path $ModelsDir "stt"
$LlmDir = Join-Path $ModelsDir "llm"
$DiarDir = Join-Path $ModelsDir "diarization"
$DenoiseDir = Join-Path $ModelsDir "denoise"
$VadDir = Join-Path $ModelsDir "vad"
$TmpDir = Join-Path $ModelsDir "_tmp"

foreach ($d in @($WhisperBin, $LlamaBin, $SherpaBin, $SttDir, $LlmDir, $DiarDir, $DenoiseDir, $VadDir, $TmpDir)) {
    New-Item -ItemType Directory -Force -Path $d | Out-Null
}

function Get-IfMissing {
    param([string]$Url, [string]$OutFile)
    if (Test-Path $OutFile) {
        Write-Output "  already have: $OutFile"
        return
    }
    Write-Output "  downloading: $Url"
    curl.exe -L --fail --progress-bar -o $OutFile $Url
}

Write-Output "=== whisper.cpp (STT) ==="
$whisperZip = Join-Path $TmpDir "whisper-bin-x64.zip"
Get-IfMissing "https://github.com/ggml-org/whisper.cpp/releases/download/b4938/whisper-bin-x64.zip" $whisperZip
if (-not (Test-Path (Join-Path $WhisperBin "whisper-cli.exe"))) {
    Expand-Archive -Path $whisperZip -DestinationPath (Join-Path $TmpDir "whisper-extract") -Force
    Get-ChildItem (Join-Path $TmpDir "whisper-extract") -Recurse -File | Copy-Item -Destination $WhisperBin -Force
}
Get-IfMissing "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin" (Join-Path $SttDir "ggml-small.bin")

Write-Output "=== llama.cpp (LLM runtime) ==="
$llamaZip = Join-Path $TmpDir "llama-bin-win-cpu-x64.zip"
Get-IfMissing "https://github.com/ggml-org/llama.cpp/releases/download/b10717/llama-b10717-bin-win-cpu-x64.zip" $llamaZip
if (-not (Test-Path (Join-Path $LlamaBin "llama-server.exe"))) {
    Expand-Archive -Path $llamaZip -DestinationPath (Join-Path $TmpDir "llama-extract") -Force
    Get-ChildItem (Join-Path $TmpDir "llama-extract") -Recurse -File | Copy-Item -Destination $LlamaBin -Force
}

Write-Output "=== Qwen3-8B Q4_K_M GGUF (LLM weights) ==="
Get-IfMissing "https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf" (Join-Path $LlmDir "Qwen3-8B-Q4_K_M.gguf")

Write-Output "=== sherpa-onnx (diarization) ==="
$sherpaArchive = Join-Path $TmpDir "sherpa-onnx-win-x64.tar.bz2"
Get-IfMissing "https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.13.6/sherpa-onnx-v1.13.6-win-x64-shared-MD-Release-no-tts.tar.bz2" $sherpaArchive
if (-not (Test-Path (Join-Path $SherpaBin "sherpa-onnx-offline-speaker-diarization.exe"))) {
    $sherpaExtract = Join-Path $TmpDir "sherpa-extract"
    New-Item -ItemType Directory -Force -Path $sherpaExtract | Out-Null
    tar.exe -xjf $sherpaArchive -C $sherpaExtract
    Get-ChildItem $sherpaExtract -Recurse -Include *.exe, *.dll | Copy-Item -Destination $SherpaBin -Force
}

$segArchive = Join-Path $TmpDir "pyannote-segmentation.tar.bz2"
Get-IfMissing "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2" $segArchive
if (-not (Test-Path (Join-Path $DiarDir "sherpa-onnx-pyannote-segmentation-3-0"))) {
    tar.exe -xjf $segArchive -C $DiarDir
}

Get-IfMissing "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/wespeaker_en_voxceleb_CAM%2B%2B.onnx" (Join-Path $DiarDir "wespeaker_en_voxceleb_CAM++.onnx")

Write-Output "=== GTCRN speech denoiser + Silero VAD (ONNX, run on sherpa-onnx) ==="
Get-IfMissing "https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/gtcrn_simple.onnx" (Join-Path $DenoiseDir "gtcrn_simple.onnx")
Get-IfMissing "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx" (Join-Path $VadDir "silero_vad.onnx")

Remove-Item -Recurse -Force $TmpDir -ErrorAction SilentlyContinue

Write-Output ""
Write-Output "=== Done. Layout under $ModelsDir : ==="
Get-ChildItem -Recurse $ModelsDir | Select-Object FullName, Length | Format-Table -AutoSize
