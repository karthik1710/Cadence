<#
.SYNOPSIS
  Fetches the native AI binaries and model weights Cadence's local pipeline
  needs (whisper.cpp, llama.cpp, sherpa-onnx + their model files) into the
  same directory the shipped app resolves at runtime (see
  src-tauri/src/pipeline/paths.rs). Dev-time convenience only — this is not
  the production Model Manager UI (src-tauri/src/models_manager.rs).

  Safe to re-run: every step is skipped if its output already exists.
#>

$ErrorActionPreference = "Stop"

$AppDataDir = Join-Path $env:APPDATA "com.cadence.app"
$ModelsDir = Join-Path $AppDataDir "models"
$BinDir = Join-Path $ModelsDir "bin"
$SttDir = Join-Path $ModelsDir "stt"
$LlmDir = Join-Path $ModelsDir "llm"
$DiarDir = Join-Path $ModelsDir "diarization"
$TmpDir = Join-Path $ModelsDir "_tmp"

foreach ($d in @($BinDir, $SttDir, $LlmDir, $DiarDir, $TmpDir)) {
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
if (-not (Test-Path (Join-Path $BinDir "whisper-cli.exe"))) {
    Expand-Archive -Path $whisperZip -DestinationPath (Join-Path $TmpDir "whisper-extract") -Force
    Get-ChildItem (Join-Path $TmpDir "whisper-extract") -Recurse -File | Copy-Item -Destination $BinDir -Force
}
Get-IfMissing "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin" (Join-Path $SttDir "ggml-small.bin")

Write-Output "=== llama.cpp (LLM runtime) ==="
$llamaZip = Join-Path $TmpDir "llama-bin-win-cpu-x64.zip"
Get-IfMissing "https://github.com/ggml-org/llama.cpp/releases/download/b10717/llama-b10717-bin-win-cpu-x64.zip" $llamaZip
if (-not (Test-Path (Join-Path $BinDir "llama-server.exe"))) {
    Expand-Archive -Path $llamaZip -DestinationPath (Join-Path $TmpDir "llama-extract") -Force
    Get-ChildItem (Join-Path $TmpDir "llama-extract") -Recurse -File | Copy-Item -Destination $BinDir -Force
}

Write-Output "=== Qwen3-8B Q4_K_M GGUF (LLM weights) ==="
Get-IfMissing "https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf" (Join-Path $LlmDir "Qwen3-8B-Q4_K_M.gguf")

Write-Output "=== sherpa-onnx (diarization) ==="
$sherpaArchive = Join-Path $TmpDir "sherpa-onnx-win-x64.tar.bz2"
Get-IfMissing "https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.13.6/sherpa-onnx-v1.13.6-win-x64-shared-MD-Release-no-tts.tar.bz2" $sherpaArchive
if (-not (Test-Path (Join-Path $BinDir "sherpa-onnx-offline-speaker-diarization.exe"))) {
    $sherpaExtract = Join-Path $TmpDir "sherpa-extract"
    New-Item -ItemType Directory -Force -Path $sherpaExtract | Out-Null
    tar.exe -xjf $sherpaArchive -C $sherpaExtract
    Get-ChildItem $sherpaExtract -Recurse -Include *.exe, *.dll | Copy-Item -Destination $BinDir -Force
}

$segArchive = Join-Path $TmpDir "pyannote-segmentation.tar.bz2"
Get-IfMissing "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2" $segArchive
if (-not (Test-Path (Join-Path $DiarDir "sherpa-onnx-pyannote-segmentation-3-0"))) {
    tar.exe -xjf $segArchive -C $DiarDir
}

Get-IfMissing "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/wespeaker_en_voxceleb_CAM++.onnx" (Join-Path $DiarDir "wespeaker_en_voxceleb_CAM++.onnx")

Remove-Item -Recurse -Force $TmpDir -ErrorAction SilentlyContinue

Write-Output ""
Write-Output "=== Done. Layout under $ModelsDir : ==="
Get-ChildItem -Recurse $ModelsDir | Select-Object FullName, Length | Format-Table -AutoSize
