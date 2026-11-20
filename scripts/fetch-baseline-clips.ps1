<#
.SYNOPSIS
  Downloads the baseline regression clips listed in tests/baseline/clips.json
  into bench-runs/baseline-cache/, for `bench_cli baseline`.

.DESCRIPTION
  For each clip it writes:
    bench-runs/baseline-cache/<id>.16k.wav   16 kHz mono 16-bit PCM audio
    bench-runs/baseline-cache/<id>.vtt       the clip's caption track
  and skips any clip whose two files already exist (-Force refetches).

  Only human-made subtitle tracks are fetched (--write-subs, never
  --write-auto-subs): a reference produced by YouTube's own ASR would measure
  agreement with another recogniser, not accuracy. Caption files are written
  as yt-dlp produces them (UTF-8), so Tamil-script tracks need no special
  handling; the Rust VTT parser is tested on Tamil and code-switched text.

  The audio and captions are COPYRIGHTED YOUTUBE CONTENT. They are cached on
  this machine only, for local regression testing, and must never be
  committed: /bench-runs/ is in .gitignore. Do not force-add it.

  Portable yt-dlp and ffmpeg/ffprobe are downloaded into
  bench-runs/baseline-cache/tools/ on first use. Nothing is installed system
  wide. yt-dlp needs a JavaScript runtime for YouTube's challenge, so node.js
  must be on PATH.

  This is a dev-time script. The app and bench_cli never touch the network for
  any of this; bench_cli only reads the cache.

.PARAMETER Force
  Re-download every clip even if its cached files exist.

.PARAMETER Clips
  Only fetch these clip ids (default: every clip in the manifest).

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\fetch-baseline-clips.ps1
  powershell -ExecutionPolicy Bypass -File scripts\fetch-baseline-clips.ps1 -Clips rXRqgTI7LgU -Force
#>
param(
    [switch]$Force,
    [string[]]$Clips
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$Manifest = Join-Path $RepoRoot "tests\baseline\clips.json"
$Cache = Join-Path $RepoRoot "bench-runs\baseline-cache"
$Tools = Join-Path $Cache "tools"
$YtDlp = Join-Path $Tools "yt-dlp.exe"
$Ffmpeg = Join-Path $Tools "ffmpeg.exe"
$Ffprobe = Join-Path $Tools "ffprobe.exe"

$YtDlpUrl = "https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp.exe"
$FfmpegUrl = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"

if (-not (Test-Path $Manifest)) {
    throw "Manifest not found: $Manifest"
}
if ($null -eq (Get-Command node -ErrorAction SilentlyContinue)) {
    throw "node.js was not found on PATH. yt-dlp needs a JavaScript runtime to pass YouTube's challenge. Install node.js (e.g. 'winget install OpenJS.NodeJS.LTS'), open a new terminal, and re-run this script."
}

New-Item -ItemType Directory -Force -Path $Tools | Out-Null

# Windows PowerShell 5.1 defaults to older TLS versions that GitHub refuses.
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
# The progress bar makes Invoke-WebRequest many times slower on 5.1.
$ProgressPreference = "SilentlyContinue"

function Invoke-Native {
    param([string]$What, [string]$Exe, [string[]]$Arguments)
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$What failed (exit code $LASTEXITCODE)."
    }
}

if (-not (Test-Path $YtDlp)) {
    Write-Output "Downloading yt-dlp -> $YtDlp"
    Invoke-WebRequest -UseBasicParsing -Uri $YtDlpUrl -OutFile $YtDlp
}
if (-not ((Test-Path $Ffmpeg) -and (Test-Path $Ffprobe))) {
    Write-Output "Downloading ffmpeg (portable build) -> $Tools"
    $Zip = Join-Path $Tools "ffmpeg.zip"
    $Unpacked = Join-Path $Tools "ffmpeg-unpacked"
    Invoke-WebRequest -UseBasicParsing -Uri $FfmpegUrl -OutFile $Zip
    if (Test-Path $Unpacked) { Remove-Item -Recurse -Force $Unpacked }
    Expand-Archive -Path $Zip -DestinationPath $Unpacked
    foreach ($exe in @("ffmpeg.exe", "ffprobe.exe")) {
        $found = Get-ChildItem -Path $Unpacked -Recurse -Filter $exe | Select-Object -First 1
        if ($null -eq $found) { throw "$exe not found in the ffmpeg archive." }
        Copy-Item -Force $found.FullName (Join-Path $Tools $exe)
    }
    Remove-Item -Recurse -Force $Unpacked
    Remove-Item -Force $Zip
}

$entries = (Get-Content -Raw -Encoding UTF8 $Manifest | ConvertFrom-Json).clips
if ($Clips) {
    $unknown = @($Clips | Where-Object { $id = $_; -not ($entries | Where-Object { $_.id -eq $id }) })
    if ($unknown.Count -gt 0) {
        throw "Unknown clip id(s): $($unknown -join ', '). Known: $(($entries | ForEach-Object { $_.id }) -join ', ')"
    }
    $entries = @($entries | Where-Object { $Clips -contains $_.id })
}

$failed = @()
foreach ($clip in $entries) {
    $id = $clip.id
    $wav16k = Join-Path $Cache "$id.16k.wav"
    $vtt = Join-Path $Cache "$id.vtt"
    Write-Output ""
    Write-Output "== $id - $($clip.title)"
    if ((Test-Path $wav16k) -and (Test-Path $vtt) -and -not $Force) {
        Write-Output "   cached, skipping (use -Force to refetch)"
        continue
    }
    try {
        # yt-dlp writes <id>.wav and <id>.<track>.vtt next to each other.
        $rawWav = Join-Path $Cache "$id.wav"
        $rawVtt = Join-Path $Cache "$id.$($clip.caption_track).vtt"
        foreach ($stale in @($rawWav, $rawVtt, $wav16k, $vtt)) {
            if (Test-Path $stale) { Remove-Item -Force $stale }
        }
        Invoke-Native "yt-dlp ($id)" $YtDlp @(
            "--ffmpeg-location", $Tools,
            "--js-runtimes", "node",
            "--remote-components", "ejs:github",
            "-f", "ba/b",
            "-x", "--audio-format", "wav",
            "--write-subs", "--sub-langs", $clip.caption_track, "--sub-format", "vtt",
            "--no-playlist",
            "-P", $Cache,
            "-o", "$id.%(ext)s",
            $clip.url
        )
        if (-not (Test-Path $rawWav)) { throw "yt-dlp produced no audio at $rawWav" }
        if (-not (Test-Path $rawVtt)) {
            throw "yt-dlp produced no '$($clip.caption_track)' caption track at $rawVtt. The track id may have changed; list them with: `"$YtDlp`" --list-subs $($clip.url)"
        }

        Invoke-Native "ffmpeg ($id)" $Ffmpeg @(
            "-hide_banner", "-loglevel", "error", "-y",
            "-i", $rawWav,
            "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
            $wav16k
        )
        Move-Item -Force $rawVtt $vtt
        Remove-Item -Force $rawWav

        $seconds = [double](& $Ffprobe -v error -show_entries format=duration -of csv=p=0 $wav16k)
        $expected = [double]$clip.duration_s
        Write-Output ("   wrote {0} ({1:N1} s) and {2}" -f $wav16k, $seconds, $vtt)
        if ([math]::Abs($seconds - $expected) -gt 2.0) {
            Write-Warning ("$id is {0:N1} s but the manifest says {1:N1} s. The upload may have changed; check before trusting its baseline." -f $seconds, $expected)
        }
    } catch {
        Write-Warning "$id failed: $_"
        $failed += $id
    }
}

Write-Output ""
if ($failed.Count -gt 0) {
    throw "Failed to fetch: $($failed -join ', ')"
}
Write-Output "Baseline clips cached in $Cache"
Write-Output "Next: cd src-tauri; cargo run --release --bin bench_cli -- baseline"
