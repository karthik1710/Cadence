<#
.SYNOPSIS
  Fills vendor/runtimes/ with the native runtimes a Cadence installer ships.

.DESCRIPTION
  Runs as Tauri's build.beforeBundleCommand, so every `tauri build` (and
  `npm run release`) bundles whisper.cpp, llama.cpp and sherpa-onnx inside the
  signed installer instead of the app downloading executables at first run.
  It can also be run by hand; it is idempotent.

  For each runtime it:

    1. Fetches the pinned upstream release archive into vendor/cache/ - only
       when it is missing or its SHA-256 does not match the pin - and refuses
       to go on if the archive does not hash to the pinned value.
    2. Unpacks it with Windows' own tar.exe into a scratch folder.
    3. Keeps only what the app runs: every .dll in the archive, plus the .exe
       files Cadence actually launches. Test and example executables are
       dropped.
    4. Signs every kept .exe/.dll through scripts/sign.ps1, which keeps any
       valid upstream signature (none of these files has one today), warns
       when no signing identity is configured, and fails under
       CADENCE_REQUIRE_SIGNING=1 (`npm run release`). Signing happens here,
       not only in the Tauri bundler, because the manifest below must describe
       the signed bytes; the bundler then finds them validly signed and leaves
       them alone.
    5. After signing, writes vendor/runtimes/manifest.json - the SHA-256 of
       every shipped file - which the app's Verify checks the installed copy
       against.

  It also copies src-tauri/THIRD_PARTY_NOTICES.txt beside them, so the license
  texts travel with the binaries.

  Everything is assembled in vendor/cache/staging and only swapped into
  vendor/runtimes/ once complete, so a failed run never leaves a half-filled
  runtime folder for a later build to bundle. vendor/runtimes/README.md is
  committed and always kept.

  Nothing downloaded here is ever executed by this script.

  Exit code is non-zero on any failure.
#>
[CmdletBinding()]
param()

# Started as Windows PowerShell 5.1 from a PowerShell 7 parent (GitHub
# Actions' pwsh steps, or `npm run tauri build` typed in a pwsh terminal), this
# process inherits pwsh's PSModulePath and picks up 7.x builds of the core
# modules it cannot load, so cmdlets such as Get-FileHash vanish. Use Windows
# PowerShell's own module path instead.
if ($PSVersionTable.PSEdition -eq 'Desktop') {
    $env:PSModulePath = [Environment]::ExpandEnvironmentVariables(
        [Environment]::GetEnvironmentVariable('PSModulePath', 'Machine'))
}

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2.0

$repo = Split-Path -Parent $PSScriptRoot
$vendor = Join-Path $repo 'vendor'
$cache = Join-Path $vendor 'cache'
$staging = Join-Path $cache 'staging'
$scratch = Join-Path $cache 'extract'
$output = Join-Path $vendor 'runtimes'
$signScript = Join-Path $PSScriptRoot 'sign.ps1'
$noticesSource = Join-Path $repo 'src-tauri\THIRD_PARTY_NOTICES.txt'
$keepInOutput = @('README.md')

# Windows' own tools. Git Bash's tar misreads a `C:` path as a remote host, so
# never rely on whichever tar happens to be first on PATH.
$tar = Join-Path $env:SystemRoot 'System32\tar.exe'
$curl = Join-Path $env:SystemRoot 'System32\curl.exe'

# The pins. Each archive is the exact one the app's catalog
# (src-tauri/src/models_manager.rs) names, pinned by URL and by the SHA-256
# of the archive itself. `Launch` lists the executables Cadence runs from that
# runtime (pipeline/paths.rs, pipeline/denoise.rs, pipeline/vad.rs); the first
# is the launcher whose presence decides that the runtime is installed.
# Changing a version means changing the URL *and* the hash here, the catalog
# entry, and re-running the baseline test (docs/benchmarks.md).
$runtimes = @(
    [ordered]@{
        Name    = 'whisper'
        Version = 'whisper.cpp b4938'
        Url     = 'https://github.com/ggml-org/whisper.cpp/releases/download/b4938/whisper-bin-x64.zip'
        Archive = 'whisper-bin-x64-b4938.zip'
        Sha256  = 'c2a4b60edb11f7e11a9191ffb50929535527d4d91c9903dbe3e554583bbbc63d'
        Launch  = @('whisper-cli.exe', 'whisper-server.exe')
    },
    [ordered]@{
        Name    = 'llama'
        Version = 'llama.cpp b10717'
        Url     = 'https://github.com/ggml-org/llama.cpp/releases/download/b10717/llama-b10717-bin-win-cpu-x64.zip'
        Archive = 'llama-b10717-bin-win-cpu-x64.zip'
        Sha256  = 'a8c6f95fc6b2f690ebbcabc8bba72cd0ba428353eca0a663556ab515f9203593'
        Launch  = @('llama-server.exe')
    },
    [ordered]@{
        Name    = 'sherpa'
        Version = 'sherpa-onnx v1.13.6'
        Url     = 'https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.13.6/sherpa-onnx-v1.13.6-win-x64-shared-MD-Release-no-tts.tar.bz2'
        Archive = 'sherpa-onnx-v1.13.6-win-x64-shared-MD-Release-no-tts.tar.bz2'
        Sha256  = '071d6641efd737a1f60de48c9c4cd596f78d5b0980815e8ad3798c95785d2b26'
        Launch  = @(
            'sherpa-onnx-offline-speaker-diarization.exe',
            'sherpa-onnx-offline-denoiser.exe',
            'sherpa-onnx-vad.exe'
        )
    }
)

function Write-Step([string]$message) {
    Write-Host "prepare-runtimes: $message"
}

function Get-Sha256([string]$path) {
    (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Remove-IfPresent([string]$path) {
    if (Test-Path -LiteralPath $path) {
        Remove-Item -LiteralPath $path -Recurse -Force
    }
}

function Write-Utf8NoBom([string]$path, [string]$text) {
    [System.IO.File]::WriteAllText($path, $text, (New-Object System.Text.UTF8Encoding $false))
}

# Returns the path of a verified copy of the runtime's archive in the cache,
# downloading it only when the cached copy is missing or does not match.
function Get-PinnedArchive($runtime) {
    $target = Join-Path $cache $runtime.Archive
    if (Test-Path -LiteralPath $target -PathType Leaf) {
        $cached = Get-Sha256 $target
        if ($cached -eq $runtime.Sha256) {
            Write-Step "$($runtime.Version): cached archive verified ($($runtime.Archive))."
            return $target
        }
        Write-Step "$($runtime.Version): cached archive does not match its pin (got $cached); downloading it again."
        Remove-Item -LiteralPath $target -Force
    }

    if (-not (Test-Path -LiteralPath $curl -PathType Leaf)) {
        throw "curl.exe not found at $curl (it ships with Windows 10 1803 and later)."
    }
    $partial = "$target.partial"
    Remove-IfPresent $partial
    Write-Step "$($runtime.Version): downloading $($runtime.Url)"
    # --disable must come first; it keeps a local .curlrc from weakening TLS.
    & $curl --disable --proto '=https' --proto-redir '=https' -L --fail --silent --show-error -o $partial $runtime.Url
    if ($LASTEXITCODE -ne 0) {
        Remove-IfPresent $partial
        throw "download of $($runtime.Url) failed (curl exit $LASTEXITCODE)."
    }
    $actual = Get-Sha256 $partial
    if ($actual -ne $runtime.Sha256) {
        Remove-IfPresent $partial
        throw ("$($runtime.Archive) does not match its pinned SHA-256.`n" +
            "  expected $($runtime.Sha256)`n  actual   $actual`n" +
            "Refusing to bundle it. If upstream re-published the release on purpose, " +
            "re-verify it and update the pin in scripts/prepare-runtimes.ps1.")
    }
    Move-Item -LiteralPath $partial -Destination $target
    Write-Step "$($runtime.Version): archive verified ($actual)."
    return $target
}

# Unpacks the archive and copies the files the app runs into $destination,
# flattened (the app runs each tool from its runtime's own folder, with its
# DLLs beside it).
function Copy-RuntimeFiles($runtime, [string]$archive, [string]$destination) {
    $extract = Join-Path $scratch $runtime.Name
    Remove-IfPresent $extract
    New-Item -ItemType Directory -Path $extract -Force | Out-Null

    & $tar -xf $archive -C $extract
    if ($LASTEXITCODE -ne 0) {
        throw "tar.exe could not unpack $archive (exit $LASTEXITCODE)."
    }

    $selected = @{}
    $dropped = 0
    foreach ($file in Get-ChildItem -LiteralPath $extract -Recurse -File) {
        $name = $file.Name
        $extension = $file.Extension.ToLowerInvariant()
        $keep = ($extension -eq '.dll') -or
            (($extension -eq '.exe') -and ($runtime.Launch -contains $name))
        if (-not $keep) {
            if ($extension -eq '.exe') { $dropped++ }
            continue
        }
        $key = $name.ToLowerInvariant()
        if ($selected.ContainsKey($key)) {
            # The same DLL can appear twice in one archive (sherpa-onnx ships
            # onnxruntime.dll in bin/ and lib/). Flattening is only safe when
            # the copies are identical.
            if ((Get-Sha256 $selected[$key].FullName) -ne (Get-Sha256 $file.FullName)) {
                throw "$($runtime.Archive) contains two different files named $name; refusing to guess which one to ship."
            }
            continue
        }
        $selected[$key] = $file
    }

    foreach ($exe in $runtime.Launch) {
        if (-not $selected.ContainsKey($exe.ToLowerInvariant())) {
            throw "$($runtime.Archive) does not contain $exe, which Cadence launches."
        }
    }

    New-Item -ItemType Directory -Path $destination -Force | Out-Null
    foreach ($file in $selected.Values) {
        Copy-Item -LiteralPath $file.FullName -Destination (Join-Path $destination $file.Name)
    }
    Remove-IfPresent $extract
    Write-Step "$($runtime.Version): kept $($selected.Count) files ($($runtime.Launch.Count) executables + DLLs), dropped $dropped other executables."
}

# Signs one file through scripts/sign.ps1 in a child PowerShell, so its `exit`
# cannot end this script, and fails on a non-zero exit.
function Invoke-Sign([string]$path) {
    & powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -File $signScript $path
    if ($LASTEXITCODE -ne 0) {
        throw "sign.ps1 failed for $path (exit $LASTEXITCODE)."
    }
}

try {
    foreach ($tool in @($tar, $signScript, $noticesSource)) {
        if (-not (Test-Path -LiteralPath $tool -PathType Leaf)) {
            throw "required file not found: $tool"
        }
    }

    New-Item -ItemType Directory -Path $cache -Force | Out-Null
    Remove-IfPresent $staging
    Remove-IfPresent $scratch
    New-Item -ItemType Directory -Path $staging -Force | Out-Null

    $manifest = [ordered]@{}
    $summary = @()
    foreach ($runtime in $runtimes) {
        $archive = Get-PinnedArchive $runtime
        $folder = Join-Path $staging $runtime.Name
        Copy-RuntimeFiles $runtime $archive $folder

        $files = @(Get-ChildItem -LiteralPath $folder -File | Sort-Object Name)
        Write-Step "$($runtime.Version): signing $($files.Count) files."
        foreach ($file in $files) {
            Invoke-Sign $file.FullName
        }

        # Digests are taken only now, after signing: they must describe the
        # bytes the installer ships.
        $digests = [ordered]@{}
        $unsigned = @()
        $bytes = 0
        foreach ($file in $files) {
            $digests[$file.Name] = Get-Sha256 $file.FullName
            $bytes += $file.Length
            if ((Get-AuthenticodeSignature -LiteralPath $file.FullName).Status -ne 'Valid') {
                $unsigned += $file.Name
            }
        }
        $manifest[$runtime.Name] = [ordered]@{
            version        = $runtime.Version
            source_url     = $runtime.Url
            archive_sha256 = $runtime.Sha256
            files          = $digests
        }
        $summary += [pscustomobject]@{
            Runtime  = $runtime.Name
            Version  = $runtime.Version
            Files    = $files.Count
            MB       = [math]::Round($bytes / 1MB, 1)
            Unsigned = $unsigned.Count
            UnsignedFiles = ($unsigned -join ', ')
        }
    }

    Write-Utf8NoBom (Join-Path $staging 'manifest.json') (($manifest | ConvertTo-Json -Depth 6) + "`n")
    Copy-Item -LiteralPath $noticesSource -Destination (Join-Path $staging 'THIRD_PARTY_NOTICES.txt')

    # Swap the finished set in: everything in vendor/runtimes except the
    # committed README goes, so no file from an earlier run survives.
    New-Item -ItemType Directory -Path $output -Force | Out-Null
    foreach ($item in Get-ChildItem -LiteralPath $output -Force) {
        if ($keepInOutput -notcontains $item.Name) {
            Remove-Item -LiteralPath $item.FullName -Recurse -Force
        }
    }
    foreach ($item in Get-ChildItem -LiteralPath $staging -Force) {
        Move-Item -LiteralPath $item.FullName -Destination (Join-Path $output $item.Name)
    }
    Remove-IfPresent $staging
    Remove-IfPresent $scratch

    Write-Step 'done.'
    $summary | Format-Table Runtime, Version, Files, MB, Unsigned -AutoSize | Out-String | Write-Host
    $unsignedTotal = ($summary | Measure-Object -Property Unsigned -Sum).Sum
    if ($unsignedTotal -gt 0) {
        Write-Warning ("prepare-runtimes: $unsignedTotal bundled file(s) carry no valid signature " +
            '(no signing identity configured?). Fine for a local test build; npm run release refuses this.')
        foreach ($row in $summary) {
            if ($row.Unsigned -gt 0) { Write-Host "  $($row.Runtime): $($row.UnsignedFiles)" }
        }
    }
    exit 0
}
catch {
    Write-Host "prepare-runtimes: FAILED: $($_.Exception.Message)" -ForegroundColor Red
    Remove-IfPresent $staging
    Remove-IfPresent $scratch
    exit 1
}
