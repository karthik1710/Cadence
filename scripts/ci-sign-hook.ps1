<#
.SYNOPSIS
  Tauri's signCommand for CI release builds: defers signing to SignPath.

.DESCRIPTION
  The CI release workflow (.github/workflows/release.yml) points
  bundle.windows.signCommand at this script through the overlay
  src-tauri/tauri.ci.conf.json. The local flow (`npm run tauri build`,
  `npm run release`) never uses it; it keeps scripts/sign.ps1.

  Why it exists. Tauri signs each file synchronously while it bundles: the
  main executable (after patching it for the NSIS package type), every
  .exe/.dll resource without a valid signature, the NSIS plugin DLLs, the
  uninstaller (from inside makensis, through !uninstfinalize) and the finished
  installer. SignPath signs a whole GitHub artifact asynchronously, behind a
  manual approval, so it cannot be called once per file. The workflow
  therefore bundles twice:

    collect  First `tauri build`. Every file Tauri asks to sign is left
             unsigned. Cadence's own binaries - the patched Cadence.exe and the
             uninstaller - are copied to $CADENCE_CI_SIGN_DIR\unsigned\ under
             fixed names, and the SHA-256 of each is recorded in
             $CADENCE_CI_SIGN_DIR\unsigned-map.json. The installer this pass
             produces is the unsigned build.

    apply    Second pass, `tauri bundle`, from the same unpatched Cadence.exe,
             after SignPath signed the collected files into
             $CADENCE_CI_SIGN_DIR\signed\. When Tauri asks to sign a file whose
             SHA-256 is in the map, the signed copy replaces it. This works
             because both are reproducible: Tauri's patch of the main binary is
             a fixed byte substitution, and makensis writes a byte-identical
             uninstaller for identical inputs (checked with NSIS 3.11 on
             2026-11-20). If a hash does not match, the bytes SignPath signed
             are not the bytes being shipped, and under CADENCE_CI_SIGN_STRICT=1
             that fails the build instead of shipping an unsigned file.
             The installer itself is left alone: the workflow sends it to
             SignPath as a separate request once it exists.
             Each replacement is appended to $CADENCE_CI_SIGN_DIR\applied.txt.
             makensis ignores the exit code of !uninstfinalize, so a failure
             while it runs this script for the uninstaller would not stop the
             build; the workflow checks applied.txt afterwards instead.

  What is never signed here. SignPath Foundation's terms allow a project to
  sign only artifacts built from its own source; upstream binaries "must not
  be signed" under the project's subscription, but may ship unsigned inside a
  signed installer (https://signpath.org/terms). So the bundled runtimes
  (vendor\runtimes\: whisper.cpp, llama.cpp, sherpa-onnx) and NSIS's own plugin
  DLLs are always left as they are. That also keeps runtimes\manifest.json,
  written by scripts/prepare-runtimes.ps1, describing the shipped bytes.

  Any file this script cannot classify fails the build: it means Tauri started
  signing something new, and this script has to be taught what it is.

  Environment:
    CADENCE_CI_SIGN_MODE        collect (default) or apply.
    CADENCE_CI_SIGN_DIR         Working folder. Default:
                                src-tauri\target\ci-sign (gitignored).
    CADENCE_CI_SIGN_STRICT=1    apply: fail when one of Cadence's binaries has
                                no signed counterpart.
    CADENCE_CI_EXPECTED_SIGNER  apply: a substring the signer certificate's
                                subject must contain (e.g. SignPath Foundation).
    CADENCE_CI_MAIN_BINARY      Main executable's file name. Default Cadence.exe.
#>
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Path
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2.0

$repo = Split-Path -Parent $PSScriptRoot
$mode = if ($env:CADENCE_CI_SIGN_MODE) { $env:CADENCE_CI_SIGN_MODE.ToLowerInvariant() } else { 'collect' }
$dir = if ($env:CADENCE_CI_SIGN_DIR) { $env:CADENCE_CI_SIGN_DIR } else { Join-Path $repo 'src-tauri\target\ci-sign' }
$strict = $env:CADENCE_CI_SIGN_STRICT -eq '1'
$mainBinary = if ($env:CADENCE_CI_MAIN_BINARY) { $env:CADENCE_CI_MAIN_BINARY } else { 'Cadence.exe' }
$mapPath = Join-Path $dir 'unsigned-map.json'

function Write-Hook([string]$message) {
    Write-Host "ci-sign-hook[$mode]: $message"
}

function Stop-Hook([string]$message) {
    # `exit 1` alone is enough for Tauri and makensis; Write-Host keeps the
    # reason in the log even when the caller swallows stderr.
    Write-Host "ci-sign-hook[$mode]: ERROR: $message"
    exit 1
}

function Get-Sha256([string]$file) {
    (Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Read-Map {
    if (-not (Test-Path -LiteralPath $mapPath -PathType Leaf)) { return @{} }
    $map = @{}
    $parsed = Get-Content -LiteralPath $mapPath -Raw | ConvertFrom-Json
    foreach ($property in $parsed.PSObject.Properties) { $map[$property.Name] = [string]$property.Value }
    return $map
}

function Write-Map($map) {
    $ordered = [ordered]@{}
    foreach ($key in ($map.Keys | Sort-Object)) { $ordered[$key] = $map[$key] }
    $json = ($ordered | ConvertTo-Json) + "`n"
    [System.IO.File]::WriteAllText($mapPath, $json, (New-Object System.Text.UTF8Encoding $false))
}

# What kind of file Tauri is asking to sign. Paths are matched on Windows'
# separators because that is what Tauri passes on this platform.
function Get-FileKind([string]$file) {
    $name = Split-Path -Leaf $file
    $parent = Split-Path -Leaf (Split-Path -Parent $file)
    if ($file -match '\\vendor\\runtimes\\') { return 'upstream-runtime' }
    if ($file -match '\\Plugins\\x86-unicode\\') { return 'upstream-nsis-plugin' }
    if ($name -like '*-setup.exe' -and $parent -eq 'nsis') { return 'installer' }
    if ($name -ieq $mainBinary) { return 'main' }
    # makensis hands !uninstfinalize a temporary file, %TEMP%\nstXXXX.tmp.
    if ($name -like 'nst*.tmp') { return 'uninstaller' }
    return 'unknown'
}

if ($mode -ne 'collect' -and $mode -ne 'apply') {
    Stop-Hook "CADENCE_CI_SIGN_MODE must be 'collect' or 'apply', not '$mode'."
}
if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
    Stop-Hook "no such file: $Path"
}
$file = (Resolve-Path -LiteralPath $Path).Path
$name = Split-Path -Leaf $file
$kind = Get-FileKind $file

$existing = Get-AuthenticodeSignature -LiteralPath $file
if ($existing.Status -eq 'Valid') {
    Write-Hook "$name already has a valid signature ($($existing.SignerCertificate.Subject)); keeping it."
    exit 0
}

switch ($kind) {
    'upstream-runtime' {
        Write-Hook "$name is an upstream runtime binary; left unsigned (SignPath Foundation signs only this project's own binaries)."
        exit 0
    }
    'upstream-nsis-plugin' {
        Write-Hook "$name is an NSIS plugin; left unsigned (upstream binary)."
        exit 0
    }
    'installer' {
        Write-Hook "$name is the installer; left unsigned here. The workflow submits it to SignPath on its own."
        exit 0
    }
    'unknown' {
        Stop-Hook ("Tauri asked to sign $file, which this script does not recognise. " +
            'Classify it in scripts/ci-sign-hook.ps1 (own binary to sign, or upstream binary to leave unsigned).')
    }
}

# From here on: one of Cadence's own binaries (main or uninstaller).
$fixedName = if ($kind -eq 'main') { $mainBinary } else { 'uninstall.exe' }
$hash = Get-Sha256 $file

if ($mode -eq 'collect') {
    $unsignedDir = Join-Path $dir 'unsigned'
    New-Item -ItemType Directory -Path $unsignedDir -Force | Out-Null
    $map = Read-Map
    foreach ($entry in @($map.GetEnumerator())) {
        if ($entry.Value -eq $fixedName -and $entry.Key -ne $hash) {
            Stop-Hook ("asked to sign two different $kind files in one build ($($entry.Key) and $hash). " +
                'The SignPath artifact holds one file per name; this needs a change to the workflow.')
        }
    }
    Copy-Item -LiteralPath $file -Destination (Join-Path $unsignedDir $fixedName) -Force
    $map[$hash] = $fixedName
    Write-Map $map
    Write-Hook "collected $kind $name as $fixedName (sha256 $hash); left unsigned in this pass."
    exit 0
}

# apply
$map = Read-Map
if (-not $map.ContainsKey($hash)) {
    $message = ("$kind $name (sha256 $hash) does not match any file that was submitted for signing " +
        "($mapPath). The build is not reproducible between the two passes, so the signed copy cannot be used.")
    if ($strict) { Stop-Hook $message }
    Write-Hook "WARNING: $message Leaving it unsigned."
    exit 0
}
$signedName = $map[$hash]
$signed = Join-Path (Join-Path $dir 'signed') $signedName
if (-not (Test-Path -LiteralPath $signed -PathType Leaf)) {
    $message = "the signed copy of $signedName is missing ($signed)."
    if ($strict) { Stop-Hook $message }
    Write-Hook "WARNING: $message Leaving $name unsigned."
    exit 0
}
$signature = Get-AuthenticodeSignature -LiteralPath $signed
if ($signature.Status -ne 'Valid') {
    Stop-Hook "the signed copy of $signedName is not validly signed: $($signature.Status) ($($signature.StatusMessage))."
}
$subject = $signature.SignerCertificate.Subject
if ($env:CADENCE_CI_EXPECTED_SIGNER -and ($subject -notlike "*$($env:CADENCE_CI_EXPECTED_SIGNER)*")) {
    Stop-Hook "the signed copy of $signedName is signed by '$subject', not by '$($env:CADENCE_CI_EXPECTED_SIGNER)'."
}
if ($null -eq $signature.TimeStamperCertificate) {
    Stop-Hook "the signed copy of $signedName carries no timestamp; it would stop validating when the certificate expires."
}

Copy-Item -LiteralPath $signed -Destination $file -Force
if ((Get-Sha256 $file) -ne (Get-Sha256 $signed)) {
    Stop-Hook "copying the signed $signedName over $file did not take."
}
# makensis runs !uninstfinalize without checking its exit code, so a failure
# above would not stop the build when this runs for the uninstaller. The
# workflow therefore checks this record afterwards and fails unless every
# collected file was replaced.
Add-Content -LiteralPath (Join-Path $dir 'applied.txt') -Value "$signedName $hash" -Encoding ascii
Write-Hook "replaced $kind $name with the SignPath-signed $signedName ($subject)."
exit 0
