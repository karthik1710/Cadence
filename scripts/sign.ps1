<#
.SYNOPSIS
  Authenticode-signs one file for a Cadence release build.

.DESCRIPTION
  Called by `tauri build` (bundle.windows.signCommand) once per binary it
  signs - the app executable, the NSIS uninstaller and the installer - and by
  scripts/prepare-runtimes.ps1 for each bundled runtime executable and DLL.

  Nothing secret lives in the repository. The signing identity comes from the
  environment, in this order:

    CADENCE_SIGN_CERT_THUMBPRINT     SHA-1 thumbprint of a code-signing
                                     certificate in CurrentUser\My or
                                     LocalMachine\My (OV/EV certificate, or a
                                     hardware token that exposes one there).

    CADENCE_SIGN_DLIB +              Azure Trusted Signing: path to
    CADENCE_SIGN_DLIB_METADATA       Azure.CodeSigning.Dlib.dll and to its
                                     metadata.json (endpoint, account, profile).

  Optional:
    CADENCE_SIGN_TIMESTAMP_URL       RFC 3161 timestamp server. Defaults to
                                     DigiCert, or Microsoft's for Trusted
                                     Signing. Timestamping keeps a signature
                                     valid after the certificate expires.
    CADENCE_SIGNTOOL                 Full path to signtool.exe. Otherwise the
                                     newest x64 signtool in the Windows SDK.
    CADENCE_REQUIRE_SIGNING=1        Fail instead of warning when no identity
                                     is configured. `npm run release` sets it,
                                     so a release can never ship unsigned.

  A file that already carries a valid signature is left as it is: an upstream
  binary its publisher signed keeps that publisher's signature and
  reputation. (None of the runtime files Cadence bundles today is
  upstream-signed - not even onnxruntime.dll, which sherpa-onnx builds itself -
  so all of them take Cadence's signature.)
#>
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Path
)

$ErrorActionPreference = 'Stop'

if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
    throw "sign.ps1: no such file: $Path"
}
$file = (Resolve-Path -LiteralPath $Path).Path
$name = Split-Path -Leaf $file

$existing = Get-AuthenticodeSignature -LiteralPath $file
if ($existing.Status -eq 'Valid') {
    Write-Host "sign.ps1: $name already has a valid signature ($($existing.SignerCertificate.Subject)); keeping it."
    exit 0
}

$thumbprint = $env:CADENCE_SIGN_CERT_THUMBPRINT
$dlib = $env:CADENCE_SIGN_DLIB
$dlibMetadata = $env:CADENCE_SIGN_DLIB_METADATA
$require = $env:CADENCE_REQUIRE_SIGNING -eq '1'

if (-not $thumbprint -and -not ($dlib -and $dlibMetadata)) {
    $message = "sign.ps1: no signing identity configured; $name is NOT signed. " +
        "Set CADENCE_SIGN_CERT_THUMBPRINT, or CADENCE_SIGN_DLIB and CADENCE_SIGN_DLIB_METADATA (see docs/release.md)."
    if ($require) {
        Write-Error $message
        exit 1
    }
    Write-Warning $message
    exit 0
}

function Find-SignTool {
    if ($env:CADENCE_SIGNTOOL) {
        if (Test-Path -LiteralPath $env:CADENCE_SIGNTOOL) { return $env:CADENCE_SIGNTOOL }
        throw "sign.ps1: CADENCE_SIGNTOOL points at a missing file: $($env:CADENCE_SIGNTOOL)"
    }
    $kits = Join-Path ${env:ProgramFiles(x86)} 'Windows Kits\10\bin'
    if (Test-Path $kits) {
        $found = Get-ChildItem -Path $kits -Filter signtool.exe -Recurse -ErrorAction SilentlyContinue |
            Where-Object { $_.Directory.Name -eq 'x64' } |
            Sort-Object { $_.Directory.Parent.Name } -Descending |
            Select-Object -First 1
        if ($found) { return $found.FullName }
    }
    $onPath = Get-Command signtool.exe -ErrorAction SilentlyContinue
    if ($onPath) { return $onPath.Source }
    throw 'sign.ps1: signtool.exe not found. Install the Windows SDK signing tools or set CADENCE_SIGNTOOL.'
}

$signtool = Find-SignTool
$arguments = @('sign', '/fd', 'SHA256', '/td', 'SHA256', '/d', 'Cadence')

if ($thumbprint) {
    $timestamp = if ($env:CADENCE_SIGN_TIMESTAMP_URL) { $env:CADENCE_SIGN_TIMESTAMP_URL } else { 'http://timestamp.digicert.com' }
    $arguments += @('/sha1', $thumbprint, '/tr', $timestamp)
} else {
    $timestamp = if ($env:CADENCE_SIGN_TIMESTAMP_URL) { $env:CADENCE_SIGN_TIMESTAMP_URL } else { 'http://timestamp.acs.microsoft.com' }
    $arguments += @('/dlib', $dlib, '/dmdf', $dlibMetadata, '/tr', $timestamp)
}
$arguments += $file

# Timestamp servers are occasionally unavailable; retry before failing.
$attempt = 0
while ($true) {
    $attempt++
    & $signtool @arguments
    if ($LASTEXITCODE -eq 0) { break }
    if ($attempt -ge 3) { throw "sign.ps1: signtool failed to sign $name (exit $LASTEXITCODE)." }
    Start-Sleep -Seconds (5 * $attempt)
}

$result = Get-AuthenticodeSignature -LiteralPath $file
if ($result.Status -eq 'NotSigned') {
    throw "sign.ps1: signtool reported success but $name carries no signature."
}
if ($result.Status -ne 'Valid') {
    # A signature from a certificate this machine does not trust (a test
    # certificate) is present but not Valid. A release must chain to a
    # trusted root, so fail there; a test build only warns.
    $message = "sign.ps1: $name is signed, but the signature is $($result.Status): $($result.StatusMessage)"
    if ($require) { Write-Error $message; exit 1 }
    Write-Warning $message
}
Write-Host "sign.ps1: signed $name ($($result.SignerCertificate.Subject))."
