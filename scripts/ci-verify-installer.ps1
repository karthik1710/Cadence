<#
.SYNOPSIS
  Checks a Cadence installer produced by the CI release workflow.

.DESCRIPTION
  Run by .github/workflows/release.yml. It checks, in order:

    1. With -SignDir: that scripts/ci-sign-hook.ps1 replaced every binary it
       collected for SignPath (Cadence.exe and the uninstaller) with its signed
       copy during `tauri bundle`. makensis ignores the exit code of the
       uninstaller-signing hook, so this record is the only reliable proof
       that the uninstaller inside the installer is the signed one.
    2. The installer's own Authenticode status: -Expect Signed requires a
       valid, timestamped signature (and, with -ExpectedSigner, a matching
       subject); -Expect Unsigned requires none, so an unsigned build can never
       be mislabelled as signed or the other way round. -ExpectContents
       (default: same as -Expect) is the expectation for Cadence.exe and the
       uninstaller inside it: after the second bundling pass the contents are
       signed while the installer is not yet.
    3. Best effort, when 7-Zip is available (it is on GitHub's Windows
       runners): unpacks the installer and checks that Cadence.exe carries the
       -ExpectContents signature state, that every file runtimes\manifest.json lists
       is present with that digest and nothing else is in the runtime folders
       (what the installed app's Verify will check), and that the developer
       CLIs (pipeline_cli, bench_cli) are not in it. If 7-Zip cannot unpack
       this NSIS version the step warns and skips this part.

  Exit code is non-zero on any failed check.
#>
param(
    [Parameter(Mandatory = $true)] [string]$Installer,
    [Parameter(Mandatory = $true)] [ValidateSet('Signed', 'Unsigned')] [string]$Expect,
    [ValidateSet('', 'Signed', 'Unsigned')] [string]$ExpectContents = '',
    [string]$SignDir,
    [string]$ExpectedSigner
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2.0

$failures = New-Object System.Collections.Generic.List[string]
function Fail([string]$message) { $failures.Add($message); Write-Host "ci-verify-installer: FAIL: $message" }
function Note([string]$message) { Write-Host "ci-verify-installer: $message" }
function Get-Sha256([string]$file) { (Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash.ToLowerInvariant() }

function Test-SignatureState([string]$file, [string]$label, [string]$expected) {
    $signature = Get-AuthenticodeSignature -LiteralPath $file
    if ($expected -eq 'Signed') {
        if ($signature.Status -ne 'Valid') {
            Fail "$label is not validly signed: $($signature.Status) ($($signature.StatusMessage))."
            return
        }
        if ($null -eq $signature.TimeStamperCertificate) { Fail "$label is signed but not timestamped." }
        $subject = $signature.SignerCertificate.Subject
        if ($ExpectedSigner -and ($subject -notlike "*$ExpectedSigner*")) {
            Fail "$label is signed by '$subject', expected '$ExpectedSigner'."
            return
        }
        Note "$label is signed by $subject."
    } else {
        if ($signature.Status -ne 'NotSigned') {
            Fail "$label was expected to be unsigned but its signature status is $($signature.Status)."
            return
        }
        Note "$label is unsigned, as expected for this build."
    }
}

if (-not (Test-Path -LiteralPath $Installer -PathType Leaf)) {
    throw "ci-verify-installer: no such installer: $Installer"
}
$Installer = (Resolve-Path -LiteralPath $Installer).Path
if (-not $ExpectContents) { $ExpectContents = $Expect }
Note "checking $Installer ($([math]::Round((Get-Item -LiteralPath $Installer).Length / 1MB, 1)) MB): installer expected $Expect, Cadence.exe and uninstaller inside it expected $ExpectContents."

# 1. Every collected binary was replaced by its signed copy.
if ($SignDir) {
    $mapPath = Join-Path $SignDir 'unsigned-map.json'
    $appliedPath = Join-Path $SignDir 'applied.txt'
    if (-not (Test-Path -LiteralPath $mapPath -PathType Leaf)) {
        Fail "no ${mapPath}: the collect pass recorded nothing to sign."
    } else {
        $collected = (Get-Content -LiteralPath $mapPath -Raw | ConvertFrom-Json).PSObject.Properties
        $applied = if (Test-Path -LiteralPath $appliedPath) { @(Get-Content -LiteralPath $appliedPath) } else { @() }
        foreach ($name in @('Cadence.exe', 'uninstall.exe')) {
            if (-not ($collected | Where-Object { $_.Value -eq $name })) { Fail "the collect pass never saw $name." }
        }
        foreach ($entry in $collected) {
            if ($applied -contains "$($entry.Value) $($entry.Name)") {
                Note "$($entry.Value) was replaced by its SignPath-signed copy during bundling."
            } else {
                Fail "$($entry.Value) (sha256 $($entry.Name)) was not replaced by its signed copy during bundling; the installer would ship it unsigned."
            }
        }
    }
}

# 2. The installer itself.
Test-SignatureState $Installer 'the installer' $Expect

# 3. Its contents, best effort.
$sevenZip = @(
    (Get-Command 7z.exe -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source -First 1),
    (Join-Path $env:ProgramFiles '7-Zip\7z.exe')
) | Where-Object { $_ -and (Test-Path -LiteralPath $_) } | Select-Object -First 1

if (-not $sevenZip) {
    Write-Host '::warning::ci-verify-installer: 7-Zip not found; installer contents were not checked.'
} else {
    $extract = Join-Path ([System.IO.Path]::GetTempPath()) ("cadence-verify-" + [guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $extract | Out-Null
    try {
        & $sevenZip x $Installer "-o$extract" -y | Out-Null
        $mainExe = Get-ChildItem -LiteralPath $extract -Recurse -File -Filter 'Cadence.exe' | Select-Object -First 1
        if ($LASTEXITCODE -ne 0 -or -not $mainExe) {
            Write-Host "::warning::ci-verify-installer: 7-Zip could not unpack the installer (exit $LASTEXITCODE); its contents were not checked."
        } else {
            Test-SignatureState $mainExe.FullName 'Cadence.exe inside the installer' $ExpectContents

            $uninstaller = Get-ChildItem -LiteralPath $extract -Recurse -File -Filter 'uninstall*.exe' | Select-Object -First 1
            if ($uninstaller) {
                Test-SignatureState $uninstaller.FullName 'the uninstaller inside the installer' $ExpectContents
            } else {
                Note 'the uninstaller is generated at install time and is not visible to 7-Zip; check 1 covers it.'
            }

            foreach ($cli in @('pipeline_cli.exe', 'bench_cli.exe')) {
                if (Get-ChildItem -LiteralPath $extract -Recurse -File -Filter $cli) {
                    Fail "$cli is in the installer; the developer CLIs must not ship (see the dev-cli feature in src-tauri/Cargo.toml)."
                }
            }

            $manifestFile = Get-ChildItem -LiteralPath $extract -Recurse -File -Filter 'manifest.json' |
                Where-Object { $_.Directory.Name -eq 'runtimes' } | Select-Object -First 1
            if (-not $manifestFile) {
                Fail 'runtimes\manifest.json is not in the installer.'
            } else {
                $root = $manifestFile.Directory.FullName
                $manifest = Get-Content -LiteralPath $manifestFile.FullName -Raw | ConvertFrom-Json
                $checked = 0
                foreach ($runtime in $manifest.PSObject.Properties) {
                    $folder = Join-Path $root $runtime.Name
                    $listed = @{}
                    foreach ($entry in $runtime.Value.files.PSObject.Properties) {
                        $listed[$entry.Name.ToLowerInvariant()] = $true
                        $path = Join-Path $folder $entry.Name
                        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
                            Fail "runtimes\$($runtime.Name)\$($entry.Name) is in the manifest but not in the installer."
                        } elseif ((Get-Sha256 $path) -ne $entry.Value) {
                            Fail "runtimes\$($runtime.Name)\$($entry.Name) does not match its manifest digest; the app's Verify would report it as changed."
                        } else {
                            $checked++
                        }
                    }
                    if (Test-Path -LiteralPath $folder) {
                        foreach ($file in Get-ChildItem -LiteralPath $folder -File) {
                            if (-not $listed.ContainsKey($file.Name.ToLowerInvariant())) {
                                Fail "runtimes\$($runtime.Name)\$($file.Name) is in the installer but not in the manifest."
                            }
                        }
                    }
                }
                Note "$checked runtime files match runtimes\manifest.json."
            }
        }
    } finally {
        Remove-Item -LiteralPath $extract -Recurse -Force -ErrorAction SilentlyContinue
    }
}

if ($failures.Count -gt 0) {
    Write-Host "ci-verify-installer: $($failures.Count) check(s) failed."
    exit 1
}
Note 'all checks passed.'
exit 0
