$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $PSScriptRoot
$AppName = if ($env:APP_NAME) { $env:APP_NAME } else { "odon" }
$McpName = if ($env:MCP_NAME) { $env:MCP_NAME } else { "odon_mcp" }
$Version = (Select-String -Path (Join-Path $RootDir "Cargo.toml") -Pattern '^version = "([^"]+)"').Matches[0].Groups[1].Value
$BundleDir = Join-Path $RootDir "target\release\bundle\windows"
$StageDir = Join-Path $BundleDir "$AppName-windows-x86_64"
$ArchivePath = Join-Path $BundleDir "$AppName-windows-x86_64.zip"
$InstallerDir = Join-Path $BundleDir "installer"
$InstallerPath = Join-Path $InstallerDir "OdonSetup-$Version-windows-x86_64.exe"

cargo build --release --bin $AppName --bin $McpName

if (Test-Path $StageDir) {
    Remove-Item -Recurse -Force $StageDir
}
New-Item -ItemType Directory -Force -Path $StageDir | Out-Null
New-Item -ItemType Directory -Force -Path $BundleDir | Out-Null

Copy-Item (Join-Path $RootDir "target\release\$AppName.exe") (Join-Path $StageDir "$AppName.exe")
Copy-Item (Join-Path $RootDir "target\release\$McpName.exe") (Join-Path $StageDir "$McpName.exe")

if (Test-Path (Join-Path $RootDir "assets")) {
    Copy-Item (Join-Path $RootDir "assets") (Join-Path $StageDir "assets") -Recurse
}

$ExamplesDir = Join-Path $StageDir "examples"
New-Item -ItemType Directory -Force -Path $ExamplesDir | Out-Null
Copy-Item (Join-Path $RootDir "fixtures\synthetic_5ch.ome.zarr") (Join-Path $ExamplesDir "synthetic_5ch.ome.zarr") -Recurse
Copy-Item (Join-Path $RootDir "fixtures\synthetic_5ch.project.json") (Join-Path $ExamplesDir "synthetic_5ch.project.json")
Copy-Item (Join-Path $RootDir "fixtures\odon-deep-link-test.html") (Join-Path $ExamplesDir "odon-deep-link-test.html")

Copy-Item (Join-Path $RootDir "README.md") (Join-Path $StageDir "README.md")
Copy-Item (Join-Path $RootDir "LICENSE") (Join-Path $StageDir "LICENSE")

if (Test-Path $ArchivePath) {
    Remove-Item -Force $ArchivePath
}

Compress-Archive -Path "$StageDir\*" -DestinationPath $ArchivePath

$IsccCommand = Get-Command "iscc" -ErrorAction SilentlyContinue
$IsccPath = if ($IsccCommand) { $IsccCommand.Source } else { $null }
if (-not $IsccPath) {
    $ProgramFilesX86 = [Environment]::GetEnvironmentVariable("ProgramFiles(x86)")
    $Candidate = if ($ProgramFilesX86) { Join-Path $ProgramFilesX86 "Inno Setup 6\ISCC.exe" } else { $null }
    if ($Candidate -and (Test-Path $Candidate)) {
        $IsccPath = $Candidate
    }
}

if (-not $IsccPath) {
    if ($env:ODON_SKIP_WINDOWS_INSTALLER) {
        Write-Warning "Inno Setup compiler not found; skipping installer because ODON_SKIP_WINDOWS_INSTALLER is set."
        Write-Output $ArchivePath
        exit 0
    }
    throw "Inno Setup compiler not found. Install Inno Setup 6 or set ODON_SKIP_WINDOWS_INSTALLER=1 to build only the zip artifact."
}

New-Item -ItemType Directory -Force -Path $InstallerDir | Out-Null
& $IsccPath `
    "/DSourceDir=$StageDir" `
    "/DOutputDir=$InstallerDir" `
    "/DAppVersion=$Version" `
    (Join-Path $RootDir "installers\windows\odon.iss")

if (-not (Test-Path $InstallerPath)) {
    throw "Expected installer was not produced at $InstallerPath"
}

Write-Output $ArchivePath
Write-Output $InstallerPath
