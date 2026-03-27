$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $PSScriptRoot
$AppName = if ($env:APP_NAME) { $env:APP_NAME } else { "odon" }
$BundleDir = Join-Path $RootDir "target\release\bundle\windows"
$StageDir = Join-Path $BundleDir "$AppName-windows-x86_64"
$ArchivePath = Join-Path $BundleDir "$AppName-windows-x86_64.zip"

cargo build --release

if (Test-Path $StageDir) {
    Remove-Item -Recurse -Force $StageDir
}
New-Item -ItemType Directory -Force -Path $StageDir | Out-Null
New-Item -ItemType Directory -Force -Path $BundleDir | Out-Null

Copy-Item (Join-Path $RootDir "target\release\$AppName.exe") (Join-Path $StageDir "$AppName.exe")

if (Test-Path (Join-Path $RootDir "assets")) {
    Copy-Item (Join-Path $RootDir "assets") (Join-Path $StageDir "assets") -Recurse
}

Copy-Item (Join-Path $RootDir "README.md") (Join-Path $StageDir "README.md")
Copy-Item (Join-Path $RootDir "LICENSE") (Join-Path $StageDir "LICENSE")

if (Test-Path $ArchivePath) {
    Remove-Item -Force $ArchivePath
}

Compress-Archive -Path "$StageDir\*" -DestinationPath $ArchivePath

Write-Output $ArchivePath
