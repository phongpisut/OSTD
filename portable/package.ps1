# Builds the portable zip: dist\OSTD-<version>-portable.zip with an OSTD\ folder inside.
# Run from the repo root: powershell -File portable\package.ps1
$ErrorActionPreference = "Stop"
$version = (Select-String -Path Cargo.toml -Pattern '^version = "(.+)"').Matches[0].Groups[1].Value
cargo build --release
# staged under %TEMP%, so an unzipped copy in dist\ (maybe running) is never touched
$stage = Join-Path $env:TEMP "ostd-package"
$dir = Join-Path $stage "OSTD"
Remove-Item $stage -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $dir | Out-Null
Copy-Item target\release\ostd.exe, target\release\DirectML.dll, portable\README.txt $dir
New-Item -ItemType Directory -Force dist | Out-Null
$zip = "dist\OSTD-$version-portable.zip"
Remove-Item $zip -ErrorAction SilentlyContinue
Compress-Archive -Path $dir -DestinationPath $zip -CompressionLevel Optimal
Remove-Item $stage -Recurse -Force
"$zip  sha256 $((Get-FileHash $zip -Algorithm SHA256).Hash.ToLower())"
