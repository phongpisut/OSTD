# Builds the portable zip: dist\PaperTranslate-<version>-portable.zip with a PaperTranslate\ folder inside.
# Run from the repo root: powershell -File portable\package.ps1
$ErrorActionPreference = "Stop"
$version = (Select-String -Path Cargo.toml -Pattern '^version = "(.+)"').Matches[0].Groups[1].Value
cargo build --release
$dir = "dist\PaperTranslate"
Remove-Item $dir -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $dir | Out-Null
Copy-Item target\release\paper-translate.exe, target\release\DirectML.dll, portable\README.txt $dir
$zip = "dist\PaperTranslate-$version-portable.zip"
Remove-Item $zip -ErrorAction SilentlyContinue
Compress-Archive -Path $dir -DestinationPath $zip -CompressionLevel Optimal
Remove-Item $dir -Recurse -Force
"$zip  sha256 $((Get-FileHash $zip -Algorithm SHA256).Hash.ToLower())"
