# Run from the qgan-qlstm-qaoa-llm-trajectory folder or anywhere:
#   powershell -ExecutionPolicy Bypass -File make_zip.ps1
#
# Creates: qgan-app-source.zip next to this script
# Excludes: venv  .git  __pycache__  .pytest_cache  data  models  logs.zip  (the 2+ GB bulk data)

$src  = "D:\dev\app\qgan-qlstm-qaoa-llm-trajectory"
$dst  = "D:\dev\app\qgan-app-source.zip"

$excludeDirs  = @("venv", ".git", "__pycache__", ".pytest_cache", "data", "models")
$excludeFiles = @("logs.zip", "qgan-app-source.zip", "make_zip.ps1")

if (Test-Path $dst) { Remove-Item $dst -Force }

Add-Type -Assembly "System.IO.Compression.FileSystem"
$zip = [System.IO.Compression.ZipFile]::Open($dst, [System.IO.Compression.ZipArchiveMode]::Create)

$files = Get-ChildItem -Path $src -Recurse -File | Where-Object {
    $rel   = $_.FullName.Substring($src.Length + 1)
    $parts = $rel -split "\\"
    $skipDir  = $parts | Where-Object { $excludeDirs  -contains $_ }
    $skipFile = $excludeFiles -contains $_.Name
    (-not $skipDir) -and (-not $skipFile)
}

$count = 0
foreach ($f in $files) {
    $arcName = $f.FullName.Substring($src.Length + 1)
    [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip, $f.FullName, $arcName) | Out-Null
    $count++
    if ($count % 100 -eq 0) { Write-Host "$count files zipped..." }
}

$zip.Dispose()

$sizeMB = [math]::Round((Get-Item $dst).Length / 1MB, 1)
Write-Host ""
Write-Host "Done: $count files -> $sizeMB MB"
Write-Host "Output: $dst"
