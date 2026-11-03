# Windows build. nvcc needs a host C++ compiler for everything outside a kernel, so
# the MSVC build tools must be on PATH. If you get
#     nvcc fatal : Cannot find compiler 'cl.exe' in PATH
# then either run this from a "x64 Native Tools Command Prompt for VS", or install
# the Build Tools for Visual Studio (the "Desktop development with C++" workload) and
# let this script locate them via vswhere.
param([string]$Arch = "native", [switch]$Run)

$ErrorActionPreference = "Stop"

if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        $vs = & $vswhere -latest -products * `
              -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
              -property installationPath
        if ($vs) {
            $devcmd = Join-Path $vs "Common7\Tools\VsDevCmd.bat"
            Write-Host "Importing MSVC environment from $devcmd"
            cmd /c "`"$devcmd`" -arch=x64 -no_logo && set" | ForEach-Object {
                if ($_ -match '^([^=]+)=(.*)$') { Set-Item -Path "env:$($matches[1])" -Value $matches[2] }
            }
        }
    }
}
if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    throw "cl.exe not found. Install the Visual Studio Build Tools (Desktop development with C++)."
}

$flags = @("-O3", "-std=c++17", "-arch=$Arch", "-lineinfo")
$targets = @{
    "01_vector_add" = @()
    "02_sgemm"      = @("-lcublas")
    "03_softmax"    = @()
    "04_attention"  = @()
}

foreach ($name in $targets.Keys | Sort-Object) {
    Write-Host "building $name ..." -ForegroundColor Cyan
    & nvcc @flags -o "$name.exe" "$name.cu" @($targets[$name])
    if ($LASTEXITCODE -ne 0) { throw "build failed: $name" }
}
Write-Host "build complete" -ForegroundColor Green

if ($Run) {
    foreach ($name in $targets.Keys | Sort-Object) {
        Write-Host "`n=== $name ===" -ForegroundColor Yellow
        & ".\$name.exe"
    }
}
