# setup.ps1 - Automated setup for Trading Screen Monitor
# Run as Administrator: Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
# Then: .\setup.ps1

param(
    [switch]$SkipCuda,
    [switch]$SkipTensorRT,
    [switch]$SkipVcpkg,
    [switch]$SkipWin32Sample,
    [string]$TensorRTPath = "C:\TensorRT-10.14.0.16"
)

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " Trading Screen Monitor - Setup Script" -ForegroundColor Cyan
Write-Host " CUDA 12.x + TensorRT 10.x" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Check if running as admin
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "WARNING: Not running as Administrator. Some installations may fail." -ForegroundColor Yellow
}

# =============================================================================
# 1. Check Prerequisites
# =============================================================================
Write-Host "`n[1/7] Checking prerequisites..." -ForegroundColor Green

# Check Visual Studio
$vsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (Test-Path $vsWhere) {
    $vsPath = & $vsWhere -latest -property installationPath
    Write-Host "  Visual Studio found: $vsPath" -ForegroundColor Gray
} else {
    Write-Host "  WARNING: Visual Studio not found. Please install Visual Studio 2022/2026 with C++ workload." -ForegroundColor Yellow
}

# Check CMake
$cmake = Get-Command cmake -ErrorAction SilentlyContinue
if ($cmake) {
    $cmakeVersion = (cmake --version | Select-Object -First 1)
    Write-Host "  $cmakeVersion" -ForegroundColor Gray
} else {
    Write-Host "  Installing CMake..." -ForegroundColor Yellow
    winget install Kitware.CMake --accept-package-agreements --accept-source-agreements
}

# Check Git
$git = Get-Command git -ErrorAction SilentlyContinue
if ($git) {
    Write-Host "  Git found: $(git --version)" -ForegroundColor Gray
} else {
    Write-Host "  Installing Git..." -ForegroundColor Yellow
    winget install Git.Git --accept-package-agreements --accept-source-agreements
}

# =============================================================================
# 2. CUDA Check (12.4+ recommended for TensorRT 10.x)
# =============================================================================
Write-Host "`n[2/7] Checking CUDA..." -ForegroundColor Green

if (-not $SkipCuda) {
    $nvcc = Get-Command nvcc -ErrorAction SilentlyContinue
    if ($nvcc) {
        $cudaVersion = (nvcc --version | Select-String "release" | ForEach-Object { $_.ToString() })
        Write-Host "  CUDA found: $cudaVersion" -ForegroundColor Green

        # Extract version number
        if ($cudaVersion -match "release (\d+)\.(\d+)") {
            $major = [int]$Matches[1]
            $minor = [int]$Matches[2]
            if ($major -lt 12 -or ($major -eq 12 -and $minor -lt 4)) {
                Write-Host "  WARNING: CUDA 12.4+ recommended for TensorRT 10.x" -ForegroundColor Yellow
            }
        }
    } else {
        Write-Host "  CUDA not found in PATH." -ForegroundColor Yellow
        Write-Host "  Please download CUDA from:" -ForegroundColor Yellow
        Write-Host "  https://developer.nvidia.com/cuda-downloads" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "  CUDA 12.4+ is recommended for TensorRT 10.x compatibility" -ForegroundColor Gray
    }
}

# =============================================================================
# 3. TensorRT 10.x Check
# =============================================================================
Write-Host "`n[3/7] Checking TensorRT 10.x..." -ForegroundColor Green

if (-not $SkipTensorRT) {
    # Check for any TensorRT 10.x installation
    $trtPaths = @($TensorRTPath) + (Get-ChildItem "C:\TensorRT-10*" -Directory -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FullName)
    $foundTrt = $null

    foreach ($path in $trtPaths) {
        if (Test-Path $path) {
            $foundTrt = $path
            break
        }
    }

    if ($foundTrt) {
        Write-Host "  TensorRT found: $foundTrt" -ForegroundColor Green

        # Set environment variable
        [Environment]::SetEnvironmentVariable("TENSORRT_ROOT", $foundTrt, "User")
        $env:TENSORRT_ROOT = $foundTrt
        Write-Host "  Set TENSORRT_ROOT environment variable" -ForegroundColor Gray
    } else {
        Write-Host "  TensorRT not found" -ForegroundColor Yellow
        Write-Host "  Please download TensorRT 10.x from:" -ForegroundColor Yellow
        Write-Host "  https://developer.nvidia.com/tensorrt/download" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "  Select: TensorRT 10.x GA for Windows 10/11 x86_64 CUDA 12.x ZIP" -ForegroundColor Gray
        Write-Host "  Extract to: C:\TensorRT-10.x.x.x\" -ForegroundColor Gray
    }
}

# =============================================================================
# 4. vcpkg Setup
# =============================================================================
Write-Host "`n[4/7] Setting up vcpkg..." -ForegroundColor Green

$vcpkgPath = "C:\vcpkg"
if (-not $SkipVcpkg) {
    if (Test-Path $vcpkgPath) {
        Write-Host "  vcpkg found: $vcpkgPath" -ForegroundColor Gray
    } else {
        Write-Host "  Cloning vcpkg..." -ForegroundColor Yellow
        git clone https://github.com/Microsoft/vcpkg.git $vcpkgPath
        & "$vcpkgPath\bootstrap-vcpkg.bat"
    }
    
    # Install dependencies
    Write-Host "  Installing nlohmann-json..." -ForegroundColor Gray
    & "$vcpkgPath\vcpkg" install nlohmann-json:x64-windows
    
    # Integrate with Visual Studio
    & "$vcpkgPath\vcpkg" integrate install
}

# =============================================================================
# 5. Clone Win32CaptureSample
# =============================================================================
Write-Host "`n[5/7] Setting up Win32CaptureSample..." -ForegroundColor Green

$externalDir = Join-Path $PSScriptRoot "external"
$sampleDir = Join-Path $externalDir "Win32CaptureSample"

if (-not $SkipWin32Sample) {
    if (-not (Test-Path $externalDir)) {
        New-Item -ItemType Directory -Path $externalDir -Force | Out-Null
    }
    
    if (Test-Path $sampleDir) {
        Write-Host "  Win32CaptureSample already cloned" -ForegroundColor Gray
    } else {
        Write-Host "  Cloning Win32CaptureSample..." -ForegroundColor Yellow
        git clone https://github.com/robmikh/Win32CaptureSample.git $sampleDir
    }
}

# =============================================================================
# 6. Create Directory Structure
# =============================================================================
Write-Host "`n[6/7] Creating project directory structure..." -ForegroundColor Green

$directories = @(
    "src/capture",
    "src/processing",
    "src/ocr",
    "src/detection",
    "src/utils",
    "include/capture",
    "include/processing",
    "include/ocr",
    "include/detection",
    "include/utils",
    "models",
    "config",
    "build"
)

foreach ($dir in $directories) {
    $fullPath = Join-Path $PSScriptRoot $dir
    if (-not (Test-Path $fullPath)) {
        New-Item -ItemType Directory -Path $fullPath -Force | Out-Null
        Write-Host "  Created: $dir" -ForegroundColor Gray
    }
}

# Copy ROI config if exists
$roiConfig = Join-Path (Split-Path $PSScriptRoot -Parent) "roi_config.json"
if (Test-Path $roiConfig) {
    Copy-Item $roiConfig -Destination (Join-Path $PSScriptRoot "config\roi_config.json") -Force
    Write-Host "  Copied roi_config.json to config/" -ForegroundColor Gray
}

# =============================================================================
# 7. Build Instructions
# =============================================================================
Write-Host "`n[7/7] Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host " Next Steps" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Verify CUDA and TensorRT installation:" -ForegroundColor White
Write-Host "   nvcc --version" -ForegroundColor Gray
Write-Host '   dir $env:TENSORRT_ROOT' -ForegroundColor Gray
Write-Host ""
Write-Host "2. Build Win32CaptureSample (Step 1 of development plan):" -ForegroundColor White
Write-Host "   cd external\Win32CaptureSample" -ForegroundColor Gray
Write-Host "   mkdir build; cd build" -ForegroundColor Gray
Write-Host '   cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release' -ForegroundColor Gray
Write-Host "   ninja" -ForegroundColor Gray
Write-Host "   .\Win32CaptureSample.exe" -ForegroundColor Gray
Write-Host ""
Write-Host "3. Build trading_monitor (from Developer PowerShell):" -ForegroundColor White
Write-Host "   cd build" -ForegroundColor Gray
Write-Host '   cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release' -ForegroundColor Gray
Write-Host "   cmake --build . --config Release" -ForegroundColor Gray
Write-Host ""
Write-Host "4. Convert ONNX models to TensorRT engines:" -ForegroundColor White
Write-Host "   trtexec --onnx=models\svtr_tiny.onnx --saveEngine=models\svtr_tiny.engine --fp16" -ForegroundColor Gray
Write-Host ""

# Summary of what was installed/found
Write-Host "========================================" -ForegroundColor Cyan
Write-Host " Installation Summary" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

$summary = @{
    "Visual Studio" = if (Test-Path $vsWhere) { "Found" } else { "Not found" }
    "CMake" = if (Get-Command cmake -ErrorAction SilentlyContinue) { "Found" } else { "Missing" }
    "Git" = if (Get-Command git -ErrorAction SilentlyContinue) { "Found" } else { "Missing" }
    "CUDA" = if (Get-Command nvcc -ErrorAction SilentlyContinue) { "Found" } else { "Missing" }
    "TensorRT" = if ($env:TENSORRT_ROOT -and (Test-Path $env:TENSORRT_ROOT)) { "Found" } else { "Missing" }
    "vcpkg" = if (Test-Path $vcpkgPath) { "Found" } else { "Missing" }
    "Win32CaptureSample" = if (Test-Path $sampleDir) { "Cloned" } else { "Not cloned" }
}

foreach ($item in $summary.GetEnumerator()) {
    $color = if ($item.Value -match "Found|Cloned") { "Green" } else { "Yellow" }
    Write-Host ("  {0,-20} : {1}" -f $item.Key, $item.Value) -ForegroundColor $color
}

Write-Host ""
Write-Host "See IMPLEMENTATION_GUIDE.md for detailed instructions." -ForegroundColor White

