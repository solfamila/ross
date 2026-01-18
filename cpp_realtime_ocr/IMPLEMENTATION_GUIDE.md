# High-Performance C++ Trading Screen Capture - Technical Implementation Guide

**Version:** 1.1
**Last Updated:** January 2026
**Target Latency:** Sub-20ms end-to-end
**Status:** ✅ Feature Complete (Active Testing)

## Version Requirements

| Component | Version | Notes |
|-----------|---------|-------|
| **CUDA Toolkit** | 13.1+ | CUDA 13.1.1 tested |
| **TensorRT** | 10.14+ | TensorRT 10.14.1.48 tested |
| **Visual Studio** | 2022/2026 | MSVC toolchain (use Ninja as generator) |
| **Ninja** | 1.11+ | Fast build system, install via `winget install Ninja-build.Ninja` |
| **Windows SDK** | 10.0.22621.0+ | Windows 11 SDK |
| **CMake** | 3.28+ | Required for modern CUDA support |
| **Windows** | 11 22H2+ | Required for Windows.Graphics.Capture |
| **GPU** | RTX 30/40/50 series | Compute 8.0+ recommended |

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [Command-Line Reference](#2-command-line-reference)
3. [Architecture Overview](#3-architecture-overview)
4. [Table Scanning & Column Extraction](#4-table-scanning--column-extraction)
5. [ROI Configuration](#5-roi-configuration)
6. [TensorRT Integration](#6-tensorrt-integration)
7. [YOLOv10 Window Detection](#7-yolov10-window-detection)
8. [Performance](#8-performance)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Quick Start

### Build

```powershell
# Recommended (Windows): use the repo build script (sets up VS DevCmd correctly)
cd cpp_realtime_ocr
cmd.exe /d /s /c ".\\build.cmd Release"

# If you prefer a one-liner without the script, you MUST run under VS DevCmd so
# MSVC standard headers + Windows SDK headers are on the include path:
cmd.exe /d /s /c '"C:\\Program Files\\Microsoft Visual Studio\\18\\Insiders\\Common7\\Tools\\VsDevCmd.bat" -arch=amd64 -host_arch=amd64 & cmake --build "C:\\Users\\forfr\\Downloads\\trade\\cpp_realtime_ocr\\build" --config Release -j 4'

# Manual configure (first time) + build (Ninja single-config)
cd cpp_realtime_ocr
mkdir build; cd build
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build . -j 4
```

### Basic Usage

```powershell
# Process a static image
.\trading_monitor.exe --image screenshot.png --scan-table positions_table

# Offline MP4 decode smoke-test (Media Foundation)
.\trading_monitor.exe --video sample.mp4 --video-max-frames 300

# Offline panel auto-detection (Phase 2 scaffolding)
# Expects these files under templates/headers/:
#   positions_hdr.png, order_hdr.png, quote_hdr.png
.\trading_monitor.exe --video sample.mp4 --video-max-frames 1 --detect-panels

# Live capture from a window
.\trading_monitor.exe --window "Lightspeed" --scan-table positions_table

# Live capture from monitor
.\trading_monitor.exe --monitor 0 --scan-table positions_table
```

### Example with Column Extraction

```powershell
.trading_monitor.exe --image screenshot.png \
  --scan-table positions_table \
  --anchor-template models/row_header_cols.png \
  --anchor-offset 0,16,300,170 \
  --anchor-threshold 0.55 \
  --anchor-max-search 640
  --verbose
```

Output:
```
=== OCR RESULTS ===
  [lightspeed_positions_table_row0_symbol] BNKK  (raw='BNKK', conf=0.9735)
  [lightspeed_positions_table_row0_open_pnl] 0.00 (raw='0.00', conf=0.9368)
  [lightspeed_positions_table_row0_realized_pnl] +14,855.05 (raw='+14,855.05', conf=0.9857)
```

---

## 2. Command-Line Reference

### Capture Source Options

| Option | Description |
|--------|-------------|
| `--window <title>` | Capture window by title (partial match) |
| `--monitor <idx>` | Capture monitor by index (default: 0) |
| `--image <path>` | Process static image instead of live capture |
| `--list-windows` | List available windows and exit |
| `--list-monitors` | List available monitors and exit |
| `--select-window [idx]` | Interactive window selection |
| `--select-monitor [idx]` | Interactive monitor selection |

### ROI Configuration

| Option | Description |
|--------|-------------|
| `--config <path>` | Config file path (default: `config/roi_config.json`) |
| `--roi <name>` | Process only this named ROI |
| `--select-roi [idx]` | Interactive ROI selection (optional index) |
| `--all-rois` | Process all configured ROIs |
| `--list-rois` | List ROIs from config and exit |
| `--create-roi [name]` | Create/update ROI by dragging on screen |
| `--test-roi <x,y,w,h>` | Test custom ROI coordinates |

### Table Scanning Options

| Option | Description |
|--------|-------------|
| `--scan-table <roi>` | Scan a table ROI by slicing into rows |
| `--table-rows <n>` | Number of rows to scan (default: 6) |
| `--row-height <px>` | Row height in pixels (default: 18) |
| `--row-stride <px>` | Row-to-row Y step (default: 18) |
| `--row-offset-y <px>` | Y offset from table top (default: 0) |
| `--auto-rows` | Auto-detect row boundaries |
| `--row-detect-mode <mode>` | Detection mode: `intensity`, `template`, `hybrid` |
| `--anchor-template <path>` | Template image used to anchor the table on screen |
| `--anchor-offset <dx,dy,w,h>` | Table ROI offset from the matched template (scaled by match) |
| `--anchor-scales <s1,s2,...>` | Template scales to search (default: `0.6,0.7,0.75,0.8,0.9,1.0,1.1,1.2`) |
| `--anchor-threshold <f>` | Minimum NCC score for anchor match (default: `0.55`) |
| `--anchor-max-search <px>` | Max search width for anchor (0 = full res, default: 640) |
| `--anchor-search <x,y,w,h>` | Restrict anchor search to a screen region |
| `--anchor-secondary-template <path>` | Secondary template to validate anchor matches |
| `--anchor-secondary-offset <dx,dy,w,h>` | Expected secondary offset from anchor (scaled) |
| `--anchor-secondary-threshold <f>` | Min NCC score for secondary template (default: 0.6) |
| `--anchor-every <n>` | Re-anchor every N frames in live mode (default: 0 = once) |
| `--template-dir <path>` | Template directory for template matching (default: `templates`) |
| `--template-threshold <f>` | Match threshold for template NCC (default: 0.7) |
| `--template-row-height <px>` | Expected row height for template detection (default: 14) |
| `--template-row-spacing <px>` | Expected row spacing (default: 2) |
| `--template-max-rows <n>` | Max rows to detect via template matching (default: 20) |
| `--print-all-rows` | Print every row on every frame |

### Column Extraction Options

| Option | Description |
|--------|-------------|
| `--columns <spec>` | Multi-column: `"name:x,w;name:x,w"` |
| `--col-x <px>` | Single column X offset (default: 0 = full row) |
| `--col-w <px>` | Single column width (default: 0 = full row) |

**Column format:** `"name:x_offset,width;name:x_offset,width;..."`

Example: `--columns "symbol:11,30;open_pnl:193,25;realized_pnl:228,60"`

### Model Options

| Option | Description |
|--------|-------------|
| `--engine <path>` | TensorRT engine path |
| `--onnx <path>` | ONNX model path (builds engine if missing) |
| `--dict <path>` | Character dictionary path |

### Output Options

| Option | Description |
|--------|-------------|
| `--json-output <file>` | Write results to JSON file |
| `--dump-roi <dir>` | Dump preprocessed ROI images as PGM |
| `--show-roi` | Draw ROI rectangles on overlay |
| `--ocr-zoom <f>` | Pre-upscale/zoom ROI content before OCR (default: 1.0) |
| `--verbose` | Enable verbose logging |

### Performance Options

| Option | Description |
|--------|-------------|
| `--every <n>` | Process every Nth frame (default: 5) |
| `--no-change-detect` | Disable change detection (always OCR) |
| `--change-threshold <n>` | Hamming distance threshold (default: 5) |
| `--benchmark` | Run benchmark mode |

---

## 3. Architecture Overview

### D3D11-CUDA Staging Texture Workaround

Windows.Graphics.Capture textures have `SHARED_NTHANDLE` / `KEYEDMUTEX` misc flags that are **incompatible** with `cudaGraphicsD3D11RegisterResource()`. Direct registration fails with `cudaErrorInvalidResourceHandle`.

**Solution:** Copy each captured frame to a staging texture created with compatible flags:

```cpp
// Create staging texture (once, or when size changes)
D3D11_TEXTURE2D_DESC desc = {};
desc.Width = width;
desc.Height = height;
desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
desc.Usage = D3D11_USAGE_DEFAULT;
desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;  // Required for CUDA
desc.MiscFlags = 0;  // NO shared flags - makes it CUDA-compatible
device->CreateTexture2D(&desc, nullptr, &stagingTexture);

// Per-frame: copy WGC texture → staging texture
context->CopyResource(stagingTexture, wgcTexture);

// Register staging texture with CUDA (once per texture)
cudaGraphicsD3D11RegisterResource(&resource, stagingTexture, cudaGraphicsRegisterFlagsNone);
```

This adds ~0.2ms latency but enables zero-copy CUDA access to the GPU texture data.

### Pipeline Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        TRADING SCREEN MONITOR PIPELINE                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐    │
│  │  Windows.Graphics│     │  Staging Texture │     │   CUDA Array     │    │
│  │  .Capture        │────▶│  (CopyResource)  │────▶│   (Zero-Copy)    │    │
│  │  ~3ms            │     │  ~0.2ms          │     │   <0.5ms         │    │
│  └──────────────────┘     └──────────────────┘     └──────────────────┘    │
│                                                             │               │
│                                                             ▼               │
│  ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐    │
│  │  Row Detection   │     │  CUDA Kernels    │     │  Column Extract  │    │
│  │  (intensity/     │────▶│  ROI+Preprocess  │────▶│  (per-row split) │    │
│  │   template)      │     │  ~1ms            │     │                  │    │
│  └──────────────────┘     └──────────────────┘     └──────────────────┘    │
│                                                             │               │
│                                                             ▼               │
│  ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐    │
│  │  Change Detector │     │  TensorRT SVTR   │     │  CTC Decoder     │    │
│  │  (Skip unchanged)│◀────│  FP16 Inference  │◀────│  (Greedy)        │    │
│  │                  │     │  ~5ms            │     │  <0.1ms          │    │
│  └──────────────────┘     └──────────────────┘     └──────────────────┘    │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  OUTPUT: Per-Column OCR Results                                     │   │
│  │  { roi: "row0_symbol", text: "BNKK", confidence: 0.97 }            │   │
│  │  { roi: "row0_pnl", text: "+14,855.05", confidence: 0.99 }         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  TOTAL LATENCY: 6-15ms per ROI (target: <20ms) ✅                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Table Scanning & Column Extraction

### Workflow

1. **Define Table ROI** - A rectangular region containing the table
2. **Slice into Rows** - Fixed or auto-detected row boundaries
3. **Extract Columns** - Split each row into named columns
4. **OCR Each Column** - Separate recognition for each field

### Template Anchoring (Window Movement)

If the trading window moves, enable anchoring to re-locate the table from a stable UI landmark (e.g., the positions header row). The matcher searches the full screen (or a restricted region) for the template and then derives the table ROI from an offset.

```powershell
.\trading_monitor.exe --image screenshot.png \
  --scan-table positions_table \
  --anchor-template models/row_header_cols.png \
  --anchor-offset 0,16,300,170 \
  --anchor-threshold 0.55 \
  --anchor-max-search 640
```

Notes:
- `--anchor-offset dx,dy,w,h` is defined relative to the matched template’s top-left corner and is scaled by the detected template scale.
- Use `--anchor-search x,y,w,h` to limit matching to the left/top area if there are multiple similar tables on screen.
- For live capture, `--anchor-every N` can periodically re-lock the table if the window moves.

### Finding Column Offsets

Use Python to analyze a screenshot and find column boundaries:

```python
from PIL import Image
import numpy as np

img = Image.open('screenshot.png')
row = img.crop((table_x, row_y, table_x + table_w, row_y + row_h))
gray = np.array(row.convert('L'))

# Find bright text on dark background
col_intensity = np.max(gray, axis=0)
text_cols = np.where(col_intensity > 100)[0]

# Group into column boundaries
# Output: Column 0: x=11-38, Column 1: x=193-211, etc.
```

### JSON Output Format

When using `--json-output results.json`:

```json
{
  "timestamp": 1737012345678,
  "source": "screenshot.png",
  "total_latency_ms": 45.23,
  "results": [
    {
      "roi": "positions_row0_symbol",
      "x": 11, "y": 627, "w": 30, "h": 10,
      "text": "BNKK",
      "raw_text": "BNKK",
      "confidence": 0.9735,
      "latency_ms": 7.42
    },
    {
      "roi": "positions_row0_pnl",
      "x": 228, "y": 627, "w": 60, "h": 10,
      "text": "+14,855.05",
      "raw_text": "+14,855.05",
      "confidence": 0.9857,
      "latency_ms": 7.18
    }
  ]
}
```

---

## 5. ROI Configuration

### Config File Format (`config/roi_config.json`)

```json
{
  "version": 1,
  "upscaleFactor": 4.0,
  "rois": [
    {
      "name": "lightspeed_positions_table",
      "x": 0,
      "y": 600,
      "w": 300,
      "h": 170
    }
  ]
}
```

### Creating ROIs Interactively

```powershell
# Select a window and drag to create ROI
.\trading_monitor.exe --select-window --create-roi my_table
```

---

## 6. TensorRT Integration

### Model Files

| File | Description |
|------|-------------|
| `models/recognition.onnx` | Default ONNX recognition model |
| `models/recognition.engine` | Default TensorRT engine (auto-built) |
| `models/ppocr_keys_v1.txt` | Default character dictionary |
| `models/en_PP-OCRv4_rec.onnx` | Alternate SVTR-tiny ONNX (v4) |
| `models/en_PP-OCRv4_rec.engine` | Alternate TensorRT engine (v4) |
| `models/en_PP-OCRv4_dict.txt` | Alternate dictionary for v4 model |

### Engine Auto-Build

TensorRT engines are **GPU-specific** and **TensorRT-version-specific**. The program automatically builds an engine from ONNX when:

1. **Engine file doesn't exist** - First run on a new machine
2. **Engine file is older than ONNX** - Model was updated
3. **Engine was built for different GPU/TensorRT** - Deserialization fails

**Auto-build logic:**
```
if (!engineExists || !canDeserialize) {
    buildFromONNX(onnxPath, enginePath, useFP16=true);
    saveEngine(enginePath);
}
```

**Build time:** 30-120 seconds depending on GPU and model complexity.

**Manual rebuild:**
```powershell
# Force rebuild by deleting old engine
del models\recognition.engine
.\trading_monitor.exe --onnx models/recognition.onnx --engine models/recognition.engine
```

### Engine Configuration

- **Input:** `x` with shape `[1, 3, 48, W]` where W is dynamic (100-640)
- **Output:** `[T, 97]` where T = W/8 timesteps, 97 character classes
- **Precision:** FP16 on RTX GPUs (auto-detected via `platformHasFastFp16()`)
- **LayerNorm fix:** Reduce/Unary ops forced to FP32 to prevent overflow
- **Workspace:** 1GB memory pool for optimization tactics

---

## 7. YOLOv10 Window Detection

YOLOv10-based detection provides robust, AI-powered window localization that replaces brittle template matching.

### Why YOLO Detection?

| Issue | Template Matching | YOLO Detection |
|-------|------------------|----------------|
| Stream quality changes | ❌ Fails | ✅ Robust |
| Window resizing | ❌ Requires multi-scale | ✅ Automatic |
| Color scheme changes | ❌ Fails | ✅ Robust |
| Multiple similar elements | ❌ False positives | ✅ Class-aware |
| Latency | ~5-15ms | ~2-4ms |

### Setup

1. **Download YOLOv10-nano ONNX:**
```powershell
# Download from Ultralytics or use custom fine-tuned model
# Place in models/yolov10n_window.onnx
```

2. **Fine-tune for trading windows (optional):**
   - Collect 50-100 screenshots of your trading platform
   - Use LabelImg to annotate bounding boxes around windows/tables
   - Fine-tune for 10-20 epochs on "lightspeed_window" class

3. **Auto-build TensorRT engine:**
```powershell
.\trading_monitor.exe --detect-onnx models/yolov10n_window.onnx \
  --window "Lightspeed" --verbose
# Engine builds automatically on first run (~30-60s)
```

### CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--detect-model <path>` | YOLOv10 TensorRT engine | models/yolov10n_window.engine |
| `--detect-onnx <path>` | ONNX for auto-build | - |
| `--detect-confidence <f>` | Detection threshold | 0.8 |
| `--detect-every <n>` | Run detection every N frames | 5 |
| `--detect-classes <names>` | Comma-separated class names | "window" |

### Usage Example

```powershell
# YOLO detection with offset to position table
.\trading_monitor.exe --window "Lightspeed" \
  --detect-model models/yolov10n_window.engine \
  --detect-confidence 0.7 \
  --detect-every 5 \
  --anchor-offset 0,50,300,200 \
  --scan-table positions_table \
  --table-rows 8 --verbose
```

### Detection Pipeline

```
Frame → YOLO Preprocess → TensorRT Inference → NMS → ROI Offset → OCR Pipeline
  |         (GPU)              (~2-3ms)        (CPU)    (apply)
  v
Letterbox resize to 640x640, BGRA→RGB, normalize [0,1]
```

### Integration with Template Matching

YOLO detection takes priority when enabled. Falls back to template matching if:
- Detection confidence below threshold
- No detections in frame
- `--detect-model` not specified

```cpp
if (useYoloDetection && yoloDetectionValid) {
    activeTableROI = calculateTableFromDetection(yoloDetectedROI, offset);
} else if (templateMatchingEnabled) {
    activeTableROI = calculateTableFromAnchor(anchorResult);
}
```

### Expected Performance

- **Detection latency:** 2-4ms (YOLOv10-nano on RTX 5070)
- **Accuracy:** 95%+ with fine-tuned model
- **False positive rate:** <1% with proper confidence threshold

---

## 8. Performance

### Measured Latency (RTX 5070 Laptop GPU)

| Stage | Time |
|-------|------|
| Screen Capture | ~3ms |
| D3D11→CUDA Map | <0.5ms |
| ROI Extract + Preprocess | ~1ms |
| TensorRT Inference | ~5ms |
| CTC Decode | <0.1ms |
| **Total per ROI** | **~10ms** |

### Optimization Tips

1. **Use `--every N`** - Skip frames for lower CPU usage
2. **Enable change detection** - Only OCR when content changes
3. **Batch columns** - Process multiple columns per row efficiently
4. **Use FP16** - Automatic on RTX GPUs

---

## 9. Troubleshooting

### TODO

- **Investigate false positives on 2026-01-16**: Anchor/template matching is locking to a visually similar region, and OCR outputs are low-confidence/noisy. On 2026-01-15 the same pipeline reads BNKK/SPHL/CJMB, but on 2026-01-16 those symbols are not present and OCR fails to resolve the actual symbols. Likely causes: scaled-down text, insufficient preprocessing/upscaling, and ambiguous template matches when the window layout changes.

### Common Issues

**CUDA device not found:**
```
ERROR: No CUDA devices found
```
→ Install NVIDIA drivers: `nvidia-smi` should show your GPU

**TensorRT engine incompatible:**
```
Engine was built for different GPU/TensorRT version
```
→ Delete `.engine` file and rebuild with `--onnx`

**Window capture permission denied:**
```
CreateCaptureSession returned nullptr
```
→ Windows Settings → Privacy → Screen capture → Allow

**Empty OCR results:**
```
[roi_name]  (raw='', conf=0)
```
→ Check ROI coordinates, use `--dump-roi` to visualize
→ Verify text is visible and high contrast

### Debug Commands

```powershell
# Dump preprocessed ROI images
.\trading_monitor.exe --image test.png --scan-table my_table --dump-roi ./debug/

# Verbose logging
.\trading_monitor.exe --image test.png --verbose

# Test specific coordinates
.\trading_monitor.exe --image test.png --test-roi 100,200,50,15
```