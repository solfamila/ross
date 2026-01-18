# Ross - Real-time Trading Screen OCR

High-performance C++ application for real-time OCR of trading platform screens using CUDA and TensorRT.

[![Build](https://github.com/solfamila/ross/actions/workflows/build.yml/badge.svg)](https://github.com/solfamila/ross/actions/workflows/build.yml)

## Features

- **Sub-20ms latency** end-to-end OCR pipeline
- **Windows Graphics Capture API** for low-latency screen capture
- **TensorRT 10.x** accelerated inference (PP-OCRv4 recognition model)
- **CUDA 13.x** GPU preprocessing kernels
- **YOLOv10** window detection (optional, for dynamic ROI positioning)
- **Template matching** for anchor-based table detection
- **Change detection** to skip redundant OCR on static frames

## Prerequisites

| Component | Version | Notes |
|-----------|---------|-------|
| **OS** | Windows 11 22H2+ | Required for Windows.Graphics.Capture |
| **GPU** | NVIDIA RTX 30/40/50 series | Compute capability 8.0+ |
| **CUDA Toolkit** | 13.1+ | [Download](https://developer.nvidia.com/cuda-downloads) |
| **TensorRT** | 10.14+ | [Download](https://developer.nvidia.com/tensorrt) |
| **Visual Studio** | 2022/2026 | MSVC toolchain |
| **CMake** | 3.28+ | |
| **Ninja** | 1.11+ | `winget install Ninja-build.Ninja` |

## Build

```powershell
# Open Visual Studio Developer Command Prompt
cd cpp_realtime_ocr
mkdir build && cd build
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release
ninja
```

## Usage

```powershell
# Process a static image
.\trading_monitor.exe --image screenshot.png --scan-table positions_table

# Live capture from a window
.\trading_monitor.exe --window "Lightspeed" --scan-table positions_table --verbose

# Live capture from monitor
.\trading_monitor.exe --monitor 0 --scan-table positions_table

# With template-based anchor detection
.\trading_monitor.exe --image screenshot.png \
  --scan-table positions_table \
  --anchor-template models/row_header.png \
  --anchor-offset 0,16,300,170 \
  --verbose
```

## Example Output

```
=== OCR RESULTS ===
  [positions_row0_symbol] BNKK  (conf=0.97)
  [positions_row0_open_pnl] 0.00 (conf=0.94)
  [positions_row0_realized_pnl] +14,855.05 (conf=0.99)
  [positions_row1_symbol] SPHL  (conf=0.95)
  ...
=== PROCESSING COMPLETE (12.3ms) ===
```

## Project Structure

```
ross/
├── cpp_realtime_ocr/          # Main C++ application
│   ├── src/                   # Source files
│   ├── include/               # Headers
│   ├── models/                # ONNX models and dictionaries
│   ├── config/                # ROI configuration
│   └── IMPLEMENTATION_GUIDE.md
├── training/                  # YOLO training pipeline
│   ├── annotate.py            # Visual annotation tool
│   ├── train.py               # Training script
│   └── trading_windows.yaml   # Dataset config
└── README.md
```

## Training Custom Window Detection (Optional)

```powershell
# 1. Copy screenshots to training/images/
# 2. Annotate bounding boxes
python training/annotate.py training/images

# 3. Train YOLOv10
cd training
python train.py --epochs 50

# 4. Use in trading_monitor
.\trading_monitor.exe --detect-onnx ..\models\trading_windows.onnx
```

## Documentation

See [IMPLEMENTATION_GUIDE.md](cpp_realtime_ocr/IMPLEMENTATION_GUIDE.md) for:
- Detailed architecture overview
- Command-line reference
- ROI configuration
- TensorRT engine management
- Performance tuning
- Troubleshooting

## License

MIT License - see [LICENSE](LICENSE)

