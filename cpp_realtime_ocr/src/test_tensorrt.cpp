/**
 * @file test_tensorrt.cpp
 * @brief Test application for TensorRT OCR inference
 * 
 * Tests:
 * - TensorRT engine loading/building from ONNX
 * - SVTR text recognition inference
 * - CTC decoding
 * - End-to-end OCR pipeline timing
 */

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <filesystem>
#include <algorithm>
#include <cmath>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <wincodec.h>
#pragma comment(lib, "windowscodecs.lib")
#endif

#include "ocr/tensorrt_engine.h"
#include "ocr/svtr_inference.h"
#include "ocr/ctc_decoder.h"
#include "utils/timer.h"

#include <cuda_runtime.h>

using namespace trading_monitor;
namespace fs = std::filesystem;

#ifdef _WIN32
static bool loadImageRGBA8WIC(const std::wstring& path, std::vector<uint8_t>& rgba, uint32_t& w, uint32_t& h) {
    HRESULT hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    const bool needUninit = SUCCEEDED(hr);
    if (FAILED(hr) && hr != RPC_E_CHANGED_MODE) {
        return false;
    }

    IWICImagingFactory* factory = nullptr;
    hr = CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&factory));
    if (FAILED(hr) || !factory) {
        if (needUninit) CoUninitialize();
        return false;
    }

    IWICBitmapDecoder* decoder = nullptr;
    hr = factory->CreateDecoderFromFilename(path.c_str(), nullptr, GENERIC_READ, WICDecodeMetadataCacheOnDemand, &decoder);
    if (FAILED(hr) || !decoder) {
        factory->Release();
        if (needUninit) CoUninitialize();
        return false;
    }

    IWICBitmapFrameDecode* frame = nullptr;
    hr = decoder->GetFrame(0, &frame);
    if (FAILED(hr) || !frame) {
        decoder->Release();
        factory->Release();
        if (needUninit) CoUninitialize();
        return false;
    }

    hr = frame->GetSize(&w, &h);
    if (FAILED(hr) || w == 0 || h == 0) {
        frame->Release();
        decoder->Release();
        factory->Release();
        if (needUninit) CoUninitialize();
        return false;
    }

    IWICFormatConverter* converter = nullptr;
    hr = factory->CreateFormatConverter(&converter);
    if (FAILED(hr) || !converter) {
        frame->Release();
        decoder->Release();
        factory->Release();
        if (needUninit) CoUninitialize();
        return false;
    }

    hr = converter->Initialize(frame, GUID_WICPixelFormat32bppRGBA, WICBitmapDitherTypeNone, nullptr, 0.0, WICBitmapPaletteTypeCustom);
    if (FAILED(hr)) {
        converter->Release();
        frame->Release();
        decoder->Release();
        factory->Release();
        if (needUninit) CoUninitialize();
        return false;
    }

    rgba.resize(static_cast<size_t>(w) * static_cast<size_t>(h) * 4);
    const UINT stride = w * 4;
    hr = converter->CopyPixels(nullptr, stride, static_cast<UINT>(rgba.size()), rgba.data());

    converter->Release();
    frame->Release();
    decoder->Release();
    factory->Release();
    if (needUninit) CoUninitialize();

    return SUCCEEDED(hr);
}

static void resizeBilinearRGB(
    const uint8_t* srcRGBA,
    uint32_t srcW,
    uint32_t srcH,
    float* dstRGB,
    int dstW,
    int dstH
) {
    // src is RGBA8, dst is RGB float in [0,1]
    const float sx = static_cast<float>(srcW) / static_cast<float>(dstW);
    const float sy = static_cast<float>(srcH) / static_cast<float>(dstH);

    for (int y = 0; y < dstH; y++) {
        float fy = (y + 0.5f) * sy - 0.5f;
        int y0 = static_cast<int>(std::floor(fy));
        int y1 = y0 + 1;
        float wy = fy - y0;
        y0 = std::clamp(y0, 0, static_cast<int>(srcH) - 1);
        y1 = std::clamp(y1, 0, static_cast<int>(srcH) - 1);

        for (int x = 0; x < dstW; x++) {
            float fx = (x + 0.5f) * sx - 0.5f;
            int x0 = static_cast<int>(std::floor(fx));
            int x1 = x0 + 1;
            float wx = fx - x0;
            x0 = std::clamp(x0, 0, static_cast<int>(srcW) - 1);
            x1 = std::clamp(x1, 0, static_cast<int>(srcW) - 1);

            const uint8_t* p00 = srcRGBA + (static_cast<size_t>(y0) * srcW + x0) * 4;
            const uint8_t* p10 = srcRGBA + (static_cast<size_t>(y0) * srcW + x1) * 4;
            const uint8_t* p01 = srcRGBA + (static_cast<size_t>(y1) * srcW + x0) * 4;
            const uint8_t* p11 = srcRGBA + (static_cast<size_t>(y1) * srcW + x1) * 4;

            for (int c = 0; c < 3; c++) {
                const float v00 = p00[c] / 255.0f;
                const float v10 = p10[c] / 255.0f;
                const float v01 = p01[c] / 255.0f;
                const float v11 = p11[c] / 255.0f;

                const float v0 = v00 + (v10 - v00) * wx;
                const float v1 = v01 + (v11 - v01) * wx;
                const float v = v0 + (v1 - v0) * wy;
                dstRGB[(static_cast<size_t>(c) * dstH + y) * dstW + x] = v;
            }
        }
    }
}

static bool preprocessPaddleOCRRec(
    const std::string& imagePath,
    int dstH,
    int dstW,
    std::vector<float>& nchw
) {
    std::wstring wpath(imagePath.begin(), imagePath.end());
    std::vector<uint8_t> rgba;
    uint32_t w = 0, h = 0;
    if (!loadImageRGBA8WIC(wpath, rgba, w, h)) {
        return false;
    }

    // Keep aspect ratio: resize to dstH, clamp width to dstW, then pad to dstW.
    const float scale = static_cast<float>(dstH) / static_cast<float>(h);
    int resizedW = static_cast<int>(std::round(static_cast<float>(w) * scale));
    resizedW = (std::max)(1, resizedW);
    resizedW = (std::min)(dstW, resizedW);

    // Temporary RGB in [0,1]
    std::vector<float> resizedRGB(static_cast<size_t>(3) * dstH * resizedW);
    resizeBilinearRGB(rgba.data(), w, h, resizedRGB.data(), resizedW, dstH);

    // Output NCHW in [-1, 1] with padding.
    // PaddleOCR commonly uses: (img/255 - 0.5) / 0.5
    nchw.assign(static_cast<size_t>(3) * dstH * dstW, -1.0f);
    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < dstH; y++) {
            for (int x = 0; x < resizedW; x++) {
                float v01 = resizedRGB[(static_cast<size_t>(c) * dstH + y) * resizedW + x];
                float v = (v01 - 0.5f) / 0.5f;
                nchw[(static_cast<size_t>(c) * dstH + y) * dstW + x] = v;
            }
        }
    }
    return true;
}
#endif

void printBanner() {
    std::cout << R"(
========================================================
  TensorRT OCR Inference Test
  Step 4: Model loading and inference verification
========================================================
)" << std::endl;
}

void printUsage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "  --onnx <path>    Path to ONNX model\n"
              << "  --engine <path>  Path to TensorRT engine (will be created if missing)\n"
              << "  --dict <path>    Path to character dictionary\n"
              << "  --fp16           Enable FP16 precision\n"
              << "  --image <path>   Run inference on a real image (png/jpg/bmp)\n"
              << "  --test-image     Run with synthetic test image\n"
              << "  --benchmark <n>  Run N inference iterations\n"
              << "  --help           Show this help\n";
}

// Create a simple test character dictionary
void createDefaultDictionary(const std::string& path) {
    std::ofstream file(path);
    // Common characters for trading screens
    const char* chars = "0123456789.,-+$%ABCDEFGHIJKLMNOPQRSTUVWXYZ ";
    for (size_t i = 0; chars[i]; i++) {
        file << chars[i] << "\n";
    }
    file.close();
}

int main(int argc, char* argv[]) {
    printBanner();

    // Parse arguments
    std::string onnxPath;
    std::string enginePath;
    std::string dictPath;
    std::string imagePath;
    bool useFP16 = true;
    bool testImage = false;
    int benchmarkIters = 10;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--onnx" && i + 1 < argc) {
            onnxPath = argv[++i];
        } else if (arg == "--engine" && i + 1 < argc) {
            enginePath = argv[++i];
        } else if (arg == "--dict" && i + 1 < argc) {
            dictPath = argv[++i];
        } else if (arg == "--fp16") {
            useFP16 = true;
        } else if (arg == "--no-fp16") {
            useFP16 = false;
        } else if (arg == "--image" && i + 1 < argc) {
            imagePath = argv[++i];
        } else if (arg == "--test-image") {
            testImage = true;
        } else if (arg == "--benchmark" && i + 1 < argc) {
            benchmarkIters = std::stoi(argv[++i]);
        } else if (arg == "--help") {
            printUsage(argv[0]);
            return 0;
        }
    }

    // Check model paths
    if (onnxPath.empty() && enginePath.empty()) {
        std::cout << "No model specified. Checking default paths..." << std::endl;
        
        // Check common locations
        std::vector<std::string> searchPaths = {
            "models/svtr_tiny.engine",
            "models/recognition.engine",
            "../models/svtr_tiny.engine",
            "models/svtr_tiny.onnx",
            "models/recognition.onnx",
            "../models/recognition.onnx",
        };
        
        for (const auto& path : searchPaths) {
            if (fs::exists(path)) {
                if (path.ends_with(".engine")) {
                    enginePath = path;
                } else {
                    onnxPath = path;
                }
                std::cout << "Found model: " << path << std::endl;
                break;
            }
        }
        
        if (onnxPath.empty() && enginePath.empty()) {
            std::cerr << "\nNo model found. Please provide --onnx or --engine path.\n";
            std::cerr << "\nTo download and convert a model, run:\n";
            std::cerr << "  python -m realtime_ocr.setup_models --output-dir models\n";
            return 1;
        }
    }

    // Set default engine path if only ONNX provided
    if (enginePath.empty() && !onnxPath.empty()) {
        enginePath = onnxPath.substr(0, onnxPath.rfind('.')) + ".engine";
    }

    // Initialize CUDA
    std::cout << "\n[1/4] Initializing CUDA..." << std::endl;
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return 1;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "  Device: " << prop.name << std::endl;
    std::cout << "  Compute: " << prop.major << "." << prop.minor << std::endl;

    // Load or build TensorRT engine
    std::cout << "\n[2/4] Loading TensorRT engine..." << std::endl;

    SVTRInference svtr;
    HighResTimer timer;

    bool modelLoaded = false;

    // Try to load existing engine first
    if (!enginePath.empty() && fs::exists(enginePath)) {
        std::cout << "  Loading engine: " << enginePath << std::endl;
        timer.start();
        modelLoaded = svtr.loadModel(enginePath);
        timer.stop();
        if (modelLoaded) {
            std::cout << "  Engine loaded in " << timer.elapsedMs() << " ms" << std::endl;
        }
    }

    // If no engine, try building from ONNX
    if (!modelLoaded && !onnxPath.empty() && fs::exists(onnxPath)) {
        std::cout << "  Building engine from ONNX: " << onnxPath << std::endl;
        std::cout << "  FP16: " << (useFP16 ? "enabled" : "disabled") << std::endl;
        std::cout << "  This may take a few minutes..." << std::endl;

        timer.start();
        modelLoaded = svtr.buildModel(onnxPath, enginePath, useFP16);
        timer.stop();

        if (modelLoaded) {
            std::cout << "  Engine built in " << timer.elapsedMs() << " ms" << std::endl;
            std::cout << "  Saved to: " << enginePath << std::endl;
        } else {
            std::cerr << "  Failed to build engine!" << std::endl;
        }
    }

    if (!modelLoaded) {
        std::cout << "\nNo model loaded. Test infrastructure verified." << std::endl;
        std::cout << "\nTo download and convert a model, run:\n";
        std::cout << "  python -m realtime_ocr.setup_models --output-dir models\n";
        std::cout << "\nThen run:\n";
        std::cout << "  test_tensorrt.exe --onnx models/recognition.onnx\n";
        return 0;
    }

    // Query output classes early so we can validate dictionary size.
    int timesteps = 0, numClasses = 0;
    svtr.getOutputDims(timesteps, numClasses);

    // Load or create dictionary
    std::cout << "\n[3/4] Setting up CTC decoder..." << std::endl;
    CTCDecoder decoder;

    // Search for dictionary in common locations
    std::vector<std::string> dictSearchPaths = {
        dictPath,
        "models/ppocr_keys_v1.txt",
        "../models/ppocr_keys_v1.txt",
        "cpp_realtime_ocr/models/ppocr_keys_v1.txt",
    };

    bool dictLoaded = false;
    for (const auto& path : dictSearchPaths) {
        if (!path.empty() && fs::exists(path)) {
            if (decoder.loadDictionary(path)) {
                std::cout << "  Loaded dictionary: " << path << std::endl;
                std::cout << "  Characters: " << decoder.getDictionarySize() << std::endl;
                dictLoaded = true;
                break;
            }
        }
    }

    if (dictLoaded) {
        // PaddleOCR rec models are commonly trained with an extra "space" class,
        // which makes numClasses = len(ppocr_keys_v1.txt) + 2 (blank + space).
        if (!decoder.ensurePaddleOCRDictionarySize(numClasses)) {
            std::cerr << "  WARNING: Dictionary size (" << decoder.getDictionarySize()
                      << ") does not match model numClasses (" << numClasses << ")" << std::endl;
        } else {
            std::cout << "  Dictionary adjusted to match model classes: "
                      << decoder.getDictionarySize() << std::endl;
        }
    }

    if (!dictLoaded) {
        std::cerr << "  WARNING: No dictionary found. Text decoding will fail." << std::endl;
        std::cerr << "  Download ppocr_keys_v1.txt to models/ directory." << std::endl;
    }

    // Run inference test
    std::cout << "\n[4/4] Running inference benchmark (" << benchmarkIters << " iterations)..." << std::endl;

    // Get model dimensions
    std::cout << "  Output: " << timesteps << " timesteps x " << numClasses << " classes" << std::endl;

    // Use non-default CUDA stream to avoid TensorRT default-stream sync warning.
    cudaStream_t stream = nullptr;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    // Create synthetic test input (pattern simulating text)
    // The exported Paddle SVTR model expects NCHW float input: (1, 3, 48, W)
    int inputWidth = 320, inputHeight = 48;
    constexpr int inputChannels = 3;
    size_t inputSize = static_cast<size_t>(inputChannels) * inputWidth * inputHeight * sizeof(float);
    float* d_input = nullptr;
    cudaMalloc(&d_input, inputSize);

    std::vector<float> h_input;

    bool haveRealInput = false;
#ifdef _WIN32
    if (!imagePath.empty()) {
        if (preprocessPaddleOCRRec(imagePath, inputHeight, inputWidth, h_input)) {
            haveRealInput = true;
            std::cout << "  Loaded and preprocessed image: " << imagePath << std::endl;
        } else {
            std::cerr << "  WARNING: Failed to load image (falling back to synthetic): " << imagePath << std::endl;
        }
    }
#endif

    if (!haveRealInput) {
        // Synthetic input in [-1,1] range
        h_input.assign(static_cast<size_t>(inputChannels) * inputWidth * inputHeight, -1.0f);
        for (int c = 0; c < inputChannels; c++) {
            for (int y = 0; y < inputHeight; y++) {
                for (int x = 0; x < inputWidth; x++) {
                    const float v = (x % 20 < 10) ? 0.6f : -0.6f;
                    h_input[static_cast<size_t>(c) * inputWidth * inputHeight + y * inputWidth + x] = v;
                }
            }
        }
    }

    cudaMemcpy(d_input, h_input.data(), inputSize, cudaMemcpyHostToDevice);

    // Warmup
    std::cout << "  Warming up..." << std::endl;
    for (int i = 0; i < 5; i++) {
        svtr.infer(d_input, inputWidth, inputHeight, stream);
    }
    cudaStreamSynchronize(stream);

    // Benchmark inference using CUDA events (more accurate, less sync overhead)
    cudaEvent_t evStart{}, evStop{};
    cudaEventCreate(&evStart);
    cudaEventCreate(&evStop);

    double totalTime = 0;
    double minTime = 1e9, maxTime = 0;

    for (int i = 0; i < benchmarkIters; i++) {
        cudaEventRecord(evStart, stream);
        svtr.infer(d_input, inputWidth, inputHeight, stream);
        cudaEventRecord(evStop, stream);
        cudaEventSynchronize(evStop);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, evStart, evStop);
        double us = static_cast<double>(ms) * 1000.0;
        totalTime += us;
        minTime = (std::min)(minTime, us);
        maxTime = (std::max)(maxTime, us);
    }

    double avgTime = totalTime / benchmarkIters;

    // Get and decode output
    float* d_output = svtr.getOutputProbs();
    std::vector<float> h_output(timesteps * numClasses);
    cudaMemcpy(h_output.data(), d_output, h_output.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // PaddleOCR uses blank index 0.
    CTCResult result = decoder.decode(h_output.data(), timesteps, numClasses, 0);

    if (result.text.empty()) {
        // Quick debug: print top-1 indices for a few timesteps.
        std::cout << "\nCTC debug (first 10 timesteps argmax indices):" << std::endl;
        for (int t = 0; t < (std::min)(10, timesteps); t++) {
            const float* row = h_output.data() + static_cast<size_t>(t) * numClasses;
            int best = 0;
            float bestv = row[0];
            for (int c = 1; c < numClasses; c++) {
                if (row[c] > bestv) { bestv = row[c]; best = c; }
            }
            std::cout << "  t=" << t << " idx=" << best << std::endl;
        }
        if (!imagePath.empty()) {
            std::cout << "(If this is a real image, preprocessing may still need tuning.)" << std::endl;
        }
    }

    // Print results
    std::cout << "\n========================================================" << std::endl;
    std::cout << "Inference Results:" << std::endl;
    std::cout << "  Iterations:  " << benchmarkIters << std::endl;
    std::cout << "  Avg time:    " << std::fixed << std::setprecision(2) << avgTime << " us" << std::endl;
    std::cout << "  Min time:    " << minTime << " us" << std::endl;
    std::cout << "  Max time:    " << maxTime << " us" << std::endl;
    std::cout << "  Throughput:  " << std::setprecision(1) << (1e6 / avgTime) << " inferences/sec" << std::endl;
    std::cout << "\nDecoded text: \"" << result.text << "\"" << std::endl;
    std::cout << "  Confidence:  " << std::setprecision(3) << result.confidence << std::endl;

    // Verify performance target
    std::cout << "\n========================================================" << std::endl;
    if (avgTime < 5000.0) {  // 5ms target for inference alone
        std::cout << "PASS: Inference time < 5ms target (" << std::setprecision(2) << avgTime << " us)" << std::endl;
    } else {
        std::cout << "WARN: Inference time > 5ms target (" << std::setprecision(2) << avgTime << " us)" << std::endl;
    }
    std::cout << "========================================================" << std::endl;

    // Cleanup
    cudaFree(d_input);
    cudaEventDestroy(evStart);
    cudaEventDestroy(evStop);
    cudaStreamDestroy(stream);

    return 0;
}

