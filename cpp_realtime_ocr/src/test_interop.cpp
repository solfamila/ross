/**
 * @file test_interop.cpp
 * @brief Test application for D3D11-CUDA interop
 * 
 * Demonstrates:
 * - Screen capture using Windows.Graphics.Capture
 * - Zero-copy transfer to CUDA via D3D11 interop
 * - Timing measurements for each stage
 * 
 * Build and run to verify Step 2 of the implementation.
 */

#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <atomic>
#include <csignal>

#include <wrl/client.h>

#include "capture/d3d11_capture.h"
#include "capture/cuda_interop.h"
#include "processing/cuda_kernels.h"
#include "utils/timer.h"

using namespace trading_monitor;

// Global flag for graceful shutdown
std::atomic<bool> g_running{true};

void signalHandler(int) {
    std::cout << "\nShutdown requested..." << std::endl;
    g_running = false;
}

void printBanner() {
    std::cout << R"(
========================================================
  D3D11-CUDA Interop Test
  Step 2: Zero-copy frame transfer verification
========================================================
)" << std::endl;
}

// Timing statistics
struct TimingStats {
    double mapTimeUs = 0;
    double unmapTimeUs = 0;
    double textureCreateTimeUs = 0;
    double preprocessTimeUs = 0;  // New: CUDA preprocessing kernels
    double totalTimeUs = 0;
    int sampleCount = 0;

    void update(double map, double unmap, double texCreate, double preprocess = 0) {
        double alpha = (sampleCount == 0) ? 1.0 : 0.1;  // EMA
        mapTimeUs = mapTimeUs * (1 - alpha) + map * alpha;
        unmapTimeUs = unmapTimeUs * (1 - alpha) + unmap * alpha;
        textureCreateTimeUs = textureCreateTimeUs * (1 - alpha) + texCreate * alpha;
        preprocessTimeUs = preprocessTimeUs * (1 - alpha) + preprocess * alpha;
        totalTimeUs = mapTimeUs + unmapTimeUs + textureCreateTimeUs + preprocessTimeUs;
        sampleCount++;
    }

    void print() const {
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "  Map:          " << std::setw(8) << mapTimeUs << " us\n";
        std::cout << "  Texture:      " << std::setw(8) << textureCreateTimeUs << " us\n";
        std::cout << "  Preprocess:   " << std::setw(8) << preprocessTimeUs << " us\n";
        std::cout << "  Unmap:        " << std::setw(8) << unmapTimeUs << " us\n";
        std::cout << "  TOTAL:        " << std::setw(8) << totalTimeUs << " us  ("
                  << (totalTimeUs / 1000.0) << " ms)\n";
    }
};

int main(int argc, char* argv[]) {
    printBanner();
    
    // Setup signal handler
    std::signal(SIGINT, signalHandler);
    
    // Parse arguments
    std::string targetWindow;
    bool captureMonitor = true;
    int testFrames = 100;
    bool debugPrint = false;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--window" && i + 1 < argc) {
            targetWindow = argv[++i];
            captureMonitor = false;
        } else if (arg == "--frames" && i + 1 < argc) {
            testFrames = std::stoi(argv[++i]);
        } else if (arg == "--debug") {
            debugPrint = true;
        } else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "  --window <title>  Capture specific window\n"
                      << "  --frames <n>      Number of test frames (default: 100)\n"
                      << "  --debug           Print one-time texture/device info\n"
                      << "  --help            Show this help\n";
            return 0;
        }
    }
    
    // Step 1: Initialize D3D11 Capture
    std::cout << "[1/4] Initializing D3D11 capture..." << std::endl;
    
    D3D11Capture capture;
    if (!capture.initializeDevice()) {
        std::cerr << "ERROR: Failed to initialize D3D11 device: " 
                  << capture.getLastError() << std::endl;
        return 1;
    }
    
    if (captureMonitor) {
        if (!capture.initializeMonitor()) {
            std::cerr << "ERROR: Failed to initialize monitor capture: "
                      << capture.getLastError() << std::endl;
            return 1;
        }
        std::cout << "  Capturing primary monitor" << std::endl;
    } else {
        if (!capture.initializeWindow(targetWindow)) {
            std::cerr << "ERROR: Failed to find window '" << targetWindow << "': "
                      << capture.getLastError() << std::endl;
            return 1;
        }
        std::cout << "  Capturing window: " << targetWindow << std::endl;
    }
    
    int w, h;
    capture.getFrameSize(w, h);
    std::cout << "  Frame size: " << w << "x" << h << std::endl;
    
    // Step 2: Initialize CUDA interop
    std::cout << "\n[2/4] Initializing CUDA interop..." << std::endl;
    
    CudaD3D11Interop interop;
    if (!interop.initialize(capture.getDevice())) {
        std::cerr << "ERROR: Failed to initialize CUDA interop: "
                  << interop.getLastError() << std::endl;
        return 1;
    }
    
    std::cout << "  CUDA device: " << interop.getCudaDevice() << std::endl;
    
    // Create texture pool for caching registrations
    CudaTexturePool texturePool(interop);
    
    // Step 3: Run capture test
    std::cout << "\n[3/4] Running interop test (" << testFrames << " frames)..." << std::endl;
    
    TimingStats stats;
    std::atomic<int> frameCount{0};
    HighResTimer timer;
    
    // Frame processing callback
    auto frameCallback = [&](const CaptureFrame& frame) {
        if (frameCount >= testFrames || !g_running) return;

        static std::atomic<bool> printedTextureInfo{ false };
        if (debugPrint && !printedTextureInfo.exchange(true)) {
            D3D11_TEXTURE2D_DESC desc{};
            frame.texture->GetDesc(&desc);
            std::cout << "\n[debug] Texture desc: "
                      << "Format=" << static_cast<int>(desc.Format)
                      << " Usage=" << static_cast<int>(desc.Usage)
                      << " Bind=0x" << std::hex << desc.BindFlags
                      << " Misc=0x" << desc.MiscFlags
                      << std::dec << std::endl;

            Microsoft::WRL::ComPtr<ID3D11Device> texDev;
            frame.texture->GetDevice(texDev.GetAddressOf());
            std::cout << "[debug] Texture device == capture device: "
                      << ((texDev.Get() == capture.getDevice()) ? "YES" : "NO")
                      << std::endl;
        }

        double mapTime = 0, unmapTime = 0, texCreateTime = 0, preprocessTime = 0;

        // Static GPU buffer for preprocessing output (reused across frames)
        static float* d_grayBuffer = nullptr;
        static int bufferWidth = 0, bufferHeight = 0;

        // Get or register texture with CUDA
        cudaGraphicsResource_t resource = texturePool.getOrRegister(frame.texture);
        if (!resource) {
            std::cerr << "Failed to register texture: " << interop.getLastError() << std::endl;
            return;
        }

        // Time the map operation
        timer.start();
        if (!interop.mapResource(resource, interop.getStream())) {
            std::cerr << "Failed to map resource: " << interop.getLastError() << std::endl;
            return;
        }
        timer.stop();
        mapTime = timer.elapsedUs();

        // Get CUDA array from mapped resource
        cudaArray_t array = interop.getMappedArray(resource);
        if (!array) {
            std::cerr << "Failed to get mapped array: " << interop.getLastError() << std::endl;
            interop.unmapResource(resource, interop.getStream());
            return;
        }

        // Create texture object for sampling
        timer.start();
        cudaTextureObject_t texObj = interop.createTextureObject(array);
        timer.stop();
        texCreateTime = timer.elapsedUs();

        if (!texObj) {
            std::cerr << "Failed to create texture object: " << interop.getLastError() << std::endl;
            interop.unmapResource(resource, interop.getStream());
            return;
        }

        // =====================================================
        // CUDA Preprocessing Pipeline Test
        // =====================================================
        timer.start();

        // Setup ROI parameters (extract center 400x100 region, upscale 2x)
        cuda::ROIParams roiParams;
        roiParams.roiX = (std::max)(0, (frame.width - 400) / 2);
        roiParams.roiY = (std::max)(0, (frame.height - 100) / 2);
        roiParams.roiWidth = (std::min)(400, frame.width);
        roiParams.roiHeight = (std::min)(100, frame.height);
        roiParams.scale = 2.0f;
        roiParams.outWidth = static_cast<int>(roiParams.roiWidth * roiParams.scale);
        roiParams.outHeight = static_cast<int>(roiParams.roiHeight * roiParams.scale);

        // Ensure output buffer is large enough
        if (roiParams.outWidth != bufferWidth || roiParams.outHeight != bufferHeight) {
            if (d_grayBuffer) cudaFree(d_grayBuffer);
            size_t bufferSize = roiParams.outWidth * roiParams.outHeight * sizeof(float);
            cudaMalloc(&d_grayBuffer, bufferSize);
            bufferWidth = roiParams.outWidth;
            bufferHeight = roiParams.outHeight;
        }

        // 1. BGRA to Grayscale with ROI extraction
        cudaError_t err = cuda::launchBgraToGrayROI(
            texObj, d_grayBuffer, roiParams, interop.getStream()
        );
        if (err != cudaSuccess) {
            std::cerr << "BgraToGrayROI failed: " << cudaGetErrorString(err) << std::endl;
        }

        // 2. Contrast enhancement
        cuda::ContrastParams contrastParams{1.2f, 0.0f};  // 20% more contrast
        err = cuda::launchContrastEnhance(
            d_grayBuffer, roiParams.outWidth, roiParams.outHeight,
            contrastParams, interop.getStream()
        );
        if (err != cudaSuccess) {
            std::cerr << "ContrastEnhance failed: " << cudaGetErrorString(err) << std::endl;
        }

        // 3. Normalize for TensorRT input
        err = cuda::launchNormalize(
            d_grayBuffer, roiParams.outWidth, roiParams.outHeight,
            0.5f, 0.5f, interop.getStream()
        );
        if (err != cudaSuccess) {
            std::cerr << "Normalize failed: " << cudaGetErrorString(err) << std::endl;
        }

        // Sync to measure preprocessing time accurately
        cudaStreamSynchronize(interop.getStream());
        timer.stop();
        preprocessTime = timer.elapsedUs();

        // Cleanup texture object
        interop.destroyTextureObject(texObj);

        // Unmap resource
        timer.start();
        interop.unmapResource(resource, interop.getStream());
        cudaStreamSynchronize(interop.getStream());  // Ensure complete for timing
        timer.stop();
        unmapTime = timer.elapsedUs();

        // Update statistics with preprocessing time
        stats.update(mapTime, unmapTime, texCreateTime, preprocessTime);
        frameCount++;

        // Progress update
        if (frameCount % 10 == 0) {
            std::cout << "  Frame " << frameCount << "/" << testFrames
                      << " - Interop time: " << std::fixed << std::setprecision(2)
                      << stats.totalTimeUs << " us" << std::endl;
        }
    };

    // Start capture
    capture.startCapture(frameCallback);

    // Wait for test to complete
    while (frameCount < testFrames && g_running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Stop capture
    capture.stopCapture();

    // Step 4: Print results
    std::cout << "\n[4/4] Results:" << std::endl;
    std::cout << "========================================================" << std::endl;
    std::cout << "Frames processed: " << frameCount << std::endl;
    std::cout << "Textures cached:  " << texturePool.size() << std::endl;
    std::cout << "\nInterop Timing (EMA):" << std::endl;
    stats.print();

    auto captureStats = capture.getStats();
    std::cout << "\nCapture Statistics:" << std::endl;
    std::cout << "  Frames received: " << captureStats.framesReceived << std::endl;
    std::cout << "  Avg frame time:  " << std::fixed << std::setprecision(2)
              << captureStats.avgFrameTimeMs << " ms" << std::endl;

    // Verify performance target
    std::cout << "\n========================================================" << std::endl;
    if (frameCount == 0 || texturePool.size() == 0) {
        std::cout << "FAIL: No frames processed (interop registration failed)" << std::endl;
        std::cout << "========================================================" << std::endl;
        return 1;
    }

    if (stats.totalTimeUs < 1000.0) {
        std::cout << "PASS: Interop time < 1ms target ("
                  << std::fixed << std::setprecision(2) << stats.totalTimeUs
                  << " us)" << std::endl;
    } else {
        std::cout << "FAIL: Interop time > 1ms target ("
                  << std::fixed << std::setprecision(2) << stats.totalTimeUs
                  << " us)" << std::endl;
    }
    std::cout << "========================================================" << std::endl;

    return 0;
}
