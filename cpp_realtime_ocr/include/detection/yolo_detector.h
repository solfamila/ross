/**
 * @file yolo_detector.h
 * @brief YOLOv10-based window/UI element detection using TensorRT
 *
 * Provides robust detection of trading windows and UI elements,
 * replacing brittle template matching with AI-based detection.
 */

#pragma once

#include "ocr/tensorrt_engine.h"
#include "types.h"
#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <vector>

namespace trading_monitor {

/**
 * @brief Detection result from YOLO inference
 */
struct DetectionResult {
    int classId;           ///< Detected class ID
    std::string className; ///< Class name (e.g., "lightspeed_window")
    float confidence;      ///< Detection confidence [0-1]
    float x;               ///< Bounding box center X (pixels)
    float y;               ///< Bounding box center Y (pixels)
    float width;           ///< Bounding box width (pixels)
    float height;          ///< Bounding box height (pixels)

    /// Convert to ROI struct
    ROI toROI(const std::string& name = "") const {
        ROI roi;
        roi.name = name.empty() ? className : name;
        roi.x = static_cast<int>(x - width / 2.0f);
        roi.y = static_cast<int>(y - height / 2.0f);
        roi.w = static_cast<int>(width);
        roi.h = static_cast<int>(height);
        return roi;
    }
};

/**
 * @brief YOLOv10 object detector for window/UI element detection
 *
 * Uses TensorRT for GPU-accelerated inference. Supports:
 * - YOLOv10-nano for minimal latency (~2-3ms on RTX)
 * - Custom fine-tuned models for trading UI detection
 * - NMS (Non-Maximum Suppression) for duplicate removal
 */
class YOLODetector {
public:
    YOLODetector();
    ~YOLODetector();

    // Non-copyable
    YOLODetector(const YOLODetector&) = delete;
    YOLODetector& operator=(const YOLODetector&) = delete;

    /**
     * @brief Load YOLOv10 TensorRT engine
     * @param enginePath Path to .engine file
     * @param classNames Vector of class names (index = class ID)
     * @return true if successful
     */
    bool loadModel(const std::string& enginePath,
                   const std::vector<std::string>& classNames = {});

    /**
     * @brief Build engine from ONNX model
     * @param onnxPath Path to YOLOv10 ONNX model
     * @param enginePath Path to save engine
     * @param useFP16 Enable FP16 precision
     * @return true if successful
     */
    bool buildModel(const std::string& onnxPath,
                    const std::string& enginePath,
                    bool useFP16 = true);

    /**
     * @brief Run detection on BGRA texture
     * @param texObj CUDA texture object from captured frame
     * @param frameWidth Frame width in pixels
     * @param frameHeight Frame height in pixels
     * @param stream CUDA stream for async execution
     * @return Vector of detected objects
     */
    std::vector<DetectionResult> detect(
        cudaTextureObject_t texObj,
        int frameWidth, int frameHeight,
        cudaStream_t stream = nullptr);

    /**
     * @brief Run detection on CPU image (for testing)
     * @param imageData BGRA pixel data
     * @param width Image width
     * @param height Image height
     * @return Vector of detected objects
     */
    std::vector<DetectionResult> detectCPU(
        const uint8_t* imageData,
        int width, int height);

    /// Set confidence threshold for detections
    void setConfidenceThreshold(float threshold) { m_confThreshold = threshold; }

    /// Set IoU threshold for NMS
    void setNMSThreshold(float threshold) { m_nmsThreshold = threshold; }

    /// Check if model is loaded
    bool isLoaded() const { return m_engine && m_engine->isLoaded(); }

    /// Get last error message
    const std::string& getLastError() const { return m_lastError; }

    /// Get input dimensions
    void getInputDims(int& width, int& height, int& channels) const;

private:
    std::unique_ptr<TensorRTEngine> m_engine;
    std::vector<std::string> m_classNames;

    // Model dimensions
    int m_inputWidth = 640;
    int m_inputHeight = 640;
    int m_inputChannels = 3;
    int m_numClasses = 1;
    int m_maxDetections = 100;

    // Detection thresholds
    float m_confThreshold = 0.5f;
    float m_nmsThreshold = 0.45f;

    // GPU buffers
    float* m_inputBuffer = nullptr;
    float* m_outputBuffer = nullptr;
    size_t m_outputBufferSize = 0;

    std::string m_lastError;

    /// Preprocess frame: resize, normalize, BGRA->RGB
    bool preprocess(cudaTextureObject_t texObj,
                    int frameWidth, int frameHeight,
                    cudaStream_t stream);

    /// Postprocess: parse detections, apply NMS
    std::vector<DetectionResult> postprocess(
        int origWidth, int origHeight);

    /// Apply Non-Maximum Suppression
    std::vector<DetectionResult> applyNMS(
        std::vector<DetectionResult>& detections);
};

} // namespace trading_monitor

