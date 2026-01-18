/**
 * @file yolo_detector.cpp
 * @brief YOLOv10 window detector implementation
 */

#include "detection/yolo_detector.h"
#include "processing/cuda_kernels.h"
#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>
#include <cmath>

namespace trading_monitor {

// Helper to calculate IoU for NMS
static float calculateIoU(const DetectionResult& a, const DetectionResult& b) {
    float x1 = std::max(a.x - a.width/2, b.x - b.width/2);
    float y1 = std::max(a.y - a.height/2, b.y - b.height/2);
    float x2 = std::min(a.x + a.width/2, b.x + b.width/2);
    float y2 = std::min(a.y + a.height/2, b.y + b.height/2);

    float intersection = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    float areaA = a.width * a.height;
    float areaB = b.width * b.height;
    float unionArea = areaA + areaB - intersection;

    return (unionArea > 0) ? intersection / unionArea : 0.0f;
}

YOLODetector::YOLODetector()
    : m_engine(std::make_unique<TensorRTEngine>()) {}

YOLODetector::~YOLODetector() {
    if (m_inputBuffer) cudaFree(m_inputBuffer);
    if (m_outputBuffer) cudaFree(m_outputBuffer);
}

bool YOLODetector::loadModel(const std::string& enginePath,
                              const std::vector<std::string>& classNames) {
    if (!m_engine->loadEngine(enginePath)) {
        m_lastError = "Failed to load engine: " + m_engine->getLastError();
        return false;
    }

    m_classNames = classNames;
    if (m_classNames.empty()) {
        m_classNames = {"window"}; // Default class
    }
    m_numClasses = static_cast<int>(m_classNames.size());

    // Get input dimensions from engine
    auto inputNames = m_engine->getInputNames();
    if (inputNames.empty()) {
        m_lastError = "No input tensors found";
        return false;
    }

    auto inputDims = m_engine->getInputDims(inputNames[0]);
    // YOLOv10 input: [batch, channels, height, width] = [1, 3, 640, 640]
    if (inputDims.nbDims >= 4) {
        m_inputChannels = (inputDims.d[1] > 0) ? static_cast<int>(inputDims.d[1]) : 3;
        m_inputHeight = (inputDims.d[2] > 0) ? static_cast<int>(inputDims.d[2]) : 640;
        m_inputWidth = (inputDims.d[3] > 0) ? static_cast<int>(inputDims.d[3]) : 640;
    }

    // Allocate input buffer
    size_t inputSize = m_inputChannels * m_inputHeight * m_inputWidth;
    cudaError_t err = cudaMalloc(&m_inputBuffer, inputSize * sizeof(float));
    if (err != cudaSuccess) {
        m_lastError = "Failed to allocate input buffer";
        return false;
    }

    // Get output dimensions
    auto outputNames = m_engine->getOutputNames();
    if (outputNames.empty()) {
        m_lastError = "No output tensors found";
        return false;
    }

    // Set input to get concrete output shape
    nvinfer1::Dims4 dims(1, m_inputChannels, m_inputHeight, m_inputWidth);
    m_engine->setInput(inputNames[0], m_inputBuffer, dims);

    auto outputDims = m_engine->getContextDims(outputNames[0]);
    // YOLOv10 output varies by model version:
    // - YOLOv10-nano: [1, 300, 6] for NMS-included or [1, 84, 8400] for raw
    size_t outputElems = 1;
    for (int i = 0; i < outputDims.nbDims; i++) {
        if (outputDims.d[i] > 0) outputElems *= outputDims.d[i];
    }
    m_outputBufferSize = outputElems;

    err = cudaMalloc(&m_outputBuffer, outputElems * sizeof(float));
    if (err != cudaSuccess) {
        m_lastError = "Failed to allocate output buffer";
        return false;
    }

    std::cout << "YOLO detector loaded: input=" << m_inputWidth << "x" << m_inputHeight
              << ", classes=" << m_numClasses << std::endl;

    return true;
}

bool YOLODetector::buildModel(const std::string& onnxPath,
                               const std::string& enginePath,
                               bool useFP16) {
    if (!m_engine->buildFromONNX(onnxPath, enginePath, useFP16, 1)) {
        m_lastError = "Failed to build engine: " + m_engine->getLastError();
        return false;
    }
    return loadModel(enginePath, m_classNames);
}

void YOLODetector::getInputDims(int& width, int& height, int& channels) const {
    width = m_inputWidth;
    height = m_inputHeight;
    channels = m_inputChannels;
}

std::vector<DetectionResult> YOLODetector::applyNMS(
    std::vector<DetectionResult>& detections) {
    // Sort by confidence descending
    std::sort(detections.begin(), detections.end(),
        [](const DetectionResult& a, const DetectionResult& b) {
            return a.confidence > b.confidence;
        });

    std::vector<DetectionResult> result;
    std::vector<bool> suppressed(detections.size(), false);

    for (size_t i = 0; i < detections.size(); i++) {
        if (suppressed[i]) continue;

        result.push_back(detections[i]);

        for (size_t j = i + 1; j < detections.size(); j++) {
            if (suppressed[j]) continue;
            if (detections[i].classId != detections[j].classId) continue;

            if (calculateIoU(detections[i], detections[j]) > m_nmsThreshold) {
                suppressed[j] = true;
            }
        }
    }

    return result;
}

bool YOLODetector::preprocess(cudaTextureObject_t texObj,
                               int frameWidth, int frameHeight,
                               cudaStream_t stream) {
    // Use CUDA kernel to resize and normalize frame to model input
    // BGRA -> RGB, resize to 640x640, normalize to [0,1]
    cuda::launchYOLOPreprocess(
        texObj, frameWidth, frameHeight,
        m_inputBuffer, m_inputWidth, m_inputHeight,
        stream
    );
    return true;
}

std::vector<DetectionResult> YOLODetector::detect(
    cudaTextureObject_t texObj,
    int frameWidth, int frameHeight,
    cudaStream_t stream) {

    if (!isLoaded()) {
        m_lastError = "Model not loaded";
        return {};
    }

    // Preprocess frame
    if (!preprocess(texObj, frameWidth, frameHeight, stream)) {
        return {};
    }

    // Set input/output
    auto inputNames = m_engine->getInputNames();
    auto outputNames = m_engine->getOutputNames();

    nvinfer1::Dims4 dims(1, m_inputChannels, m_inputHeight, m_inputWidth);
    if (!m_engine->setInput(inputNames[0], m_inputBuffer, dims)) {
        m_lastError = "Failed to set input";
        return {};
    }
    if (!m_engine->setOutput(outputNames[0], m_outputBuffer)) {
        m_lastError = "Failed to set output";
        return {};
    }

    // Run inference
    if (!m_engine->inferAsync(stream)) {
        m_lastError = "Inference failed";
        return {};
    }
    cudaStreamSynchronize(stream);

    // Postprocess
    return postprocess(frameWidth, frameHeight);
}

std::vector<DetectionResult> YOLODetector::detectCPU(
    const uint8_t* imageData,
    int width, int height) {

    if (!isLoaded()) {
        m_lastError = "Model not loaded";
        return {};
    }

    // Upload to GPU and preprocess
    size_t imageSize = width * height * 4; // BGRA
    uint8_t* d_image = nullptr;
    cudaMalloc(&d_image, imageSize);
    cudaMemcpy(d_image, imageData, imageSize, cudaMemcpyHostToDevice);

    // Create simple texture for preprocessing
    cuda::launchYOLOPreprocessFromBGRA(
        d_image, width, height,
        m_inputBuffer, m_inputWidth, m_inputHeight,
        nullptr
    );
    cudaFree(d_image);

    // Set input/output and run inference
    auto inputNames = m_engine->getInputNames();
    auto outputNames = m_engine->getOutputNames();

    nvinfer1::Dims4 dims(1, m_inputChannels, m_inputHeight, m_inputWidth);
    m_engine->setInput(inputNames[0], m_inputBuffer, dims);
    m_engine->setOutput(outputNames[0], m_outputBuffer);

    if (!m_engine->inferSync()) {
        m_lastError = "Inference failed";
        return {};
    }

    return postprocess(width, height);
}

std::vector<DetectionResult> YOLODetector::postprocess(
    int origWidth, int origHeight) {

    // Copy output to host
    std::vector<float> output(m_outputBufferSize);
    cudaMemcpy(output.data(), m_outputBuffer,
               m_outputBufferSize * sizeof(float), cudaMemcpyDeviceToHost);

    std::vector<DetectionResult> detections;

    // YOLOv10 with built-in NMS outputs [1, N, 6]:
    // [x1, y1, x2, y2, confidence, class_id]
    // For end2end models, N=300 max detections
    //
    // Scale factors to map from 640x640 back to original
    float scaleX = static_cast<float>(origWidth) / m_inputWidth;
    float scaleY = static_cast<float>(origHeight) / m_inputHeight;

    // Determine output format from buffer size
    // end2end: 300 * 6 = 1800
    // raw: 84 * 8400 = 705600 (for COCO 80 classes + 4 coords)
    size_t numDetections = m_outputBufferSize / 6;
    if (numDetections > 300) numDetections = 300; // Limit for end2end

    for (size_t i = 0; i < numDetections; i++) {
        float* det = output.data() + i * 6;
        float x1 = det[0];
        float y1 = det[1];
        float x2 = det[2];
        float y2 = det[3];
        float conf = det[4];
        int classId = static_cast<int>(det[5]);

        // Skip low confidence
        if (conf < m_confThreshold) continue;

        // Skip invalid boxes
        if (x2 <= x1 || y2 <= y1) continue;

        DetectionResult result;
        result.classId = classId;
        result.confidence = conf;
        result.x = ((x1 + x2) / 2.0f) * scaleX;
        result.y = ((y1 + y2) / 2.0f) * scaleY;
        result.width = (x2 - x1) * scaleX;
        result.height = (y2 - y1) * scaleY;

        if (classId >= 0 && classId < static_cast<int>(m_classNames.size())) {
            result.className = m_classNames[classId];
        } else {
            result.className = "class_" + std::to_string(classId);
        }

        detections.push_back(result);
    }

    // Apply NMS (may be redundant for end2end models but ensures consistency)
    return applyNMS(detections);
}

} // namespace trading_monitor

