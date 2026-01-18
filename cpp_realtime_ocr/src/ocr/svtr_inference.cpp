/**
 * @file svtr_inference.cpp
 * @brief SVTR-tiny OCR model inference
 *
 * Wraps TensorRTEngine for text recognition
 */

#include "ocr/svtr_inference.h"
#include <cuda_runtime.h>
#include <iostream>

namespace trading_monitor {

static size_t dimsVolume(const nvinfer1::Dims& dims) {
    size_t v = 1;
    for (int i = 0; i < dims.nbDims; i++) {
        if (dims.d[i] < 0) return 0;
        v *= static_cast<size_t>(dims.d[i]);
    }
    return v;
}

SVTRInference::SVTRInference()
    : m_engine(std::make_unique<TensorRTEngine>()) {}

SVTRInference::~SVTRInference() {
    if (m_inputBuffer) cudaFree(m_inputBuffer);
    if (m_outputBuffer) cudaFree(m_outputBuffer);
}

bool SVTRInference::loadModel(const std::string& enginePath) {
    if (!m_engine->loadEngine(enginePath)) {
        return false;
    }

    // Get input/output tensor info
    auto inputNames = m_engine->getInputNames();
    auto outputNames = m_engine->getOutputNames();

    if (inputNames.empty() || outputNames.empty()) {
        return false;
    }

    // Get dimensions
    auto inputDims = m_engine->getInputDims(inputNames[0]);
    auto outputDims = m_engine->getOutputDims(outputNames[0]);

    // Typical SVTR input: [batch, channels, height, width] (NCHW)
    // Paddle2ONNX export uses dynamic batch and width.
    if (inputDims.nbDims >= 4) {
        m_inputChannels = (inputDims.d[1] > 0) ? static_cast<int>(inputDims.d[1]) : 3;
        m_inputHeight = (inputDims.d[2] > 0) ? static_cast<int>(inputDims.d[2]) : 48;
        m_inputWidth = (inputDims.d[3] > 0) ? static_cast<int>(inputDims.d[3]) : 320;
    } else {
        // Fallback defaults
        m_inputChannels = 3;
        m_inputHeight = 48;
        m_inputWidth = 320;
    }

    // Allocate buffers sized for the maximum width in the optimization profile.
    // Keep reported output dims based on the "opt" width (320) for test_tensorrt.
    const int maxWidth = 1024;
    const int optWidth = 320;

    nvinfer1::Dims4 optInputDims(1, m_inputChannels, m_inputHeight, optWidth);
    nvinfer1::Dims4 maxInputDims(1, m_inputChannels, m_inputHeight, maxWidth);

    const size_t inputMaxElems = dimsVolume(maxInputDims);
    if (inputMaxElems == 0) {
        std::cerr << "SVTR: invalid input dims" << std::endl;
        return false;
    }

    cudaMalloc(&m_inputBuffer, inputMaxElems * sizeof(float));

    // Set an input shape so dynamic outputs become concrete in the context.
    if (!m_engine->setInput(inputNames[0], m_inputBuffer, optInputDims)) {
        std::cerr << "SVTR: failed to set default input shape" << std::endl;
        return false;
    }

    auto ctxOutDims = m_engine->getContextDims(outputNames[0]);
    if (ctxOutDims.nbDims >= 3 && ctxOutDims.d[1] > 0 && ctxOutDims.d[2] > 0) {
        m_outputTimesteps = static_cast<int>(ctxOutDims.d[1]);
        m_outputClasses = static_cast<int>(ctxOutDims.d[2]);
    } else if (outputDims.nbDims >= 3 && outputDims.d[1] > 0 && outputDims.d[2] > 0) {
        m_outputTimesteps = static_cast<int>(outputDims.d[1]);
        m_outputClasses = static_cast<int>(outputDims.d[2]);
    }

    // Allocate an output buffer large enough for the max profile width.
    if (!m_engine->setInput(inputNames[0], m_inputBuffer, maxInputDims)) {
        std::cerr << "SVTR: failed to set max input shape" << std::endl;
        return false;
    }
    auto maxOutDims = m_engine->getContextDims(outputNames[0]);
    const size_t outputMaxElems = dimsVolume(maxOutDims);
    if (outputMaxElems == 0) {
        std::cerr << "SVTR: invalid output dims" << std::endl;
        return false;
    }
    cudaMalloc(&m_outputBuffer, outputMaxElems * sizeof(float));

    std::cout << "SVTR model loaded: input=" << m_inputWidth << "x" << m_inputHeight
              << ", output=" << m_outputTimesteps << "x" << m_outputClasses << std::endl;

    return true;
}

bool SVTRInference::buildModel(
    const std::string& onnxPath,
    const std::string& enginePath,
    bool useFP16
) {
    if (!m_engine->buildFromONNX(onnxPath, enginePath, useFP16, 1)) {
        return false;
    }
    return loadModel(enginePath);
}

bool SVTRInference::infer(float* input, int width, int height, cudaStream_t stream) {
    if (!m_engine->isLoaded()) return false;

    auto inputNames = m_engine->getInputNames();
    auto outputNames = m_engine->getOutputNames();

    // Set input dimensions [batch=1, channels=3, height, width]
    if (m_inputHeight > 0 && height != m_inputHeight) {
        std::cerr << "SVTR: input height mismatch. Expected " << m_inputHeight
                  << ", got " << height << std::endl;
        return false;
    }
    nvinfer1::Dims4 dims(1, m_inputChannels, height, width);

    if (!m_engine->setInput(inputNames[0], input, dims)) {
        return false;
    }

    if (!m_engine->setOutput(outputNames[0], m_outputBuffer)) {
        return false;
    }

    return m_engine->inferAsync(stream);
}

float* SVTRInference::getOutputProbs() const {
    return m_outputBuffer;
}

void SVTRInference::getOutputDims(int& timesteps, int& numClasses) const {
    timesteps = m_outputTimesteps;
    numClasses = m_outputClasses;
}

bool SVTRInference::isLoaded() const {
    return m_engine && m_engine->isLoaded();
}

} // namespace trading_monitor

