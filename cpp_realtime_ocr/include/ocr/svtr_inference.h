#pragma once
/**
 * @file svtr_inference.h
 * @brief SVTR-tiny OCR model inference wrapper
 */

#include "ocr/tensorrt_engine.h"
#include <memory>
#include <vector>
#include <string>

namespace trading_monitor {

/**
 * @brief SVTR-tiny text recognition inference
 * 
 * Wraps TensorRT engine for text recognition with proper
 * input preprocessing and output handling.
 */
class SVTRInference {
public:
    SVTRInference();
    ~SVTRInference();
    
    /**
     * @brief Load SVTR model
     * @param enginePath Path to TensorRT engine
     * @return true if successful
     */
    bool loadModel(const std::string& enginePath);
    
    /**
     * @brief Build engine from ONNX
     * @param onnxPath Path to ONNX model
     * @param enginePath Path to save engine
     * @param useFP16 Enable FP16 precision
     * @return true if successful
     */
    bool buildModel(
        const std::string& onnxPath,
        const std::string& enginePath,
        bool useFP16 = true
    );
    
    /**
     * @brief Run inference on preprocessed image
     * @param input Device pointer to preprocessed grayscale image
     * @param width Image width
     * @param height Image height
     * @param stream CUDA stream
     * @return true if inference successful
     */
    bool infer(float* input, int width, int height, cudaStream_t stream = 0);
    
    /**
     * @brief Get output probabilities
     * @return Pointer to output buffer (device memory)
     */
    float* getOutputProbs() const;
    
    /**
     * @brief Get output dimensions
     * @param timesteps Number of output timesteps
     * @param numClasses Number of character classes
     */
    void getOutputDims(int& timesteps, int& numClasses) const;
    
    /**
     * @brief Check if model is loaded
     */
    bool isLoaded() const;
    
private:
    std::unique_ptr<TensorRTEngine> m_engine;
    float* m_inputBuffer = nullptr;
    float* m_outputBuffer = nullptr;
    int m_inputChannels = 3;
    int m_inputWidth = 0;
    int m_inputHeight = 0;
    int m_outputTimesteps = 0;
    int m_outputClasses = 0;
};

} // namespace trading_monitor

