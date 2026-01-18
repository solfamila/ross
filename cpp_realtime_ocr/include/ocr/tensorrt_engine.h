#pragma once
/**
 * @file tensorrt_engine.h
 * @brief TensorRT engine wrapper for OCR inference
 * 
 * Compatible with TensorRT 10.14
 */

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4100) // unreferenced parameter (TensorRT headers)
#endif

#include <NvInfer.h>

#ifdef _MSC_VER
#pragma warning(pop)
#endif
#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

namespace trading_monitor {

/**
 * @brief Custom TensorRT logger
 */
class TRTLogger : public nvinfer1::ILogger {
public:
    explicit TRTLogger(Severity minSeverity = Severity::kWARNING);
    void log(Severity severity, const char* msg) noexcept override;
    
    void setVerbose(bool verbose) { m_verbose = verbose; }
    
private:
    Severity m_minSeverity;
    bool m_verbose = false;
};

/**
 * @brief TensorRT engine wrapper for efficient inference
 * 
 * Handles engine loading, memory allocation, and asynchronous inference.
 * Supports TensorRT 10.14 API including tensor-based I/O.
 */
class TensorRTEngine {
public:
    TensorRTEngine();
    ~TensorRTEngine();
    
    // Non-copyable
    TensorRTEngine(const TensorRTEngine&) = delete;
    TensorRTEngine& operator=(const TensorRTEngine&) = delete;
    
    /**
     * @brief Load a serialized TensorRT engine
     * 
     * @param enginePath Path to .engine file
     * @return true if loading successful
     */
    bool loadEngine(const std::string& enginePath);
    
    /**
     * @brief Build engine from ONNX model
     * 
     * @param onnxPath Path to ONNX model
     * @param enginePath Path to save the engine
     * @param useFP16 Enable FP16 precision
     * @param maxBatchSize Maximum batch size
     * @return true if build successful
     */
    bool buildFromONNX(
        const std::string& onnxPath,
        const std::string& enginePath,
        bool useFP16 = true,
        int maxBatchSize = 1
    );
    
    /**
     * @brief Set input tensor data
     * 
     * @param name Input tensor name
     * @param data Device pointer to input data
     * @param dims Input dimensions (must match profile)
     * @return true if successful
     */
    bool setInput(
        const std::string& name,
        void* data,
        const nvinfer1::Dims& dims
    );
    
    /**
     * @brief Set output buffer
     * 
     * @param name Output tensor name
     * @param data Device pointer for output
     * @return true if successful
     */
    bool setOutput(const std::string& name, void* data);
    
    /**
     * @brief Execute inference asynchronously
     * 
     * @param stream CUDA stream for execution
     * @return true if enqueue successful
     */
    bool inferAsync(cudaStream_t stream);
    
    /**
     * @brief Execute inference synchronously
     * 
     * @return true if inference successful
     */
    bool inferSync();
    
    /**
     * @brief Get input tensor shape
     */
    nvinfer1::Dims getInputDims(const std::string& name) const;
    
    /**
     * @brief Get output tensor shape
     */
    nvinfer1::Dims getOutputDims(const std::string& name) const;

    /**
     * @brief Get tensor shape from the execution context (after setting input shapes)
     *
     * For dynamic networks, the engine's tensor shapes may include -1.
     * The execution context provides concrete shapes after `setInputShape`.
     */
    nvinfer1::Dims getContextDims(const std::string& name) const;
    
    /**
     * @brief Get all input tensor names
     */
    std::vector<std::string> getInputNames() const;
    
    /**
     * @brief Get all output tensor names
     */
    std::vector<std::string> getOutputNames() const;
    
    /**
     * @brief Check if engine is loaded
     */
    bool isLoaded() const { return m_engine != nullptr; }
    
    /**
     * @brief Get last error message
     */
    const std::string& getLastError() const { return m_lastError; }

private:
    void cleanup();
    
    TRTLogger m_logger;
    std::unique_ptr<nvinfer1::IRuntime> m_runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> m_engine;
    std::unique_ptr<nvinfer1::IExecutionContext> m_context;
    
    std::unordered_map<std::string, int> m_tensorIndices;
    std::string m_lastError;
};

} // namespace trading_monitor

