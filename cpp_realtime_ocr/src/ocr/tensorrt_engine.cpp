/**
 * @file tensorrt_engine.cpp
 * @brief TensorRT engine implementation
 * 
 * Compatible with TensorRT 10.14
 */

#include "ocr/tensorrt_engine.h"
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4100) // unreferenced parameter (TensorRT headers)
#endif

#include <NvOnnxParser.h>

#ifdef _MSC_VER
#pragma warning(pop)
#endif
#include <fstream>
#include <iostream>
#include <string>

namespace trading_monitor {

// =============================================================================
// TRTLogger Implementation
// =============================================================================

TRTLogger::TRTLogger(Severity minSeverity) 
    : m_minSeverity(minSeverity) {}

void TRTLogger::log(Severity severity, const char* msg) noexcept {
    if (severity <= m_minSeverity || m_verbose) {
        const char* severityStr = "";
        switch (severity) {
            case Severity::kINTERNAL_ERROR: severityStr = "[INTERNAL_ERROR]"; break;
            case Severity::kERROR: severityStr = "[ERROR]"; break;
            case Severity::kWARNING: severityStr = "[WARNING]"; break;
            case Severity::kINFO: severityStr = "[INFO]"; break;
            case Severity::kVERBOSE: severityStr = "[VERBOSE]"; break;
        }
        std::cout << "[TRT]" << severityStr << " " << msg << std::endl;
    }
}

// =============================================================================
// TensorRTEngine Implementation
// =============================================================================

TensorRTEngine::TensorRTEngine() = default;

TensorRTEngine::~TensorRTEngine() {
    cleanup();
}

void TensorRTEngine::cleanup() {
    m_context.reset();
    m_engine.reset();
    m_runtime.reset();
}

bool TensorRTEngine::loadEngine(const std::string& enginePath) {
    cleanup();
    
    // Read engine file
    std::ifstream file(enginePath, std::ios::binary);
    if (!file.good()) {
        m_lastError = "Cannot open engine file: " + enginePath;
        return false;
    }
    
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    std::vector<char> engineData(size);
    file.read(engineData.data(), size);
    file.close();
    
    // Create runtime
    m_runtime.reset(nvinfer1::createInferRuntime(m_logger));
    if (!m_runtime) {
        m_lastError = "Failed to create TensorRT runtime";
        return false;
    }
    
    // Deserialize engine
    m_engine.reset(m_runtime->deserializeCudaEngine(engineData.data(), size));
    if (!m_engine) {
        m_lastError = "Failed to deserialize engine";
        return false;
    }
    
    // Create execution context
    m_context.reset(m_engine->createExecutionContext());
    if (!m_context) {
        m_lastError = "Failed to create execution context";
        return false;
    }
    
    // Build tensor index map
    for (int i = 0; i < m_engine->getNbIOTensors(); i++) {
        const char* name = m_engine->getIOTensorName(i);
        m_tensorIndices[name] = i;
    }
    
    return true;
}

bool TensorRTEngine::buildFromONNX(
    const std::string& onnxPath,
    const std::string& enginePath,
    bool useFP16,
    int /*maxBatchSize*/  // Note: TRT 10.x uses explicit batch in network definition
) {
    // Create builder
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(
        nvinfer1::createInferBuilder(m_logger));
    if (!builder) {
        m_lastError = "Failed to create builder";
        return false;
    }
    
    // Create network (explicit batch)
    const uint32_t explicitBatch = 1U << static_cast<uint32_t>(
        nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(
        builder->createNetworkV2(explicitBatch));
    if (!network) {
        m_lastError = "Failed to create network";
        return false;
    }
    
    // Create parser
    auto parser = std::unique_ptr<nvonnxparser::IParser>(
        nvonnxparser::createParser(*network, m_logger));
    if (!parser) {
        m_lastError = "Failed to create ONNX parser";
        return false;
    }
    
    // Parse ONNX file
    if (!parser->parseFromFile(onnxPath.c_str(), 
            static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        m_lastError = "Failed to parse ONNX file: " + onnxPath;
        return false;
    }
    
    // Create builder config
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(
        builder->createBuilderConfig());
    
    // Set memory pool limits (TensorRT 10.14)
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30);
    
    // Enable FP16 if requested and supported
    if (useFP16 && builder->platformHasFastFp16()) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
        std::cout << "Enabled FP16 precision" << std::endl;

        // Best-effort accuracy guardrail for LayerNorm-heavy transformer blocks:
        // force numerically sensitive ops (reduce/pow) to run in FP32.
        // This often removes TensorRT's overflow warnings while keeping most layers FP16.
        config->setFlag(nvinfer1::BuilderFlag::kOBEY_PRECISION_CONSTRAINTS);
        for (int i = 0; i < network->getNbLayers(); i++) {
            auto* layer = network->getLayer(i);
            if (!layer) continue;

            const auto type = layer->getType();
            const char* lname = layer->getName();
            const std::string name = (lname != nullptr) ? std::string(lname) : std::string();

            const bool looksLikeLayerNorm =
                (name.find("layer_norm") != std::string::npos) ||
                (name.find("LayerNorm") != std::string::npos) ||
                (name.find("layernorm") != std::string::npos);

            const bool forceFp32 = looksLikeLayerNorm ||
                (type == nvinfer1::LayerType::kREDUCE) ||
                (type == nvinfer1::LayerType::kUNARY);

            if (forceFp32) {
                layer->setPrecision(nvinfer1::DataType::kFLOAT);
                for (int o = 0; o < layer->getNbOutputs(); o++) {
                    layer->setOutputType(o, nvinfer1::DataType::kFLOAT);
                }
            }
        }
    }

    // Add optimization profile for dynamic inputs (required for TensorRT 10.x).
    // Paddle2ONNX export uses input name "x" with dynamic batch and width.
    // Default profile matches the Python pipeline defaults.
    {
        const int nbInputs = network->getNbInputs();
        bool needsProfile = false;
        for (int i = 0; i < nbInputs; i++) {
            const auto* t = network->getInput(i);
            if (!t) continue;
            const auto dims = t->getDimensions();
            for (int d = 0; d < dims.nbDims; d++) {
                if (dims.d[d] == -1) {
                    needsProfile = true;
                    break;
                }
            }
            if (t->isShapeTensor()) {
                needsProfile = true;
            }
        }

        if (needsProfile) {
            auto* profile = builder->createOptimizationProfile();
            if (!profile) {
                m_lastError = "Failed to create optimization profile";
                return false;
            }

            for (int i = 0; i < nbInputs; i++) {
                const auto* input = network->getInput(i);
                if (!input) continue;
                const char* name = input->getName();
                const auto dims = input->getDimensions();

                if (input->isShapeTensor()) {
                    // No shape-tensor inputs expected for this model.
                    m_lastError = std::string("Shape-tensor input not supported in this wrapper: ") + name;
                    return false;
                }

                if (dims.nbDims == 4 && std::string(name) == "x") {
                    // NCHW
                    nvinfer1::Dims4 minDims(1, 3, 48, 32);
                    nvinfer1::Dims4 optDims(1, 3, 48, 320);
                    nvinfer1::Dims4 maxDims(1, 3, 48, 1024);
                    if (!profile->setDimensions(name, nvinfer1::OptProfileSelector::kMIN, minDims) ||
                        !profile->setDimensions(name, nvinfer1::OptProfileSelector::kOPT, optDims) ||
                        !profile->setDimensions(name, nvinfer1::OptProfileSelector::kMAX, maxDims)) {
                        m_lastError = std::string("Failed to set optimization profile dims for input: ") + name;
                        return false;
                    }
                } else {
                    // Generic fallback: clamp dynamic dims to 1.
                    nvinfer1::Dims minDims = dims;
                    nvinfer1::Dims optDims = dims;
                    nvinfer1::Dims maxDims = dims;
                    for (int d = 0; d < dims.nbDims; d++) {
                        if (dims.d[d] == -1) {
                            minDims.d[d] = 1;
                            optDims.d[d] = 1;
                            maxDims.d[d] = 1;
                        }
                    }
                    if (!profile->setDimensions(name, nvinfer1::OptProfileSelector::kMIN, minDims) ||
                        !profile->setDimensions(name, nvinfer1::OptProfileSelector::kOPT, optDims) ||
                        !profile->setDimensions(name, nvinfer1::OptProfileSelector::kMAX, maxDims)) {
                        m_lastError = std::string("Failed to set optimization profile dims for input: ") + name;
                        return false;
                    }
                }
            }

            if (!profile->isValid()) {
                m_lastError = "Optimization profile is invalid";
                return false;
            }
            config->addOptimizationProfile(profile);
        }
    }
    
    // Build serialized engine
    auto serializedEngine = std::unique_ptr<nvinfer1::IHostMemory>(
        builder->buildSerializedNetwork(*network, *config));
    if (!serializedEngine) {
        m_lastError = "Failed to build engine";
        return false;
    }
    
    // Save to file
    std::ofstream engineFile(enginePath, std::ios::binary);
    engineFile.write(reinterpret_cast<const char*>(serializedEngine->data()),
                     serializedEngine->size());
    engineFile.close();
    
    std::cout << "Engine saved to: " << enginePath << std::endl;
    
    // Load the engine we just built
    return loadEngine(enginePath);
}

bool TensorRTEngine::setInput(
    const std::string& name,
    void* data,
    const nvinfer1::Dims& dims
) {
    if (!m_context) {
        m_lastError = "No execution context";
        return false;
    }

    // Set input shape for dynamic dimensions
    if (!m_context->setInputShape(name.c_str(), dims)) {
        m_lastError = "Failed to set input shape for: " + name;
        return false;
    }

    // Set tensor address (TensorRT 10.x API)
    m_context->setTensorAddress(name.c_str(), data);
    return true;
}

bool TensorRTEngine::setOutput(const std::string& name, void* data) {
    if (!m_context) {
        m_lastError = "No execution context";
        return false;
    }

    m_context->setTensorAddress(name.c_str(), data);
    return true;
}

bool TensorRTEngine::inferAsync(cudaStream_t stream) {
    if (!m_context) {
        m_lastError = "No execution context";
        return false;
    }

    return m_context->enqueueV3(stream);
}

bool TensorRTEngine::inferSync() {
    if (!m_context) {
        m_lastError = "No execution context";
        return false;
    }

    bool success = m_context->enqueueV3(nullptr);
    cudaDeviceSynchronize();
    return success;
}

nvinfer1::Dims TensorRTEngine::getInputDims(const std::string& name) const {
    if (m_engine) {
        return m_engine->getTensorShape(name.c_str());
    }
    return nvinfer1::Dims{};
}

nvinfer1::Dims TensorRTEngine::getOutputDims(const std::string& name) const {
    if (m_engine) {
        return m_engine->getTensorShape(name.c_str());
    }
    return nvinfer1::Dims{};
}

nvinfer1::Dims TensorRTEngine::getContextDims(const std::string& name) const {
    if (m_context) {
        return m_context->getTensorShape(name.c_str());
    }
    if (m_engine) {
        return m_engine->getTensorShape(name.c_str());
    }
    return nvinfer1::Dims{};
}

std::vector<std::string> TensorRTEngine::getInputNames() const {
    std::vector<std::string> names;
    if (m_engine) {
        for (int i = 0; i < m_engine->getNbIOTensors(); i++) {
            const char* name = m_engine->getIOTensorName(i);
            if (m_engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
                names.push_back(name);
            }
        }
    }
    return names;
}

std::vector<std::string> TensorRTEngine::getOutputNames() const {
    std::vector<std::string> names;
    if (m_engine) {
        for (int i = 0; i < m_engine->getNbIOTensors(); i++) {
            const char* name = m_engine->getIOTensorName(i);
            if (m_engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT) {
                names.push_back(name);
            }
        }
    }
    return names;
}

} // namespace trading_monitor

