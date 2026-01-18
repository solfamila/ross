#pragma once
/**
 * @file cuda_interop.h
 * @brief D3D11-CUDA interoperability for zero-copy frame access
 *
 * Provides efficient transfer of D3D11 textures to CUDA without CPU copies.
 * Compatible with CUDA 12.x
 */

#include <d3d11.h>
#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

namespace trading_monitor {

/**
 * @brief Frame info passed to CUDA processing
 */
struct CudaFrameInfo {
    cudaTextureObject_t texObj = 0;    ///< Texture object for sampling
    cudaArray_t array = nullptr;        ///< CUDA array backing the texture
    int width = 0;                      ///< Frame width in pixels
    int height = 0;                     ///< Frame height in pixels
    uint64_t timestamp = 0;             ///< Frame timestamp (QPC)
    cudaStream_t stream = nullptr;      ///< Stream for async operations
};

/**
 * @brief CUDA-D3D11 interop manager
 * 
 * Handles registration and mapping of D3D11 textures for CUDA access.
 * Designed for frame-by-frame access pattern with minimal overhead.
 */
class CudaD3D11Interop {
public:
    CudaD3D11Interop();
    ~CudaD3D11Interop();
    
    // Non-copyable
    CudaD3D11Interop(const CudaD3D11Interop&) = delete;
    CudaD3D11Interop& operator=(const CudaD3D11Interop&) = delete;
    
    /**
     * @brief Initialize CUDA for D3D11 interop
     * 
     * @param d3dDevice D3D11 device used for capture
     * @return true if initialization successful
     */
    bool initialize(ID3D11Device* d3dDevice);
    
    /**
     * @brief Register a D3D11 texture for CUDA access
     * 
     * Should be called once per unique texture. For frame pools,
     * register each texture in the pool.
     * 
     * @param texture D3D11 texture to register
     * @param flags CUDA graphics register flags
     * @return CUDA graphics resource handle (nullptr on failure)
     */
    cudaGraphicsResource_t registerTexture(
        ID3D11Texture2D* texture,
        unsigned int flags = cudaGraphicsRegisterFlagsNone
    );
    
    /**
     * @brief Unregister a previously registered texture
     * 
     * @param resource Graphics resource to unregister
     */
    void unregisterTexture(cudaGraphicsResource_t resource);
    
    /**
     * @brief Map a registered resource for CUDA access
     * 
     * Must be called before accessing the texture data in CUDA.
     * Must be followed by unmapResource() when done.
     * 
     * @param resource Resource to map
     * @param stream CUDA stream for async operations
     * @return true if mapping successful
     */
    bool mapResource(cudaGraphicsResource_t resource, cudaStream_t stream = 0);
    
    /**
     * @brief Unmap a previously mapped resource
     * 
     * @param resource Resource to unmap
     * @param stream CUDA stream
     */
    void unmapResource(cudaGraphicsResource_t resource, cudaStream_t stream = 0);
    
    /**
     * @brief Get CUDA array from mapped resource
     * 
     * @param resource Mapped resource
     * @param arrayIndex Array index (0 for most textures)
     * @param mipLevel Mip level (0 for most cases)
     * @return CUDA array pointer
     */
    cudaArray_t getMappedArray(
        cudaGraphicsResource_t resource,
        unsigned int arrayIndex = 0,
        unsigned int mipLevel = 0
    );
    
    /**
     * @brief Create a texture object for efficient sampling
     * 
     * @param array CUDA array from getMappedArray()
     * @return Texture object handle
     */
    cudaTextureObject_t createTextureObject(cudaArray_t array);
    
    /**
     * @brief Destroy a texture object
     * 
     * @param texObj Texture object to destroy
     */
    void destroyTextureObject(cudaTextureObject_t texObj);
    
    /**
     * @brief Get CUDA stream for operations
     */
    cudaStream_t getStream() const { return m_stream; }
    
    /**
     * @brief Get last error message
     */
    const std::string& getLastError() const { return m_lastError; }

    /**
     * @brief Check if initialized
     */
    bool isInitialized() const { return m_initialized; }

    /**
     * @brief Get CUDA device ID
     */
    int getCudaDevice() const { return m_cudaDevice; }

private:
    bool m_initialized = false;
    int m_cudaDevice = 0;
    cudaStream_t m_stream = nullptr;
    std::string m_lastError;
};

/**
 * @brief RAII wrapper for mapped CUDA resource
 *
 * Automatically maps on construction and unmaps on destruction.
 * Use for exception-safe resource management.
 */
class ScopedCudaMap {
public:
    ScopedCudaMap(CudaD3D11Interop& interop, cudaGraphicsResource_t resource,
                  cudaStream_t stream = nullptr)
        : m_interop(interop), m_resource(resource), m_stream(stream), m_mapped(false)
    {
        m_mapped = m_interop.mapResource(m_resource, m_stream);
    }

    ~ScopedCudaMap() {
        if (m_mapped) {
            m_interop.unmapResource(m_resource, m_stream);
        }
    }

    // Non-copyable
    ScopedCudaMap(const ScopedCudaMap&) = delete;
    ScopedCudaMap& operator=(const ScopedCudaMap&) = delete;

    bool isMapped() const { return m_mapped; }

    cudaArray_t getArray(unsigned int arrayIndex = 0, unsigned int mipLevel = 0) {
        if (!m_mapped) return nullptr;
        return m_interop.getMappedArray(m_resource, arrayIndex, mipLevel);
    }

private:
    CudaD3D11Interop& m_interop;
    cudaGraphicsResource_t m_resource;
    cudaStream_t m_stream;
    bool m_mapped;
};

/**
 * @brief Manages a pool of registered textures for efficient reuse
 *
 * D3D11 capture frame pools typically have 2-3 textures that get reused.
 * This class caches CUDA registrations to avoid re-registering every frame.
 */
class CudaTexturePool {
public:
    CudaTexturePool(CudaD3D11Interop& interop) : m_interop(interop) {}

    ~CudaTexturePool() {
        clear();
    }

    /**
     * @brief Get or register a texture
     *
     * If the texture is already registered, returns cached resource.
     * Otherwise, registers the texture and caches the result.
     *
     * @param texture D3D11 texture to register
     * @return CUDA graphics resource (nullptr on failure)
     */
    cudaGraphicsResource_t getOrRegister(ID3D11Texture2D* texture);

    /**
     * @brief Unregister a specific texture
     */
    void unregister(ID3D11Texture2D* texture);

    /**
     * @brief Clear all registrations
     */
    void clear();

    /**
     * @brief Get number of registered textures
     */
    size_t size() const { return m_resources.size(); }

private:
    CudaD3D11Interop& m_interop;
    std::unordered_map<ID3D11Texture2D*, cudaGraphicsResource_t> m_resources;
};

} // namespace trading_monitor

