/**
 * @file cuda_interop.cpp
 * @brief D3D11-CUDA interoperability implementation
 *
 * Compatible with CUDA 12.x
 */

#include "capture/cuda_interop.h"
#include <dxgi.h>
#include <wrl/client.h>
#include <iostream>
#include <sstream>

namespace trading_monitor {

CudaD3D11Interop::CudaD3D11Interop() = default;

CudaD3D11Interop::~CudaD3D11Interop() {
    if (m_stream) {
        cudaStreamDestroy(m_stream);
    }
}

bool CudaD3D11Interop::initialize(ID3D11Device* d3dDevice) {
    if (m_initialized) {
        return true;
    }

    cudaError_t err = cudaSuccess;

    // Determine the CUDA device that corresponds to the D3D11 device's adapter.
    // CUDA 13.x expects an IDXGIAdapter* (not an ID3D11Device*)
    {
        Microsoft::WRL::ComPtr<IDXGIDevice> dxgiDevice;
        Microsoft::WRL::ComPtr<IDXGIAdapter> adapter;

        HRESULT hr = d3dDevice->QueryInterface(IID_PPV_ARGS(dxgiDevice.GetAddressOf()));
        if (SUCCEEDED(hr)) {
            hr = dxgiDevice->GetAdapter(adapter.GetAddressOf());
        }

        if (SUCCEEDED(hr) && adapter) {
            err = cudaD3D11GetDevice(&m_cudaDevice, adapter.Get());
        } else {
            err = cudaErrorUnknown;
        }

        if (err != cudaSuccess) {
            m_lastError = "Failed to find CUDA device for D3D11 adapter: " +
                          std::string(cudaGetErrorString(err));
            return false;
        }
    }

    err = cudaSetDevice(m_cudaDevice);
    if (err != cudaSuccess) {
        m_lastError = "Failed to set CUDA device: " + std::string(cudaGetErrorString(err));
        return false;
    }

    // Associate the D3D11 device with CUDA interop.
    // This API is deprecated but still required/useful on many Windows setups.
    // It can fail if any CUDA runtime usage already activated a device; in that case
    // we continue and rely on the adapter/device match instead.
    {
        cudaError_t setErr = cudaD3D11SetDirect3DDevice(d3dDevice);
        if (setErr != cudaSuccess && setErr != cudaErrorSetOnActiveProcess) {
            m_lastError = "Failed to bind D3D11 device for CUDA interop: " +
                          std::string(cudaGetErrorString(setErr));
            return false;
        }
    }
    
    // Create CUDA stream
    err = cudaStreamCreate(&m_stream);
    if (err != cudaSuccess) {
        m_lastError = "Failed to create CUDA stream: " + std::string(cudaGetErrorString(err));
        return false;
    }
    
    m_initialized = true;
    return true;
}

cudaGraphicsResource_t CudaD3D11Interop::registerTexture(
    ID3D11Texture2D* texture,
    unsigned int flags
) {
    if (!m_initialized || !texture) {
        m_lastError = "Not initialized or null texture";
        return nullptr;
    }
    
    cudaGraphicsResource_t resource = nullptr;
    cudaError_t err = cudaGraphicsD3D11RegisterResource(
        &resource,
        texture,
        flags
    );
    
    if (err != cudaSuccess) {
        m_lastError = "Failed to register D3D11 resource: " + 
                      std::string(cudaGetErrorString(err));
        return nullptr;
    }
    
    return resource;
}

void CudaD3D11Interop::unregisterTexture(cudaGraphicsResource_t resource) {
    if (resource) {
        cudaGraphicsUnregisterResource(resource);
    }
}

bool CudaD3D11Interop::mapResource(cudaGraphicsResource_t resource, cudaStream_t stream) {
    if (!resource) {
        m_lastError = "Null resource";
        return false;
    }
    
    cudaError_t err = cudaGraphicsMapResources(1, &resource, stream);
    if (err != cudaSuccess) {
        m_lastError = "Failed to map resource: " + std::string(cudaGetErrorString(err));
        return false;
    }
    
    return true;
}

void CudaD3D11Interop::unmapResource(cudaGraphicsResource_t resource, cudaStream_t stream) {
    if (resource) {
        cudaGraphicsUnmapResources(1, &resource, stream);
    }
}

cudaArray_t CudaD3D11Interop::getMappedArray(
    cudaGraphicsResource_t resource,
    unsigned int arrayIndex,
    unsigned int mipLevel
) {
    cudaArray_t array = nullptr;
    cudaError_t err = cudaGraphicsSubResourceGetMappedArray(
        &array, resource, arrayIndex, mipLevel
    );
    
    if (err != cudaSuccess) {
        m_lastError = "Failed to get mapped array: " + std::string(cudaGetErrorString(err));
        return nullptr;
    }
    
    return array;
}

cudaTextureObject_t CudaD3D11Interop::createTextureObject(cudaArray_t array) {
    if (!array) {
        return 0;
    }
    
    // Resource descriptor
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = array;
    
    // Texture descriptor
    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;  // Bilinear interpolation
    texDesc.readMode = cudaReadModeNormalizedFloat;  // Return [0,1]
    texDesc.normalizedCoords = 0;  // Use pixel coordinates
    
    cudaTextureObject_t texObj = 0;
    cudaError_t err = cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);
    
    if (err != cudaSuccess) {
        m_lastError = "Failed to create texture object: " + std::string(cudaGetErrorString(err));
        return 0;
    }
    
    return texObj;
}

void CudaD3D11Interop::destroyTextureObject(cudaTextureObject_t texObj) {
    if (texObj) {
        cudaDestroyTextureObject(texObj);
    }
}

// =============================================================================
// CudaTexturePool Implementation
// =============================================================================

cudaGraphicsResource_t CudaTexturePool::getOrRegister(ID3D11Texture2D* texture) {
    if (!texture) {
        return nullptr;
    }

    // Check if already registered
    auto it = m_resources.find(texture);
    if (it != m_resources.end()) {
        return it->second;
    }

    // Register new texture
    cudaGraphicsResource_t resource = m_interop.registerTexture(
        texture,
        cudaGraphicsRegisterFlagsNone
    );

    if (resource) {
        m_resources[texture] = resource;
    }

    return resource;
}

void CudaTexturePool::unregister(ID3D11Texture2D* texture) {
    auto it = m_resources.find(texture);
    if (it != m_resources.end()) {
        m_interop.unregisterTexture(it->second);
        m_resources.erase(it);
    }
}

void CudaTexturePool::clear() {
    for (auto& pair : m_resources) {
        m_interop.unregisterTexture(pair.second);
    }
    m_resources.clear();
}

} // namespace trading_monitor

