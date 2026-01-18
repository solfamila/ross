/**
 * @file d3d11_capture.cpp
 * @brief Windows.Graphics.Capture implementation
 *
 * Based on robmikh/Win32CaptureSample
 * Uses CreateFreeThreaded for low-latency frame delivery
 */

#ifndef NOMINMAX
#define NOMINMAX
#endif

// Must come before Windows headers
#include <Unknwn.h>

// Windows/WinRT headers
#include <Windows.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.Graphics.Capture.h>
#include <winrt/Windows.Graphics.DirectX.h>
#include <winrt/Windows.Graphics.DirectX.Direct3D11.h>
#include <windows.graphics.capture.interop.h>
#include <windows.graphics.directx.direct3d11.interop.h>

#include "capture/d3d11_capture.h"

// CUDA Driver API (used only to pick the correct DXGI adapter for interop)
#include <cuda.h>

#include <atomic>
#include <algorithm>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>
#include <iostream>

namespace winrt {
    using namespace Windows::Foundation;
    using namespace Windows::Graphics;
    using namespace Windows::Graphics::Capture;
    using namespace Windows::Graphics::DirectX;
    using namespace Windows::Graphics::DirectX::Direct3D11;
}

namespace trading_monitor {

// =============================================================================
// Helper Functions
// =============================================================================

// Convert HWND to GraphicsCaptureItem
winrt::GraphicsCaptureItem CreateCaptureItemForWindow(HWND hwnd) {
    auto factory = winrt::get_activation_factory<winrt::GraphicsCaptureItem>();
    auto interop = factory.as<IGraphicsCaptureItemInterop>();

    winrt::GraphicsCaptureItem item{ nullptr };
    winrt::check_hresult(interop->CreateForWindow(
        hwnd,
        winrt::guid_of<ABI::Windows::Graphics::Capture::IGraphicsCaptureItem>(),
        winrt::put_abi(item)
    ));
    return item;
}

// Convert HMONITOR to GraphicsCaptureItem
winrt::GraphicsCaptureItem CreateCaptureItemForMonitor(HMONITOR hmon) {
    auto factory = winrt::get_activation_factory<winrt::GraphicsCaptureItem>();
    auto interop = factory.as<IGraphicsCaptureItemInterop>();

    winrt::GraphicsCaptureItem item{ nullptr };
    winrt::check_hresult(interop->CreateForMonitor(
        hmon,
        winrt::guid_of<ABI::Windows::Graphics::Capture::IGraphicsCaptureItem>(),
        winrt::put_abi(item)
    ));
    return item;
}

// Create IDirect3DDevice from ID3D11Device
winrt::IDirect3DDevice CreateDirect3DDevice(ID3D11Device* d3dDevice) {
    winrt::com_ptr<IDXGIDevice> dxgiDevice;
    winrt::check_hresult(d3dDevice->QueryInterface(IID_PPV_ARGS(dxgiDevice.put())));

    winrt::com_ptr<::IInspectable> inspectable;
    winrt::check_hresult(CreateDirect3D11DeviceFromDXGIDevice(
        dxgiDevice.get(),
        inspectable.put()
    ));

    return inspectable.as<winrt::IDirect3DDevice>();
}

// Get ID3D11Texture2D from IDirect3DSurface
winrt::com_ptr<ID3D11Texture2D> GetTextureFromSurface(winrt::IDirect3DSurface const& surface) {
    auto access = surface.as<Windows::Graphics::DirectX::Direct3D11::IDirect3DDxgiInterfaceAccess>();

    winrt::com_ptr<ID3D11Texture2D> texture;
    winrt::check_hresult(access->GetInterface(IID_PPV_ARGS(texture.put())));
    return texture;
}

// =============================================================================
// D3D11Capture::Impl
// =============================================================================

class D3D11Capture::Impl {
public:
    Impl() = default;
    ~Impl() { stopCapture(); }

    bool initializeDevice() {
        if (m_d3dDevice) return true;

        D3D_FEATURE_LEVEL featureLevels[] = {
            D3D_FEATURE_LEVEL_11_1,
            D3D_FEATURE_LEVEL_11_0
        };

        UINT flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
#ifdef _DEBUG
        flags |= D3D11_CREATE_DEVICE_DEBUG;
#endif

        // If we have a CUDA device, try to create the D3D11 device on the matching DXGI adapter.
        // This avoids cudaGraphicsD3D11RegisterResource failing when D3D and CUDA are on different GPUs.
        winrt::com_ptr<IDXGIAdapter1> preferredAdapter;
        {
            if (cuInit(0) == CUDA_SUCCESS) {
                int deviceCount = 0;
                if (cuDeviceGetCount(&deviceCount) == CUDA_SUCCESS && deviceCount > 0) {
                    CUdevice cuDev{};
                    if (cuDeviceGet(&cuDev, 0) == CUDA_SUCCESS) {
                        char luidBytes[8] = {};
                        unsigned int nodeMask = 0;
                        if (cuDeviceGetLuid(luidBytes, &nodeMask, cuDev) == CUDA_SUCCESS) {
                            LUID cudaLuid{};
                            static_assert(sizeof(cudaLuid) == 8, "Unexpected LUID size");
                            std::memcpy(&cudaLuid, luidBytes, sizeof(cudaLuid));

                            winrt::com_ptr<IDXGIFactory1> factory;
                            if (SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(factory.put())))) {
                                for (UINT i = 0;; ++i) {
                                    winrt::com_ptr<IDXGIAdapter1> adapter;
                                    if (factory->EnumAdapters1(i, adapter.put()) == DXGI_ERROR_NOT_FOUND) {
                                        break;
                                    }

                                    DXGI_ADAPTER_DESC1 desc{};
                                    if (SUCCEEDED(adapter->GetDesc1(&desc))) {
                                        if (desc.AdapterLuid.HighPart == cudaLuid.HighPart &&
                                            desc.AdapterLuid.LowPart == cudaLuid.LowPart) {
                                            preferredAdapter = adapter;
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        HRESULT hr = E_FAIL;
        if (preferredAdapter) {
            hr = D3D11CreateDevice(
                preferredAdapter.get(),
                D3D_DRIVER_TYPE_UNKNOWN,
                nullptr,
                flags,
                featureLevels,
                ARRAYSIZE(featureLevels),
                D3D11_SDK_VERSION,
                m_d3dDevice.put(),
                nullptr,
                m_d3dContext.put()
            );
        }

        if (FAILED(hr)) {
            hr = D3D11CreateDevice(
                nullptr,                    // Use default adapter
                D3D_DRIVER_TYPE_HARDWARE,
                nullptr,
                flags,
                featureLevels,
                ARRAYSIZE(featureLevels),
                D3D11_SDK_VERSION,
                m_d3dDevice.put(),
                nullptr,
                m_d3dContext.put()
            );
        }

        if (FAILED(hr)) {
            m_lastError = "Failed to create D3D11 device: " + std::to_string(hr);
            return false;
        }

        // Create WinRT device wrapper
        m_device = CreateDirect3DDevice(m_d3dDevice.get());

        return true;
    }

    bool initializeWindow(const std::string& title) {
        // Find window by partial title match
        HWND foundHwnd = nullptr;
        std::string searchTitle = title;

        auto enumParams = std::pair<std::string*, HWND*>{ &searchTitle, &foundHwnd };

        EnumWindows([](HWND hwnd, LPARAM lParam) -> BOOL {
            auto* params = reinterpret_cast<std::pair<std::string*, HWND*>*>(lParam);

            char windowTitle[256];
            GetWindowTextA(hwnd, windowTitle, sizeof(windowTitle));

            if (strlen(windowTitle) > 0 &&
                strstr(windowTitle, params->first->c_str()) != nullptr) {
                *params->second = hwnd;
                return FALSE;  // Stop enumeration
            }
            return TRUE;
        }, reinterpret_cast<LPARAM>(&enumParams));

        if (!foundHwnd) {
            m_lastError = "Window not found: " + title;
            return false;
        }

        return initializeWindow(foundHwnd);
    }

    bool initializeWindow(HWND hwnd) {
        if (!m_d3dDevice && !initializeDevice()) {
            return false;
        }

        try {
            m_captureItem = CreateCaptureItemForWindow(hwnd);
            m_frameSize = m_captureItem.Size();
            return true;
        } catch (winrt::hresult_error const& ex) {
            m_lastError = "Failed to create capture item: " +
                          winrt::to_string(ex.message());
            return false;
        }
    }

    bool initializeMonitor(int index = 0) {
        if (!m_d3dDevice && !initializeDevice()) {
            return false;
        }

        // Enumerate monitors
        std::vector<HMONITOR> monitors;
        EnumDisplayMonitors(nullptr, nullptr, [](HMONITOR hmon, HDC, LPRECT, LPARAM lParam) -> BOOL {
            auto* vec = reinterpret_cast<std::vector<HMONITOR>*>(lParam);
            vec->push_back(hmon);
            return TRUE;
        }, reinterpret_cast<LPARAM>(&monitors));

        if (index >= static_cast<int>(monitors.size())) {
            m_lastError = "Monitor index out of range";
            return false;
        }

        try {
            m_captureItem = CreateCaptureItemForMonitor(monitors[index]);
            m_frameSize = m_captureItem.Size();
            return true;
        } catch (winrt::hresult_error const& ex) {
            m_lastError = "Failed to create capture item: " +
                          winrt::to_string(ex.message());
            return false;
        }
    }

    void startCapture(FrameCallback callback) {
        if (m_capturing) return;
        if (!m_captureItem) {
            m_lastError = "No capture item initialized";
            return;
        }

        m_callback = std::move(callback);
        m_frameNumber = 0;
        m_stats = {};

        try {
            // Create frame pool with CreateFreeThreaded for low-latency
            m_framePool = winrt::Direct3D11CaptureFramePool::CreateFreeThreaded(
                m_device,
                winrt::DirectXPixelFormat::B8G8R8A8UIntNormalized,
                m_framePoolSize,
                m_frameSize
            );

            // Create capture session
            m_session = m_framePool.CreateCaptureSession(m_captureItem);

            // Configure session
            m_session.IsCursorCaptureEnabled(m_cursorEnabled);
            if (m_session.IsBorderRequired() != m_borderEnabled) {
                m_session.IsBorderRequired(m_borderEnabled);
            }

            // Subscribe to frame arrived event
            m_frameArrivedToken = m_framePool.FrameArrived(
                [this](winrt::Direct3D11CaptureFramePool const& sender,
                       winrt::IInspectable const&) {
                    onFrameArrived(sender);
                }
            );

            // Start capture
            m_session.StartCapture();
            m_capturing = true;

            QueryPerformanceCounter(&m_lastFrameTime);
            QueryPerformanceFrequency(&m_qpcFrequency);

        } catch (winrt::hresult_error const& ex) {
            m_lastError = "Failed to start capture: " +
                          winrt::to_string(ex.message());
        }
    }

    void stopCapture() {
        if (!m_capturing) return;
        m_capturing = false;

        if (m_framePool) {
            m_framePool.FrameArrived(m_frameArrivedToken);
        }

        if (m_session) {
            m_session.Close();
            m_session = nullptr;
        }

        if (m_framePool) {
            m_framePool.Close();
            m_framePool = nullptr;
        }

        m_captureItem = nullptr;
    }

    void onFrameArrived(winrt::Direct3D11CaptureFramePool const& sender) {
        auto frame = sender.TryGetNextFrame();
        if (!frame) return;

        m_stats.framesReceived++;

        // Timing
        LARGE_INTEGER now;
        QueryPerformanceCounter(&now);
        double frameTimeMs = (now.QuadPart - m_lastFrameTime.QuadPart) * 1000.0 / m_qpcFrequency.QuadPart;
        m_stats.lastFrameTimeMs = frameTimeMs;
        m_stats.avgFrameTimeMs = (m_stats.avgFrameTimeMs * (m_stats.framesReceived - 1) + frameTimeMs) / m_stats.framesReceived;
        m_lastFrameTime = now;

        // Check for resize
        auto contentSize = frame.ContentSize();
        bool needsResize = (contentSize.Width != m_frameSize.Width) ||
                          (contentSize.Height != m_frameSize.Height);

        // Get texture from frame
        auto surface = frame.Surface();
        auto texture = GetTextureFromSurface(surface);

        // Invoke callback
        if (m_callback && texture) {
            D3D11_TEXTURE2D_DESC desc;
            texture->GetDesc(&desc);

            // Ensure we have a CUDA-compatible staging texture
            // WGC textures have SHARED_NTHANDLE flags incompatible with CUDA
            ensureStagingTexture(desc.Width, desc.Height);

            // Copy captured frame to staging texture for CUDA interop
            ID3D11Texture2D* textureForCallback = texture.get();
            if (m_cudaStagingTexture) {
                m_d3dContext->CopyResource(m_cudaStagingTexture.get(), texture.get());
                textureForCallback = m_cudaStagingTexture.get();
            }

            CaptureFrame captureFrame;
            captureFrame.texture = textureForCallback;
            captureFrame.width = desc.Width;
            captureFrame.height = desc.Height;
            captureFrame.timestamp = now.QuadPart;
            captureFrame.frameNumber = m_frameNumber++;

            m_callback(captureFrame);
        }

        // Handle resize after callback
        if (needsResize) {
            m_frameSize = contentSize;
            m_framePool.Recreate(
                m_device,
                winrt::DirectXPixelFormat::B8G8R8A8UIntNormalized,
                m_framePoolSize,
                m_frameSize
            );
        }
    }

    ID3D11Device* getDevice() const { return m_d3dDevice.get(); }
    ID3D11DeviceContext* getDeviceContext() const { return m_d3dContext.get(); }
    bool isCapturing() const { return m_capturing; }
    CaptureStats getStats() const { return m_stats; }
    const std::string& getLastError() const { return m_lastError; }

    void getFrameSize(int& w, int& h) const {
        w = m_frameSize.Width;
        h = m_frameSize.Height;
    }

    void setFramePoolSize(int size) { m_framePoolSize = (std::max)(1, (std::min)(size, 4)); }
    void setCursorEnabled(bool enabled) { m_cursorEnabled = enabled; }
    void setBorderEnabled(bool enabled) { m_borderEnabled = enabled; }

private:
    // D3D11 resources
    winrt::com_ptr<ID3D11Device> m_d3dDevice;
    winrt::com_ptr<ID3D11DeviceContext> m_d3dContext;
    winrt::IDirect3DDevice m_device{ nullptr };

    // Capture resources
    winrt::GraphicsCaptureItem m_captureItem{ nullptr };
    winrt::Direct3D11CaptureFramePool m_framePool{ nullptr };
    winrt::GraphicsCaptureSession m_session{ nullptr };
    winrt::event_token m_frameArrivedToken;

    // State
    std::atomic<bool> m_capturing{ false };
    winrt::SizeInt32 m_frameSize{};
    FrameCallback m_callback;
    uint64_t m_frameNumber = 0;

    // Settings
    int m_framePoolSize = 2;
    bool m_cursorEnabled = false;
    bool m_borderEnabled = false;

    // Timing
    LARGE_INTEGER m_lastFrameTime{};
    LARGE_INTEGER m_qpcFrequency{};
    CaptureStats m_stats{};

    std::string m_lastError;

    // CUDA-compatible staging texture
    // Windows.Graphics.Capture textures have SHARED_NTHANDLE/KEYEDMUTEX flags
    // which are incompatible with cudaGraphicsD3D11RegisterResource.
    // We copy to this staging texture which has D3D11_BIND_SHADER_RESOURCE.
    winrt::com_ptr<ID3D11Texture2D> m_cudaStagingTexture;
    int m_stagingWidth = 0;
    int m_stagingHeight = 0;

    void ensureStagingTexture(int width, int height) {
        if (m_cudaStagingTexture && m_stagingWidth == width && m_stagingHeight == height) {
            return; // Already have correct size
        }

        m_cudaStagingTexture = nullptr;

        D3D11_TEXTURE2D_DESC desc = {};
        desc.Width = width;
        desc.Height = height;
        desc.MipLevels = 1;
        desc.ArraySize = 1;
        desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        desc.SampleDesc.Count = 1;
        desc.SampleDesc.Quality = 0;
        desc.Usage = D3D11_USAGE_DEFAULT;
        // D3D11_BIND_SHADER_RESOURCE is required for CUDA interop
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        desc.CPUAccessFlags = 0;
        desc.MiscFlags = 0; // No SHARED flags - this makes it CUDA-compatible

        HRESULT hr = m_d3dDevice->CreateTexture2D(&desc, nullptr, m_cudaStagingTexture.put());
        if (SUCCEEDED(hr)) {
            m_stagingWidth = width;
            m_stagingHeight = height;
        } else {
            m_lastError = "Failed to create CUDA staging texture: " + std::to_string(hr);
        }
    }
};

// =============================================================================
// D3D11Capture Public Interface
// =============================================================================

D3D11Capture::D3D11Capture() : m_impl(std::make_unique<Impl>()) {}
D3D11Capture::~D3D11Capture() = default;
D3D11Capture::D3D11Capture(D3D11Capture&&) noexcept = default;
D3D11Capture& D3D11Capture::operator=(D3D11Capture&&) noexcept = default;

bool D3D11Capture::initializeDevice() { return m_impl->initializeDevice(); }
bool D3D11Capture::initializeWindow(const std::string& title) { return m_impl->initializeWindow(title); }
bool D3D11Capture::initializeWindow(void* hwnd) { return m_impl->initializeWindow(static_cast<HWND>(hwnd)); }
bool D3D11Capture::initializeMonitor() { return m_impl->initializeMonitor(0); }
bool D3D11Capture::initializeMonitor(int index) { return m_impl->initializeMonitor(index); }
void D3D11Capture::startCapture(FrameCallback callback) { m_impl->startCapture(std::move(callback)); }
void D3D11Capture::stopCapture() { m_impl->stopCapture(); }
ID3D11Device* D3D11Capture::getDevice() const { return m_impl->getDevice(); }
ID3D11DeviceContext* D3D11Capture::getDeviceContext() const { return m_impl->getDeviceContext(); }
bool D3D11Capture::isCapturing() const { return m_impl->isCapturing(); }
CaptureStats D3D11Capture::getStats() const { return m_impl->getStats(); }
const std::string& D3D11Capture::getLastError() const { return m_impl->getLastError(); }
void D3D11Capture::getFrameSize(int& w, int& h) const { m_impl->getFrameSize(w, h); }
void D3D11Capture::setFramePoolSize(int size) { m_impl->setFramePoolSize(size); }
void D3D11Capture::setCursorEnabled(bool enabled) { m_impl->setCursorEnabled(enabled); }
void D3D11Capture::setBorderEnabled(bool enabled) { m_impl->setBorderEnabled(enabled); }

// =============================================================================
// Enumeration Helpers
// =============================================================================

std::vector<std::pair<void*, std::string>> enumerateWindows() {
    std::vector<std::pair<void*, std::string>> windows;

    EnumWindows([](HWND hwnd, LPARAM lParam) -> BOOL {
        if (!IsWindowVisible(hwnd)) return TRUE;

        char title[256];
        GetWindowTextA(hwnd, title, sizeof(title));
        if (strlen(title) > 0) {
            auto* vec = reinterpret_cast<std::vector<std::pair<void*, std::string>>*>(lParam);
            vec->emplace_back(hwnd, title);
        }
        return TRUE;
    }, reinterpret_cast<LPARAM>(&windows));

    return windows;
}

std::vector<std::pair<void*, std::string>> enumerateMonitors() {
    std::vector<std::pair<void*, std::string>> monitors;

    EnumDisplayMonitors(nullptr, nullptr, [](HMONITOR hmon, HDC, LPRECT, LPARAM lParam) -> BOOL {
        MONITORINFOEXA info;
        info.cbSize = sizeof(info);
        GetMonitorInfoA(hmon, &info);

        auto* vec = reinterpret_cast<std::vector<std::pair<void*, std::string>>*>(lParam);
        vec->emplace_back(hmon, info.szDevice);
        return TRUE;
    }, reinterpret_cast<LPARAM>(&monitors));

    return monitors;
}

} // namespace trading_monitor

