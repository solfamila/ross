#pragma once
/**
 * @file d3d11_capture.h
 * @brief Windows.Graphics.Capture wrapper for screen capture
 *
 * Based on robmikh/Win32CaptureSample
 * Uses CreateFreeThreaded for low-latency frame delivery
 */

#include <d3d11.h>
#include <dxgi1_2.h>
#include <functional>
#include <memory>
#include <string>
#include <atomic>
#include <cstdint>

namespace trading_monitor {

/**
 * @brief Frame data passed to callback
 */
struct CaptureFrame {
    ID3D11Texture2D* texture;    ///< D3D11 texture (valid only during callback)
    int width;                    ///< Frame width
    int height;                   ///< Frame height
    uint64_t timestamp;           ///< Frame timestamp (QPC value)
    uint64_t frameNumber;         ///< Sequential frame counter
};

/**
 * @brief Frame callback type
 * @param frame Frame data (texture valid only during callback)
 */
using FrameCallback = std::function<void(const CaptureFrame& frame)>;

/**
 * @brief Capture statistics
 */
struct CaptureStats {
    uint64_t framesReceived = 0;   ///< Total frames received
    uint64_t framesDropped = 0;    ///< Frames dropped due to slow processing
    double avgFrameTimeMs = 0.0;   ///< Average time between frames
    double lastFrameTimeMs = 0.0;  ///< Time since last frame
};

/**
 * @brief D3D11 screen capture using Windows.Graphics.Capture
 *
 * Key features:
 * - CreateFreeThreaded frame pool for low-latency delivery
 * - Window or monitor capture
 * - Frame callback with ID3D11Texture2D access
 * - Automatic frame pool resizing
 */
class D3D11Capture {
public:
    D3D11Capture();
    ~D3D11Capture();

    // Non-copyable, movable
    D3D11Capture(const D3D11Capture&) = delete;
    D3D11Capture& operator=(const D3D11Capture&) = delete;
    D3D11Capture(D3D11Capture&&) noexcept;
    D3D11Capture& operator=(D3D11Capture&&) noexcept;

    /**
     * @brief Initialize D3D11 device for capture
     * @return true if successful
     */
    bool initializeDevice();

    /**
     * @brief Initialize capture for a window by title (partial match)
     * @param windowTitle Window title to capture
     * @return true if window found and capture initialized
     */
    bool initializeWindow(const std::string& windowTitle);

    /**
     * @brief Initialize capture for a window by HWND
     * @param hwnd Window handle
     * @return true if capture initialized
     */
    bool initializeWindow(void* hwnd);

    /**
     * @brief Initialize capture for primary monitor
     * @return true if successful
     */
    bool initializeMonitor();

    /**
     * @brief Initialize capture for a specific monitor by index
     * @param monitorIndex Monitor index (0 = primary)
     * @return true if successful
     */
    bool initializeMonitor(int monitorIndex);

    /**
     * @brief Start capture with frame callback
     *
     * The callback is called from a background thread for each frame.
     * The texture is only valid during the callback.
     *
     * @param callback Called for each captured frame
     */
    void startCapture(FrameCallback callback);

    /**
     * @brief Stop capture and release resources
     */
    void stopCapture();

    /**
     * @brief Get D3D11 device
     */
    ID3D11Device* getDevice() const;

    /**
     * @brief Get D3D11 device context
     */
    ID3D11DeviceContext* getDeviceContext() const;

    /**
     * @brief Check if capture is active
     */
    bool isCapturing() const;

    /**
     * @brief Get capture statistics
     */
    CaptureStats getStats() const;

    /**
     * @brief Get last error message
     */
    const std::string& getLastError() const;

    /**
     * @brief Get current frame dimensions
     */
    void getFrameSize(int& width, int& height) const;

    /**
     * @brief Set frame pool size (number of buffered frames)
     * Default is 2. Call before startCapture.
     */
    void setFramePoolSize(int size);

    /**
     * @brief Enable/disable cursor capture
     */
    void setCursorEnabled(bool enabled);

    /**
     * @brief Enable/disable capture border highlight
     */
    void setBorderEnabled(bool enabled);

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

/**
 * @brief List available windows for capture
 * @return Vector of (HWND, title) pairs
 */
std::vector<std::pair<void*, std::string>> enumerateWindows();

/**
 * @brief List available monitors for capture
 * @return Vector of (HMONITOR, name) pairs
 */
std::vector<std::pair<void*, std::string>> enumerateMonitors();

} // namespace trading_monitor

