#pragma once
/**
 * @file timer.h
 * @brief High-resolution timing utilities for performance measurement
 */

#include <Windows.h>
#include <cuda_runtime.h>
#include <chrono>

namespace trading_monitor {

/**
 * @brief High-resolution CPU timer using QueryPerformanceCounter
 */
class HighResTimer {
public:
    HighResTimer() {
        QueryPerformanceFrequency(&m_frequency);
    }
    
    void start() {
        QueryPerformanceCounter(&m_start);
    }
    
    void stop() {
        QueryPerformanceCounter(&m_stop);
    }
    
    double elapsedMs() const {
        return (m_stop.QuadPart - m_start.QuadPart) * 1000.0 / m_frequency.QuadPart;
    }
    
    double elapsedUs() const {
        return (m_stop.QuadPart - m_start.QuadPart) * 1000000.0 / m_frequency.QuadPart;
    }
    
    double currentElapsedMs() const {
        LARGE_INTEGER now;
        QueryPerformanceCounter(&now);
        return (now.QuadPart - m_start.QuadPart) * 1000.0 / m_frequency.QuadPart;
    }
    
private:
    LARGE_INTEGER m_frequency;
    LARGE_INTEGER m_start;
    LARGE_INTEGER m_stop;
};

/**
 * @brief CUDA event-based timer for GPU operations
 */
class CudaTimer {
public:
    CudaTimer() {
        cudaEventCreate(&m_start);
        cudaEventCreate(&m_stop);
    }
    
    ~CudaTimer() {
        cudaEventDestroy(m_start);
        cudaEventDestroy(m_stop);
    }
    
    // Non-copyable
    CudaTimer(const CudaTimer&) = delete;
    CudaTimer& operator=(const CudaTimer&) = delete;
    
    void start(cudaStream_t stream = 0) {
        cudaEventRecord(m_start, stream);
    }
    
    void stop(cudaStream_t stream = 0) {
        cudaEventRecord(m_stop, stream);
    }
    
    /**
     * @brief Wait for stop event and get elapsed time
     * @return Elapsed time in milliseconds
     */
    float elapsedMs() {
        cudaEventSynchronize(m_stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, m_start, m_stop);
        return ms;
    }
    
private:
    cudaEvent_t m_start;
    cudaEvent_t m_stop;
};

/**
 * @brief RAII timer for automatic scope timing
 */
class ScopedTimer {
public:
    explicit ScopedTimer(double& output) : m_output(output) {
        m_timer.start();
    }
    
    ~ScopedTimer() {
        m_timer.stop();
        m_output = m_timer.elapsedMs();
    }
    
private:
    HighResTimer m_timer;
    double& m_output;
};

/**
 * @brief Macro for easy scope timing
 */
#define SCOPE_TIMER(var) ScopedTimer _scopedTimer##__LINE__(var)

} // namespace trading_monitor

