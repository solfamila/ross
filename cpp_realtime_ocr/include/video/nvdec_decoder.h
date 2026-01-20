#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <cuda.h>
#include "NvDecoder/NvDecoder.h"

namespace trading_monitor::video {

struct DecodedFrame {
    CUdeviceptr devPtr = 0;
    int pitch = 0;
    int width = 0;
    int height = 0;
    int64_t pts100ns = 0;
};

class NvdecDecoder {
public:
    NvdecDecoder() = default;
    ~NvdecDecoder();

    NvdecDecoder(const NvdecDecoder&) = delete;
    NvdecDecoder& operator=(const NvdecDecoder&) = delete;

    bool open(int width, int height, cudaVideoCodec codec, std::string& err);
    bool decode(const uint8_t* data, int size, int64_t pts100ns, std::vector<DecodedFrame>& frames, std::string& err);
    void close();

    CUcontext context() const { return m_ctx; }
    int width() const { return m_width; }
    int height() const { return m_height; }

private:
    CUcontext m_ctx = nullptr;
    CUdevice m_dev = 0;
    std::unique_ptr<NvDecoder> m_decoder;
    std::mutex m_mutex;
    int m_width = 0;
    int m_height = 0;
};

} // namespace trading_monitor::video
