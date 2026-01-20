#include "video/nvdec_decoder.h"

#include <sstream>

namespace trading_monitor::video {

NvdecDecoder::~NvdecDecoder() { close(); }

static std::string cuErrorToString(CUresult res) {
    const char* name = nullptr;
    const char* desc = nullptr;
    cuGetErrorName(res, &name);
    cuGetErrorString(res, &desc);
    std::ostringstream oss;
    oss << (name ? name : "CUDA_ERROR") << ": " << (desc ? desc : "unknown");
    return oss.str();
}

bool NvdecDecoder::open(int width, int height, cudaVideoCodec codec, std::string& err) {
    close();

    CUresult res = cuInit(0);
    if (res != CUDA_SUCCESS) {
        err = "cuInit failed: " + cuErrorToString(res);
        return false;
    }

    res = cuDeviceGet(&m_dev, 0);
    if (res != CUDA_SUCCESS) {
        err = "cuDeviceGet failed: " + cuErrorToString(res);
        return false;
    }

    res = cuDevicePrimaryCtxRetain(&m_ctx, m_dev);
    if (res != CUDA_SUCCESS) {
        err = "cuDevicePrimaryCtxRetain failed: " + cuErrorToString(res);
        return false;
    }

    res = cuCtxSetCurrent(m_ctx);
    if (res != CUDA_SUCCESS) {
        err = "cuCtxSetCurrent failed: " + cuErrorToString(res);
        return false;
    }

    try {
        m_width = width;
        m_height = height;
        m_decoder = std::make_unique<NvDecoder>(m_ctx, width, height, true, codec, &m_mutex, false, true);
    } catch (const std::exception& ex) {
        err = std::string("NvDecoder init failed: ") + ex.what();
        close();
        return false;
    }

    return true;
}

bool NvdecDecoder::decode(const uint8_t* data, int size, int64_t pts100ns, std::vector<DecodedFrame>& frames, std::string& err) {
    frames.clear();
    if (!m_decoder) {
        err = "NvdecDecoder not open";
        return false;
    }

    uint8_t** ppFrame = nullptr;
    int nFrames = 0;
    int64_t* pTimestamps = nullptr;

    try {
        bool ok = m_decoder->Decode(data, size, &ppFrame, &nFrames, 0, &pTimestamps, pts100ns);
        if (!ok) {
            err = "NvDecoder::Decode returned false";
            return false;
        }
    } catch (const std::exception& ex) {
        err = std::string("NvDecoder::Decode exception: ") + ex.what();
        return false;
    }

    for (int i = 0; i < nFrames; ++i) {
        DecodedFrame f;
        f.devPtr = reinterpret_cast<CUdeviceptr>(ppFrame[i]);
        f.pitch = m_decoder->GetDeviceFramePitch();
        f.width = m_decoder->GetWidth();
        f.height = m_decoder->GetHeight();
        f.pts100ns = pTimestamps ? pTimestamps[i] : pts100ns;
        frames.push_back(f);
    }

    return true;
}

void NvdecDecoder::close() {
    m_decoder.reset();
    if (m_ctx) {
        cuDevicePrimaryCtxRelease(m_dev);
        m_ctx = nullptr;
    }
    m_dev = 0;
    m_width = 0;
    m_height = 0;
}

} // namespace trading_monitor::video
