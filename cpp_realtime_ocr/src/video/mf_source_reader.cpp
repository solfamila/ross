#include "video/mf_source_reader.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <sstream>

#include <combaseapi.h>

namespace {

static std::string hresultToString(HRESULT hr) {
    std::ostringstream oss;
    oss << "HRESULT=0x" << std::hex << static_cast<unsigned long>(hr);
    return oss.str();
}

} // namespace

namespace trading_monitor::video {

MFSourceReaderDecoder::~MFSourceReaderDecoder() { close(); }

bool MFSourceReaderDecoder::open(const std::wstring& pathOrUrl, std::string& err) {
    close();

    // COM init (best-effort). If the process has already initialized COM with
    // a different apartment model, tolerate RPC_E_CHANGED_MODE and do not
    // call CoUninitialize() later.
    HRESULT hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    if (SUCCEEDED(hr)) {
        m_comInited = true;
    } else if (hr == RPC_E_CHANGED_MODE) {
        m_comInited = false;
    } else {
        err = "CoInitializeEx failed: " + hresultToString(hr);
        return false;
    }

    hr = MFStartup(MF_VERSION);
    if (FAILED(hr)) {
        err = "MFStartup failed: " + hresultToString(hr);
        if (m_comInited) {
            CoUninitialize();
            m_comInited = false;
        }
        return false;
    }
    m_mfInited = true;

    Microsoft::WRL::ComPtr<IMFAttributes> attrs;
    hr = MFCreateAttributes(&attrs, 4);
    if (FAILED(hr)) {
        err = "MFCreateAttributes failed: " + hresultToString(hr);
        close();
        return false;
    }

    // Low-latency hint (best-effort)
    attrs->SetUINT32(MF_LOW_LATENCY, TRUE);
    // Hardware transforms if available
    attrs->SetUINT32(MF_READWRITE_ENABLE_HARDWARE_TRANSFORMS, TRUE);
    // Avoid internal format converters if possible (still best-effort)
    attrs->SetUINT32(MF_SOURCE_READER_ENABLE_VIDEO_PROCESSING, TRUE);

    hr = MFCreateSourceReaderFromURL(pathOrUrl.c_str(), attrs.Get(), &m_reader);
    if (FAILED(hr)) {
        err = "MFCreateSourceReaderFromURL failed: " + hresultToString(hr);
        close();
        return false;
    }

    if (!configureOutputType(err)) {
        close();
        return false;
    }

    // Best-effort metadata
    (void)queryFrameSizeAndRate(err);

    m_frameIndex = 0;
    return true;
}

void MFSourceReaderDecoder::close() {
    // Release reader first.
    m_reader.Reset();
    m_width = 0;
    m_height = 0;
    m_fps = 0.0;
    m_frameIndex = 0;

    if (m_mfInited) {
        MFShutdown();
        m_mfInited = false;
    }

    if (m_comInited) {
        CoUninitialize();
        m_comInited = false;
    }
}

bool MFSourceReaderDecoder::configureOutputType(std::string& err) {
    if (!m_reader) {
        err = "Decoder not open";
        return false;
    }

    // Force output to NV12 for maximum decoder compatibility.
    Microsoft::WRL::ComPtr<IMFMediaType> type;
    HRESULT hr = MFCreateMediaType(&type);
    if (FAILED(hr)) {
        err = "MFCreateMediaType failed: " + hresultToString(hr);
        return false;
    }

    hr = type->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video);
    if (FAILED(hr)) {
        err = "SetGUID(MF_MT_MAJOR_TYPE) failed: " + hresultToString(hr);
        return false;
    }

    hr = type->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_NV12);
    if (FAILED(hr)) {
        err = "SetGUID(MF_MT_SUBTYPE) failed: " + hresultToString(hr);
        return false;
    }

    hr = m_reader->SetCurrentMediaType(MF_SOURCE_READER_FIRST_VIDEO_STREAM, nullptr, type.Get());
    if (FAILED(hr)) {
        err = "SetCurrentMediaType(NV12) failed: " + hresultToString(hr);
        return false;
    }

    // Select only the first video stream.
    m_reader->SetStreamSelection(MF_SOURCE_READER_ALL_STREAMS, FALSE);
    m_reader->SetStreamSelection(MF_SOURCE_READER_FIRST_VIDEO_STREAM, TRUE);

    return true;
}

bool MFSourceReaderDecoder::queryFrameSizeAndRate(std::string& err) {
    if (!m_reader) {
        err = "Decoder not open";
        return false;
    }

    Microsoft::WRL::ComPtr<IMFMediaType> type;
    HRESULT hr = m_reader->GetCurrentMediaType(MF_SOURCE_READER_FIRST_VIDEO_STREAM, &type);
    if (FAILED(hr)) {
        err = "GetCurrentMediaType failed: " + hresultToString(hr);
        return false;
    }

    UINT32 w = 0, h = 0;
    hr = MFGetAttributeSize(type.Get(), MF_MT_FRAME_SIZE, &w, &h);
    if (SUCCEEDED(hr)) {
        m_width = static_cast<int>(w);
        m_height = static_cast<int>(h);
    }

    UINT32 num = 0, den = 0;
    hr = MFGetAttributeRatio(type.Get(), MF_MT_FRAME_RATE, &num, &den);
    if (SUCCEEDED(hr) && den != 0) {
        m_fps = static_cast<double>(num) / static_cast<double>(den);
    }

    return true;
}

bool MFSourceReaderDecoder::readFrame(VideoFrameY& out, std::string& err) {
    if (!m_reader) {
        err = "Decoder not open";
        return false;
    }

    while (true) {
        DWORD streamIndex = 0;
        DWORD flags = 0;
        LONGLONG ts100ns = 0;

        Microsoft::WRL::ComPtr<IMFSample> sample;
        HRESULT hr = m_reader->ReadSample(
            MF_SOURCE_READER_FIRST_VIDEO_STREAM,
            0,
            &streamIndex,
            &flags,
            &ts100ns,
            &sample);

        static int mfPrint = 0;
        if (mfPrint++ < 5 || (flags != 0)) {
            std::cerr << "[MF] flags=0x" << std::hex << flags
                      << " ts100ns=" << std::dec << ts100ns
                      << " sample=" << (sample ? "yes" : "no")
                      << "\n";
        }

        if (FAILED(hr)) {
            err = "ReadSample failed: " + hresultToString(hr);
            return false;
        }

        if (flags & MF_SOURCE_READERF_ENDOFSTREAM) {
            err = "EOF";
            return false;
        }

        if (flags & MF_SOURCE_READERF_STREAMTICK) {
            // Not a decoded frame; caller should continue.
            err = "STREAMTICK";
            return false;
        }

        if (flags & MF_SOURCE_READERF_CURRENTMEDIATYPECHANGED) {
            std::string qerr;
            (void)queryFrameSizeAndRate(qerr);
        }

        if (!sample) {
            err = "NOSAMPLE";
            return false;
        }

        Microsoft::WRL::ComPtr<IMFMediaBuffer> buf;
        hr = sample->ConvertToContiguousBuffer(&buf);
        if (FAILED(hr)) {
            err = "ConvertToContiguousBuffer failed: " + hresultToString(hr);
            return false;
        }

        Microsoft::WRL::ComPtr<IMF2DBuffer> buf2d;
        (void)buf.As(&buf2d);

        BYTE* data = nullptr;
        LONG stride = 0;
        DWORD maxLen = 0;
        DWORD curLen = 0;

        bool locked2d = false;
        if (buf2d) {
            hr = buf2d->Lock2D(&data, &stride);
            if (FAILED(hr)) {
                err = "Lock2D failed: " + hresultToString(hr);
                return false;
            }
            locked2d = true;
            (void)buf->GetCurrentLength(&curLen);
            (void)buf->GetMaxLength(&maxLen);
        } else {
            hr = buf->Lock(&data, &maxLen, &curLen);
            if (FAILED(hr)) {
                err = "Lock failed: " + hresultToString(hr);
                return false;
            }
            stride = (m_width > 0) ? (m_width * 4) : 0;
        }

        // Best-effort size query if unknown.
        if (m_width <= 0 || m_height <= 0) {
            std::string qerr;
            (void)queryFrameSizeAndRate(qerr);
        }

        if (m_width <= 0 || m_height <= 0 || stride == 0) {
            if (locked2d) buf2d->Unlock2D(); else buf->Unlock();
            err = "Unknown frame size/stride";
            return false;
        }

        const int absStride = static_cast<int>(std::abs(stride));

        out.width = m_width;
        out.height = m_height;
        out.strideY = absStride;
        out.pts100ns = static_cast<int64_t>(ts100ns);
        out.frameIndex = m_frameIndex++;
        out.devPtr = nullptr;
        out.devPitch = 0;
        out.onGpu = false;

        out.y.resize(static_cast<size_t>(out.strideY) * static_cast<size_t>(out.height));

        // Copy Y plane row-by-row to handle stride safely.
        const uint8_t* srcBase = reinterpret_cast<const uint8_t*>(data);
        if (stride > 0) {
            for (int y = 0; y < out.height; ++y) {
                const uint8_t* srcRow = srcBase + static_cast<size_t>(y) * static_cast<size_t>(absStride);
                uint8_t* dstRow = out.y.data() + static_cast<size_t>(y) * static_cast<size_t>(out.strideY);
                std::memcpy(dstRow, srcRow, static_cast<size_t>(out.strideY));
            }
        } else {
            // Bottom-up
            for (int y = 0; y < out.height; ++y) {
                const uint8_t* srcRow = srcBase + static_cast<size_t>(out.height - 1 - y) * static_cast<size_t>(absStride);
                uint8_t* dstRow = out.y.data() + static_cast<size_t>(y) * static_cast<size_t>(out.strideY);
                std::memcpy(dstRow, srcRow, static_cast<size_t>(out.strideY));
            }
        }

        if (locked2d) {
            buf2d->Unlock2D();
        } else {
            buf->Unlock();
        }

        return true;
    }
}

} // namespace trading_monitor::video
