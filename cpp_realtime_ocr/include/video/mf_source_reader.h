#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <mfapi.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#include <wrl/client.h>

namespace trading_monitor::video {

struct VideoFrameY {
    int width = 0;
    int height = 0;
    int strideY = 0;  // bytes per row (Y plane)
    int64_t pts100ns = 0; // presentation timestamp in 100ns units
    uint64_t frameIndex = 0;

    // Luma plane (size = strideY * height)
    std::vector<uint8_t> y;
};

// Media Foundation source reader decoder for offline MP4 (and other MF-supported sources).
// Output is requested as MFVideoFormat_NV12 for maximum decoder compatibility.
class MFSourceReaderDecoder {
public:
    MFSourceReaderDecoder() = default;
    ~MFSourceReaderDecoder();

    MFSourceReaderDecoder(const MFSourceReaderDecoder&) = delete;
    MFSourceReaderDecoder& operator=(const MFSourceReaderDecoder&) = delete;

    bool open(const std::wstring& pathOrUrl, std::string& err);
    void close();

    // Returns true and fills `out` when a frame is produced.
    // Returns false on EOF or error; `err` is set.
    bool readFrame(VideoFrameY& out, std::string& err);

    int width() const { return m_width; }
    int height() const { return m_height; }
    double fps() const { return m_fps; } // best-effort

private:
    bool configureOutputType(std::string& err);
    bool queryFrameSizeAndRate(std::string& err);

    Microsoft::WRL::ComPtr<IMFSourceReader> m_reader;
    bool m_comInited = false;
    bool m_mfInited = false;
    int m_width = 0;
    int m_height = 0;
    double m_fps = 0.0;
    uint64_t m_frameIndex = 0;
};

} // namespace trading_monitor::video
