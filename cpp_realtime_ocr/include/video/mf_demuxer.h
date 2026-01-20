#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <string>
#include <vector>

#include <mfapi.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#include <wrl/client.h>

struct AVBSFContext;
struct AVCodecParameters;

namespace trading_monitor::video {

enum class DemuxCodec {
    Unknown,
    H264,
    HEVC
};

struct DemuxPacket {
    std::vector<uint8_t> data;
    int64_t pts100ns = 0;
    bool isKey = false;
    bool endOfStream = false;
};

class MFDemuxer {
public:
    MFDemuxer() = default;
    ~MFDemuxer();

    MFDemuxer(const MFDemuxer&) = delete;
    MFDemuxer& operator=(const MFDemuxer&) = delete;

    bool open(const std::wstring& pathOrUrl, std::string& err);
    void close();

    // Returns true and fills `out` when a packet is produced.
    // Returns false on error; `err` is set.
    bool readPacket(DemuxPacket& out, std::string& err);

    int width() const { return m_width; }
    int height() const { return m_height; }
    double fps() const { return m_fps; }
    DemuxCodec codec() const { return m_codec; }

private:
    bool configureOutputType(std::string& err);
    bool queryFrameSizeAndRate(std::string& err);
    bool querySequenceHeader(std::string& err);
    bool setupBitstreamFilter(std::string& err);
    bool pushPacketToBsf(const uint8_t* data, size_t size, int64_t pts100ns, bool isKey, std::string& err);
    bool drainBsf(std::string& err);
    bool flushBsf(std::string& err);

    static std::string hresultToString(HRESULT hr);

    Microsoft::WRL::ComPtr<IMFSourceReader> m_reader;
    bool m_comInited = false;
    bool m_mfInited = false;
    bool m_eos = false;

    int m_width = 0;
    int m_height = 0;
    double m_fps = 0.0;
    DemuxCodec m_codec = DemuxCodec::Unknown;
    std::vector<uint8_t> m_seqHeader;

    AVCodecParameters* m_codecParams = nullptr;
    AVBSFContext* m_bsf = nullptr;

    std::deque<DemuxPacket> m_pending;
};

} // namespace trading_monitor::video
