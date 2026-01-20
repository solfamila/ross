#include "video/mf_demuxer.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <sstream>

#include <combaseapi.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavcodec/bsf.h>
#include <libavutil/avutil.h>
#include <libavutil/error.h>
}

namespace {

std::string avErrorToString(int err) {
    char buf[256] = {0};
    av_strerror(err, buf, sizeof(buf));
    return std::string(buf);
}

bool isH264Subtype(const GUID& subtype) {
    if (subtype == MFVideoFormat_H264) {
        return true;
    }
#ifdef MFVideoFormat_H264_ES
    if (subtype == MFVideoFormat_H264_ES) {
        return true;
    }
#endif
    return false;
}

bool isHevcSubtype(const GUID& subtype) {
    if (subtype == MFVideoFormat_HEVC) {
        return true;
    }
#ifdef MFVideoFormat_HEVC_ES
    if (subtype == MFVideoFormat_HEVC_ES) {
        return true;
    }
#endif
    return false;
}

} // namespace

namespace trading_monitor::video {

MFDemuxer::~MFDemuxer() { close(); }

std::string MFDemuxer::hresultToString(HRESULT hr) {
    std::ostringstream oss;
    oss << "HRESULT=0x" << std::hex << static_cast<unsigned long>(hr);
    return oss.str();
}

bool MFDemuxer::open(const std::wstring& pathOrUrl, std::string& err) {
    close();

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

    attrs->SetUINT32(MF_LOW_LATENCY, TRUE);
    attrs->SetUINT32(MF_READWRITE_ENABLE_HARDWARE_TRANSFORMS, TRUE);
    attrs->SetUINT32(MF_SOURCE_READER_ENABLE_VIDEO_PROCESSING, FALSE);

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

    (void)queryFrameSizeAndRate(err);
    if (!querySequenceHeader(err)) {
        close();
        return false;
    }

    if (!setupBitstreamFilter(err)) {
        close();
        return false;
    }

    m_eos = false;
    return true;
}

void MFDemuxer::close() {
    m_reader.Reset();

    if (m_bsf) {
        av_bsf_free(&m_bsf);
    }
    if (m_codecParams) {
        avcodec_parameters_free(&m_codecParams);
    }

    m_pending.clear();
    m_seqHeader.clear();
    m_codec = DemuxCodec::Unknown;
    m_width = 0;
    m_height = 0;
    m_fps = 0.0;
    m_eos = false;

    if (m_mfInited) {
        MFShutdown();
        m_mfInited = false;
    }

    if (m_comInited) {
        CoUninitialize();
        m_comInited = false;
    }
}

bool MFDemuxer::configureOutputType(std::string& err) {
    if (!m_reader) {
        err = "Demuxer not open";
        return false;
    }

    // Select only the first video stream.
    m_reader->SetStreamSelection(MF_SOURCE_READER_ALL_STREAMS, FALSE);
    m_reader->SetStreamSelection(MF_SOURCE_READER_FIRST_VIDEO_STREAM, TRUE);

    Microsoft::WRL::ComPtr<IMFMediaType> chosen;
    for (DWORD i = 0; ; ++i) {
        Microsoft::WRL::ComPtr<IMFMediaType> type;
        HRESULT hr = m_reader->GetNativeMediaType(MF_SOURCE_READER_FIRST_VIDEO_STREAM, i, &type);
        if (FAILED(hr)) break;

        GUID subtype = {};
        if (SUCCEEDED(type->GetGUID(MF_MT_SUBTYPE, &subtype))) {
            if (isH264Subtype(subtype) || isHevcSubtype(subtype)) {
                chosen = type;
                break;
            }
        }
    }

    if (!chosen) {
        err = "No supported compressed video subtype (H264/HEVC) found";
        return false;
    }

    GUID subtype = {};
    if (FAILED(chosen->GetGUID(MF_MT_SUBTYPE, &subtype))) {
        err = "Failed to read chosen subtype";
        return false;
    }

    if (isH264Subtype(subtype)) {
        m_codec = DemuxCodec::H264;
    } else if (isHevcSubtype(subtype)) {
        m_codec = DemuxCodec::HEVC;
    }

    HRESULT hr = m_reader->SetCurrentMediaType(MF_SOURCE_READER_FIRST_VIDEO_STREAM, nullptr, chosen.Get());
    if (FAILED(hr)) {
        err = "SetCurrentMediaType(compressed) failed: " + hresultToString(hr);
        return false;
    }

    return true;
}

bool MFDemuxer::queryFrameSizeAndRate(std::string& err) {
    if (!m_reader) {
        err = "Demuxer not open";
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

bool MFDemuxer::querySequenceHeader(std::string& err) {
    if (!m_reader) {
        err = "Demuxer not open";
        return false;
    }

    Microsoft::WRL::ComPtr<IMFMediaType> type;
    HRESULT hr = m_reader->GetCurrentMediaType(MF_SOURCE_READER_FIRST_VIDEO_STREAM, &type);
    if (FAILED(hr)) {
        err = "GetCurrentMediaType failed: " + hresultToString(hr);
        return false;
    }

    UINT32 blobSize = 0;
    hr = type->GetBlobSize(MF_MT_MPEG_SEQUENCE_HEADER, &blobSize);
    if (FAILED(hr) || blobSize == 0) {
        err = "Missing MF_MT_MPEG_SEQUENCE_HEADER";
        return false;
    }

    m_seqHeader.resize(blobSize);
    hr = type->GetBlob(MF_MT_MPEG_SEQUENCE_HEADER, m_seqHeader.data(), blobSize, &blobSize);
    if (FAILED(hr)) {
        err = "MFGetAttributeBlob(MF_MT_MPEG_SEQUENCE_HEADER) failed: " + hresultToString(hr);
        return false;
    }

    return true;
}

bool MFDemuxer::setupBitstreamFilter(std::string& err) {
    if (m_codec == DemuxCodec::Unknown) {
        err = "Unsupported codec (only H264/HEVC)";
        return false;
    }

    const char* bsfName = (m_codec == DemuxCodec::H264) ? "h264_mp4toannexb" : "hevc_mp4toannexb";
    const AVBitStreamFilter* filter = av_bsf_get_by_name(bsfName);
    if (!filter) {
        err = std::string("FFmpeg bitstream filter not found: ") + bsfName;
        return false;
    }

    if (av_bsf_alloc(filter, &m_bsf) < 0) {
        err = "av_bsf_alloc failed";
        return false;
    }

    m_codecParams = avcodec_parameters_alloc();
    if (!m_codecParams) {
        err = "avcodec_parameters_alloc failed";
        return false;
    }

    m_codecParams->codec_type = AVMEDIA_TYPE_VIDEO;
    m_codecParams->codec_id = (m_codec == DemuxCodec::H264) ? AV_CODEC_ID_H264 : AV_CODEC_ID_HEVC;
    m_codecParams->width = m_width;
    m_codecParams->height = m_height;

    if (!m_seqHeader.empty()) {
        m_codecParams->extradata_size = static_cast<int>(m_seqHeader.size());
        m_codecParams->extradata = static_cast<uint8_t*>(av_malloc(m_codecParams->extradata_size + AV_INPUT_BUFFER_PADDING_SIZE));
        if (!m_codecParams->extradata) {
            err = "av_malloc extradata failed";
            return false;
        }
        std::memcpy(m_codecParams->extradata, m_seqHeader.data(), m_seqHeader.size());
        std::memset(m_codecParams->extradata + m_seqHeader.size(), 0, AV_INPUT_BUFFER_PADDING_SIZE);
    }

    if (avcodec_parameters_copy(m_bsf->par_in, m_codecParams) < 0) {
        err = "avcodec_parameters_copy failed";
        return false;
    }

    if (av_bsf_init(m_bsf) < 0) {
        err = "av_bsf_init failed";
        return false;
    }

    return true;
}

bool MFDemuxer::pushPacketToBsf(const uint8_t* data, size_t size, int64_t pts100ns, bool isKey, std::string& err) {
    if (!m_bsf) {
        DemuxPacket pkt;
        pkt.data.assign(data, data + size);
        pkt.pts100ns = pts100ns;
        pkt.isKey = isKey;
        m_pending.push_back(std::move(pkt));
        return true;
    }

    AVPacket* pkt = av_packet_alloc();
    if (!pkt) {
        err = "av_packet_alloc failed";
        return false;
    }

    if (av_new_packet(pkt, static_cast<int>(size)) < 0) {
        av_packet_free(&pkt);
        err = "av_new_packet failed";
        return false;
    }

    std::memcpy(pkt->data, data, size);
    pkt->pts = pts100ns;
    pkt->dts = pts100ns;
    if (isKey) {
        pkt->flags |= AV_PKT_FLAG_KEY;
    }

    int ret = av_bsf_send_packet(m_bsf, pkt);
    av_packet_free(&pkt);
    if (ret < 0 && ret != AVERROR(EAGAIN)) {
        err = "av_bsf_send_packet failed: " + avErrorToString(ret);
        return false;
    }

    return drainBsf(err);
}

bool MFDemuxer::drainBsf(std::string& err) {
    if (!m_bsf) return true;

    AVPacket* outPkt = av_packet_alloc();
    if (!outPkt) {
        err = "av_packet_alloc failed";
        return false;
    }

    while (true) {
        int ret = av_bsf_receive_packet(m_bsf, outPkt);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
            break;
        }
        if (ret < 0) {
            av_packet_free(&outPkt);
            err = "av_bsf_receive_packet failed: " + avErrorToString(ret);
            return false;
        }

        DemuxPacket pkt;
        pkt.data.assign(outPkt->data, outPkt->data + outPkt->size);
        pkt.pts100ns = outPkt->pts;
        pkt.isKey = (outPkt->flags & AV_PKT_FLAG_KEY) != 0;
        m_pending.push_back(std::move(pkt));

        av_packet_unref(outPkt);
    }

    av_packet_free(&outPkt);
    return true;
}

bool MFDemuxer::flushBsf(std::string& err) {
    if (!m_bsf) return true;

    int ret = av_bsf_send_packet(m_bsf, nullptr);
    if (ret < 0 && ret != AVERROR_EOF) {
        err = "av_bsf_send_packet(flush) failed: " + avErrorToString(ret);
        return false;
    }

    return drainBsf(err);
}

bool MFDemuxer::readPacket(DemuxPacket& out, std::string& err) {
    out = DemuxPacket{};

    if (!m_reader) {
        err = "Demuxer not open";
        return false;
    }

    if (!m_pending.empty()) {
        out = std::move(m_pending.front());
        m_pending.pop_front();
        return true;
    }

    if (m_eos) {
        out.endOfStream = true;
        return true;
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

        if (FAILED(hr)) {
            err = "ReadSample failed: " + hresultToString(hr);
            return false;
        }

        if (flags & MF_SOURCE_READERF_ENDOFSTREAM) {
            m_eos = true;
            if (!flushBsf(err)) {
                return false;
            }
            if (!m_pending.empty()) {
                out = std::move(m_pending.front());
                m_pending.pop_front();
                return true;
            }
            out.endOfStream = true;
            return true;
        }

        if (!sample) {
            continue;
        }

        Microsoft::WRL::ComPtr<IMFMediaBuffer> buffer;
        hr = sample->ConvertToContiguousBuffer(&buffer);
        if (FAILED(hr)) {
            err = "ConvertToContiguousBuffer failed: " + hresultToString(hr);
            return false;
        }

        BYTE* data = nullptr;
        DWORD maxLen = 0;
        DWORD curLen = 0;
        hr = buffer->Lock(&data, &maxLen, &curLen);
        if (FAILED(hr) || !data || curLen == 0) {
            err = "IMFMediaBuffer::Lock failed: " + hresultToString(hr);
            return false;
        }

        UINT32 isKey = 0;
        (void)sample->GetUINT32(MFSampleExtension_CleanPoint, &isKey);

        bool ok = pushPacketToBsf(data, curLen, static_cast<int64_t>(ts100ns), isKey != 0, err);

        buffer->Unlock();

        if (!ok) {
            return false;
        }

        if (!m_pending.empty()) {
            out = std::move(m_pending.front());
            m_pending.pop_front();
            return true;
        }
    }
}

} // namespace trading_monitor::video
