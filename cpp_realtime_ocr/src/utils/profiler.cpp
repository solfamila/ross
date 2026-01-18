#include "utils/profiler.h"

#include <Windows.h>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <numeric>

namespace trading_monitor {

static double qpcToMs(int64_t dt) {
    static double freq = []{
        LARGE_INTEGER f{};
        QueryPerformanceFrequency(&f);
        return (double)f.QuadPart;
    }();
    return 1000.0 * (double)dt / freq;
}

static int64_t qpcNow() {
    LARGE_INTEGER t{};
    QueryPerformanceCounter(&t);
    return t.QuadPart;
}

void Profiler::beginFrame(uint64_t frameNumber) {
    if (!m_enabled) return;
    m_frameNumber = frameNumber;
    m_t0 = qpcNow();
    m_last = m_t0;
    FrameProfileRow row{};
    row.frameNumber = frameNumber;
    m_rows.push_back(row);
}

void Profiler::markGray() {
    if (!m_enabled || m_rows.empty()) return;
    int64_t now = qpcNow();
    m_rows.back().grayMs = qpcToMs(now - m_last);
    m_last = now;
}
void Profiler::markFind() {
    if (!m_enabled || m_rows.empty()) return;
    int64_t now = qpcNow();
    m_rows.back().findMs = qpcToMs(now - m_last);
    m_last = now;
}
void Profiler::markTrack() {
    if (!m_enabled || m_rows.empty()) return;
    int64_t now = qpcNow();
    m_rows.back().trackMs = qpcToMs(now - m_last);
    m_last = now;
}
void Profiler::markRoiBuild() {
    if (!m_enabled || m_rows.empty()) return;
    int64_t now = qpcNow();
    m_rows.back().roiBuildMs = qpcToMs(now - m_last);
    m_last = now;
}
void Profiler::markMatch(float score) {
    if (!m_enabled || m_rows.empty()) return;
    int64_t now = qpcNow();
    m_rows.back().matchMs = qpcToMs(now - m_last);
    m_rows.back().triggerScore = score;
    m_last = now;
}
void Profiler::markState(bool armed, bool triggered) {
    if (!m_enabled || m_rows.empty()) return;
    int64_t now = qpcNow();
    m_rows.back().stateMs = qpcToMs(now - m_last);
    m_rows.back().armed = armed;
    m_rows.back().triggered = triggered;
    m_last = now;
}

void Profiler::endFrame() {
    if (!m_enabled || m_rows.empty()) return;
    int64_t now = qpcNow();
    m_rows.back().totalMs = qpcToMs(now - m_t0);
    m_last = now;
}

static double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    double idx = p * (v.size() - 1);
    size_t lo = (size_t)std::floor(idx);
    size_t hi = (size_t)std::ceil(idx);
    double a = v[lo];
    double b = v[hi];
    double t = idx - (double)lo;
    return a + (b - a) * t;
}

void Profiler::flush(const std::string& jsonPath, const std::string& summaryPath) {
    if (!m_enabled) return;

    if (!jsonPath.empty()) {
        std::ofstream out(jsonPath);
        if (out.good()) {
            out << "[\n";
            for (size_t i=0;i<m_rows.size();++i) {
                const auto& r = m_rows[i];
                out << "  {\"frame\":" << r.frameNumber
                    << ",\"total_ms\":" << std::fixed << std::setprecision(3) << r.totalMs
                    << ",\"gray_ms\":" << r.grayMs
                    << ",\"find_ms\":" << r.findMs
                    << ",\"track_ms\":" << r.trackMs
                    << ",\"roi_ms\":" << r.roiBuildMs
                    << ",\"match_ms\":" << r.matchMs
                    << ",\"state_ms\":" << r.stateMs
                    << ",\"armed\":" << (r.armed?"true":"false")
                    << ",\"triggered\":" << (r.triggered?"true":"false")
                    << ",\"score\":" << r.triggerScore
                    << "}" << (i+1<m_rows.size()?",":"") << "\n";
            }
            out << "]\n";
        }
    }

    if (!summaryPath.empty()) {
        std::vector<double> totals; totals.reserve(m_rows.size());
        std::vector<double> match; match.reserve(m_rows.size());
        for (const auto& r : m_rows) { totals.push_back(r.totalMs); match.push_back(r.matchMs); }
        std::ofstream out(summaryPath);
        if (out.good()) {
            out << "Frames: " << m_rows.size() << "\n";
            out << "Total ms p50=" << percentile(totals,0.5) << " p90=" << percentile(totals,0.9) << " p99=" << percentile(totals,0.99) << "\n";
            out << "Match ms p50=" << percentile(match,0.5) << " p90=" << percentile(match,0.9) << " p99=" << percentile(match,0.99) << "\n";
        }
    }
}

} // namespace trading_monitor
