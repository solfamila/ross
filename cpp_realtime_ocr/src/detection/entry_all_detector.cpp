#include "detection/entry_all_detector.h"
#include "detection/glyph_ocr.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <iostream>

namespace trading_monitor::detect {

static bool isHeaderWord(const std::string& s) {
    static const char* kBad[] = {
        "SYMBOL","POSITION","SIZE","COST","BASIS","OPEN","PNL","REALIZED","YOUR"
    };
    for (auto* w : kBad) {
        if (s == w) return true;
    }
    return false;
}

static std::vector<int> detectRowCentersByProjection(const std::vector<uint8_t>& frameGray, int frameW, int frameH,
                                                     int roiX, int roiY, int roiW, int roiH,
                                                     int x0, int x1, int yBase, int yH,
                                                     int maxRows) {
    std::vector<int> centers;
    if (maxRows <= 0) return centers;

    x0 = std::max(0, std::min(x0, roiW-1));
    x1 = std::max(x0+1, std::min(x1, roiW));
    const int bw = x1 - x0;

    const int y0 = std::max(0, std::min(yBase, roiH-1));
    const int y1 = std::max(y0+1, std::min(yBase + yH, roiH));
    const int bh = y1 - y0;

    // Compute per-row "ink" score: count bright pixels in symbol column
    std::vector<float> score(bh, 0.0f);
    for (int yy = 0; yy < bh; ++yy) {
        const int fy = roiY + y0 + yy;
        if (fy < 0 || fy >= frameH) continue;
        const uint8_t* row = frameGray.data() + static_cast<size_t>(fy) * static_cast<size_t>(frameW) + static_cast<size_t>(roiX + x0);
        int cnt = 0;
        for (int xx = 0; xx < bw; ++xx) {
            const uint8_t v = row[xx];
            if (v >= 200) cnt++;
        }
        score[yy] = static_cast<float>(cnt);
    }

    // Smooth with a small moving average
    std::vector<float> sm(bh, 0.0f);
    const int win = 5;
    for (int yy = 0; yy < bh; ++yy) {
        float s = 0.0f;
        int c = 0;
        for (int k = -win; k <= win; ++k) {
            int yk = yy + k;
            if (yk < 0 || yk >= bh) continue;
            s += score[yk];
            c++;
        }
        sm[yy] = (c > 0) ? (s / c) : 0.0f;
    }

    // Find local maxima above a threshold
    float maxv = 0.0f;
    for (float v : sm) maxv = std::max(maxv, v);
    if (maxv <= 0.01f) return centers;
    const float thr = std::max(2.0f, maxv * 0.25f);

    struct Peak { int y; float v; };
    std::vector<Peak> peaks;
    for (int yy = 2; yy < bh - 2; ++yy) {
        float v = sm[yy];
        if (v < thr) continue;
        if (v >= sm[yy-1] && v >= sm[yy+1] && v >= sm[yy-2] && v >= sm[yy+2]) {
            peaks.push_back({yy, v});
        }
    }
    std::sort(peaks.begin(), peaks.end(), [](const Peak& a, const Peak& b){ return a.v > b.v; });

    const int minDist = 14;
    for (const auto& p : peaks) {
        bool ok = true;
        for (int c : centers) {
            if (std::abs(c - p.y) < minDist) { ok = false; break; }
        }
        if (!ok) continue;
        centers.push_back(p.y);
        if ((int)centers.size() >= maxRows) break;
    }
    std::sort(centers.begin(), centers.end());
    // Convert to ROI-relative Y centers
    for (int& c : centers) c = y0 + c;
    return centers;
}


void EntryAllDetector::reset() {
	m_rows.clear();
	m_rows.resize(static_cast<size_t>(m_cfg.maxRows));
	m_debugRemaining = m_cfg.debug ? m_cfg.debugMax : 0;
}

bool EntryAllDetector::isTickerLike(const std::string& s) {
	if (s.size() < 2 || s.size() > 6) return false;
	for (char c : s) {
		if (c < 'A' || c > 'Z') return false;
	}
	return true;
}

std::vector<EntryAllEvent> EntryAllDetector::update(uint64_t frameIdx, double frameTimeMs,
													const std::vector<uint8_t>& frameGray, int frameW, int frameH,
													int roiX, int roiY, int roiW, int roiH) {
	std::vector<EntryAllEvent> out;
	if (frameGray.empty() || frameW <= 0 || frameH <= 0) return out;
	if (roiW <= 0 || roiH <= 0) return out;
	if (m_rows.empty()) reset();

	const int x0 = std::max(0, std::min(m_cfg.symbolX0, roiW - 1));
	const int x1 = std::max(x0 + 1, std::min(m_cfg.symbolX1, roiW));
	const int bandW = x1 - x0;

	const int bandH = std::max(8, m_cfg.bandH);

	// Optional body region inside ROI
	int bodyY = std::max(0, m_cfg.bodyOffsetY);
	int bodyH = (m_cfg.bodyH > 0) ? m_cfg.bodyH : roiH;
	if (bodyY >= roiH) bodyY = 0;
	if (bodyY + bodyH > roiH) bodyH = roiH - bodyY;

	// Determine row centers
	std::vector<int> rowCenters;
	if (m_cfg.rowStride == 0) {
		rowCenters = detectRowCentersByProjection(frameGray, frameW, frameH, roiX, roiY, roiW, roiH,
										  x0, x1, bodyY, bodyH, m_cfg.maxRows);
		// Fallback: equally spaced within body
		if ((int)rowCenters.size() < m_cfg.maxRows) {
			rowCenters.clear();
			const int stride = std::max(1, bodyH / std::max(1, m_cfg.maxRows));
			for (int r = 0; r < m_cfg.maxRows; ++r) rowCenters.push_back(bodyY + r * stride + stride / 2);
		}
	} else {
		const int stride = std::max(1, m_cfg.rowStride);
		for (int r = 0; r < m_cfg.maxRows; ++r) rowCenters.push_back(bodyY + r * stride + stride / 2);
	}
	for (int r = 0; r < m_cfg.maxRows; ++r) {
		const int rowCenter = rowCenters[static_cast<size_t>(r)];
		int y0 = rowCenter - bandH / 2;
		y0 = std::max(0, std::min(y0, roiH - 1));
		// Keep bands inside body region when configured
		if (m_cfg.bodyH > 0) {
			if (y0 < bodyY) y0 = bodyY;
			if (y0 + bandH > bodyY + bodyH) y0 = std::max(bodyY, bodyY + bodyH - bandH);
		}
		int y1 = std::min(roiH, y0 + bandH);
		int h = y1 - y0;
		if (h < 8) continue;

		const int fx0 = roiX + x0;
		const int fy0 = roiY + y0;
		if (fx0 < 0 || fy0 < 0) continue;
		if (fx0 + bandW > frameW) continue;
		if (fy0 + h > frameH) continue;

		std::vector<uint8_t> band(static_cast<size_t>(bandW) * static_cast<size_t>(h));
		for (int yy = 0; yy < h; ++yy) {
			const uint8_t* src = frameGray.data() + static_cast<size_t>(fy0 + yy) * static_cast<size_t>(frameW) + static_cast<size_t>(fx0);
			uint8_t* dst = band.data() + static_cast<size_t>(yy) * static_cast<size_t>(bandW);
			std::memcpy(dst, src, static_cast<size_t>(bandW));
		}

		RowState& st = m_rows[static_cast<size_t>(r)];

		bool changed = true;
		if (!st.prevBand.empty() && st.prevW == bandW && st.prevH == h) {
			double sum = 0.0;
			const size_t n = band.size();
			for (size_t i = 0; i < n; ++i) sum += std::abs((int)band[i] - (int)st.prevBand[i]);
			const double mean = sum / (double)n;
			changed = (mean >= m_cfg.diffThreshold);
		}
		st.prevBand = band;
		st.prevW = bandW;
		st.prevH = h;

		if (!changed) continue;

		GlyphOCRResult bestOcr;
		bestOcr.score = -1.0f;
		for (int rowH : {24, 28, 32, 36}) {
			for (int off : {-8, -4, 0, 4, 8}) {
				int cy0 = rowCenter + off - (rowH / 2);
				cy0 = std::max(0, std::min(cy0, roiH - 1));
				int cy1 = std::min(roiH, cy0 + rowH);
				int ch = cy1 - cy0;
				if (ch < 8) continue;
				if (fy0 + (cy0 - y0) < 0 || (fy0 + (cy0 - y0) + ch) > frameH) continue;

				std::vector<uint8_t> cand(static_cast<size_t>(bandW) * static_cast<size_t>(ch));
				for (int yy = 0; yy < ch; ++yy) {
					const uint8_t* src = frameGray.data() + static_cast<size_t>(fy0 + (cy0 - y0) + yy) * static_cast<size_t>(frameW) + static_cast<size_t>(fx0);
					uint8_t* dst = cand.data() + static_cast<size_t>(yy) * static_cast<size_t>(bandW);
					std::memcpy(dst, src, static_cast<size_t>(bandW));
				}

				auto ocr = ocrTickerFromRowGray(cand, bandW, ch);
				if (ocr.score > bestOcr.score) {
					bestOcr = ocr;
				}
			}
		}

		if (m_cfg.debug && m_debugRemaining > 0) {
			std::cout << "[entry-all-debug] row=" << r
					  << " text=" << bestOcr.text
					  << " score=" << bestOcr.score
					  << " bandW=" << bandW << " bandH=" << h
					  << " roiY=" << roiY << " y0=" << y0
					  << " rowCenter=" << rowCenter << "\n";
			--m_debugRemaining;
		}

		if (isHeaderWord(bestOcr.text)) {
			continue;
		}
		if (!isTickerLike(bestOcr.text) || bestOcr.score < m_cfg.ocrThreshold) {
			continue;
		}

		if (!st.active) {
			st.active = true;
			st.symbol = bestOcr.text;

			EntryAllEvent ev;
			ev.frame = frameIdx;
			ev.t_ms = frameTimeMs;
			ev.row = r;
			ev.symbol = bestOcr.text;
			ev.ocr_score = bestOcr.score;
			out.push_back(ev);
		}
	}

	return out;
}

} // namespace trading_monitor::detect
