#include "detection/entry_all_detector.h"
#include "detection/glyph_ocr.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <iostream>

namespace trading_monitor::detect {

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
	int stride = (m_cfg.rowStride > 0) ? m_cfg.rowStride : (roiH / std::max(1, m_cfg.maxRows));
	stride = std::max(1, stride);

	for (int r = 0; r < m_cfg.maxRows; ++r) {
		const int rowTop = r * stride;
		const int rowCenter = rowTop + (stride / 2);
		int y0 = rowTop + (stride - bandH) / 2;
		y0 = std::max(0, std::min(y0, roiH - 1));
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
