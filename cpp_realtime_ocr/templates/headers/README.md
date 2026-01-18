# Panel Header Templates

Place header templates (cropped screenshots of the panel title bars) in this folder.

Expected filenames (used by `--detect-panels` offline mode):

- `positions_hdr.png`
- `order_hdr.png`
- `quote_hdr.png`

Notes:
- PNG recommended (lossless).
- Crop tightly around the header text + background, but avoid including overlapping UI elements.
- These are matched using grayscale NCC across multiple scales.
