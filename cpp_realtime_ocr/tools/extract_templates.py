#!/usr/bin/env python3
"""
Template Extraction Utility for Lightspeed Positions Table

Extracts reference templates from a screenshot for CUDA template matching.
Templates are saved in a simple binary format: [width:i32][height:i32][pixels:u8[]]

Usage:
    python extract_templates.py <screenshot_path> <output_dir> [--roi x,y,w,h]
"""

import argparse
import struct
import sys
from pathlib import Path

try:
    from PIL import Image
    import numpy as np
except ImportError:
    print("Error: PIL and numpy required. Install with: pip install pillow numpy")
    sys.exit(1)

# Lightspeed positions table configuration (from analysis)
DEFAULT_ROI = {
    'x': 0,
    'y': 0,
    'width': 593,
    'height': 352
}

# Expected table structure
TABLE_CONFIG = {
    'header_height': 16,      # Header row height
    'row_height': 14,         # Data row height
    'row_spacing': 2,         # Gap between rows
    'column_separator': 1,    # Column separator width
    'header_bg': (60, 60, 60),  # Header background color (approx)
    'row_bg_even': (30, 30, 30),  # Even row background
    'row_bg_odd': (40, 40, 40),   # Odd row background
}


def save_template(pixels: np.ndarray, path: Path, name: str):
    """Save template in binary format."""
    height, width = pixels.shape[:2]
    
    # Convert to grayscale if needed
    if len(pixels.shape) == 3:
        gray = np.mean(pixels, axis=2).astype(np.uint8)
    else:
        gray = pixels.astype(np.uint8)
    
    filepath = path / f"{name}.tmpl"
    with open(filepath, 'wb') as f:
        f.write(struct.pack('ii', width, height))
        f.write(gray.tobytes())
    
    print(f"  Saved: {filepath} ({width}x{height})")
    return filepath


def extract_header_template(image: np.ndarray, roi: dict, config: dict) -> np.ndarray:
    """Extract table header template."""
    x, y = roi['x'], roi['y']
    w = roi['width']
    h = config['header_height']
    
    return image[y:y+h, x:x+w]


def extract_row_templates(image: np.ndarray, roi: dict, config: dict, num_rows: int = 3):
    """Extract sample row templates."""
    templates = []
    x = roi['x']
    w = roi['width']
    
    start_y = roi['y'] + config['header_height'] + config['row_spacing']
    row_stride = config['row_height'] + config['row_spacing']
    
    for i in range(num_rows):
        y = start_y + i * row_stride
        h = config['row_height']
        
        if y + h <= roi['y'] + roi['height']:
            row_img = image[y:y+h, x:x+w]
            templates.append((f"row_{i}", row_img))
    
    return templates


def extract_column_headers(image: np.ndarray, roi: dict, config: dict):
    """Extract individual column header templates."""
    # Known column positions for Lightspeed (approximate)
    columns = [
        ('col_symbol', 0, 60),
        ('col_qty', 60, 50),
        ('col_price', 110, 55),
        ('col_value', 165, 60),
        ('col_pnl', 225, 55),
    ]
    
    templates = []
    y = roi['y']
    h = config['header_height']
    
    for name, x_offset, width in columns:
        x = roi['x'] + x_offset
        if x + width <= roi['x'] + roi['width']:
            col_img = image[y:y+h, x:x+width]
            templates.append((name, col_img))
    
    return templates


def main():
    parser = argparse.ArgumentParser(description='Extract templates from Lightspeed screenshot')
    parser.add_argument('screenshot', help='Path to screenshot image')
    parser.add_argument('output_dir', help='Output directory for templates')
    parser.add_argument('--roi', help='ROI as x,y,w,h (e.g., 0,0,593,352)')
    parser.add_argument('--preview', action='store_true', help='Show template previews')
    
    args = parser.parse_args()
    
    # Load image
    img_path = Path(args.screenshot)
    if not img_path.exists():
        print(f"Error: Image not found: {img_path}")
        return 1
    
    print(f"Loading: {img_path}")
    img = Image.open(img_path)
    img_array = np.array(img)
    print(f"  Image size: {img.width}x{img.height}")
    
    # Parse ROI
    if args.roi:
        parts = [int(x) for x in args.roi.split(',')]
        roi = {'x': parts[0], 'y': parts[1], 'width': parts[2], 'height': parts[3]}
    else:
        roi = DEFAULT_ROI.copy()
        roi['width'] = min(roi['width'], img.width)
        roi['height'] = min(roi['height'], img.height)
    
    print(f"  ROI: x={roi['x']}, y={roi['y']}, w={roi['width']}, h={roi['height']}")
    
    # Create output directory
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Extract templates
    print("\nExtracting templates:")
    
    # Header template
    header = extract_header_template(img_array, roi, TABLE_CONFIG)
    save_template(header, out_path, "table_header")
    
    # Row templates
    rows = extract_row_templates(img_array, roi, TABLE_CONFIG)
    for name, row_img in rows:
        save_template(row_img, out_path, name)
    
    # Column headers
    cols = extract_column_headers(img_array, roi, TABLE_CONFIG)
    for name, col_img in cols:
        save_template(col_img, out_path, name)
    
    # Preview if requested
    if args.preview:
        try:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(2, 3, figsize=(12, 6))
            axes = axes.flatten()
            axes[0].imshow(header)
            axes[0].set_title("Header")
            for i, (name, row_img) in enumerate(rows[:2]):
                axes[i+1].imshow(row_img)
                axes[i+1].set_title(name)
            for i, (name, col_img) in enumerate(cols[:3]):
                axes[i+3].imshow(col_img)
                axes[i+3].set_title(name)
            plt.tight_layout()
            plt.savefig(out_path / "preview.png")
            print(f"\nPreview saved: {out_path / 'preview.png'}")
        except ImportError:
            print("Note: matplotlib not available for preview")
    
    print(f"\nTemplates saved to: {out_path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())

