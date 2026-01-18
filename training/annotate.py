#!/usr/bin/env python3
"""
Simple Visual Annotation Tool for YOLO Training
Usage: python annotate.py [image_directory]

Controls:
  - Left click + drag: Draw bounding box
  - Right click: Delete last box
  - 1-9: Select class (shown at top)
  - N / Right Arrow: Next image
  - P / Left Arrow: Previous image  
  - S: Save current annotations
  - Q / ESC: Quit and save all
"""

import cv2
import os
import sys
import glob
from pathlib import Path

# Class definitions - customize these for your use case
CLASSES = [
    "positions_table",    # 0: The main positions/P&L table
    "orders_table",       # 1: Pending orders table
    "watchlist",          # 2: Stock watchlist
    "chart",              # 3: Price chart
    "row_header",         # 4: Table header row
]

# Colors for each class (BGR)
COLORS = [
    (0, 255, 0),    # Green
    (255, 0, 0),    # Blue
    (0, 255, 255),  # Yellow
    (255, 0, 255),  # Magenta
    (0, 165, 255),  # Orange
]

class AnnotationTool:
    def __init__(self, image_dir):
        self.image_dir = Path(image_dir)
        self.images = sorted(glob.glob(str(self.image_dir / "*.png")) + 
                            glob.glob(str(self.image_dir / "*.jpg")))
        if not self.images:
            print(f"No images found in {image_dir}")
            sys.exit(1)
        
        self.current_idx = 0
        self.current_class = 0
        self.drawing = False
        self.start_x, self.start_y = 0, 0
        self.boxes = []  # List of (class_id, x1, y1, x2, y2)
        self.scale = 1.0
        
        # Create labels directory
        self.labels_dir = self.image_dir.parent / "labels"
        self.labels_dir.mkdir(exist_ok=True)
        
        cv2.namedWindow("Annotate", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Annotate", self.mouse_callback)
        
    def mouse_callback(self, event, x, y, flags, param):
        # Scale coordinates back to original image size
        ox, oy = int(x / self.scale), int(y / self.scale)
        
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_x, self.start_y = ox, oy
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.temp_box = (self.start_x, self.start_y, ox, oy)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            x1, y1 = min(self.start_x, ox), min(self.start_y, oy)
            x2, y2 = max(self.start_x, ox), max(self.start_y, oy)
            if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:  # Min size
                self.boxes.append((self.current_class, x1, y1, x2, y2))
                print(f"Added {CLASSES[self.current_class]}: ({x1},{y1}) to ({x2},{y2})")
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.boxes:
                removed = self.boxes.pop()
                print(f"Removed: {CLASSES[removed[0]]}")
    
    def load_annotations(self, image_path):
        """Load existing YOLO annotations for an image."""
        self.boxes = []
        label_path = self.labels_dir / (Path(image_path).stem + ".txt")
        if label_path.exists():
            img = cv2.imread(image_path)
            h, w = img.shape[:2]
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls, xc, yc, bw, bh = int(parts[0]), *map(float, parts[1:])
                        x1 = int((xc - bw/2) * w)
                        y1 = int((yc - bh/2) * h)
                        x2 = int((xc + bw/2) * w)
                        y2 = int((yc + bh/2) * h)
                        self.boxes.append((cls, x1, y1, x2, y2))
            print(f"Loaded {len(self.boxes)} existing annotations")
    
    def save_annotations(self, image_path):
        """Save annotations in YOLO format."""
        img = cv2.imread(image_path)
        h, w = img.shape[:2]
        label_path = self.labels_dir / (Path(image_path).stem + ".txt")
        
        with open(label_path, 'w') as f:
            for cls, x1, y1, x2, y2 in self.boxes:
                xc = ((x1 + x2) / 2) / w
                yc = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                f.write(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
        print(f"Saved {len(self.boxes)} annotations to {label_path}")

    def draw_ui(self, img):
        """Draw boxes and UI overlay."""
        display = img.copy()
        h, w = img.shape[:2]

        # Draw existing boxes
        for cls, x1, y1, x2, y2 in self.boxes:
            color = COLORS[cls % len(COLORS)]
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display, CLASSES[cls], (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw temp box while drawing
        if self.drawing and hasattr(self, 'temp_box'):
            x1, y1, x2, y2 = self.temp_box
            color = COLORS[self.current_class % len(COLORS)]
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 1)

        # Draw class selector at top
        bar_h = 30
        cv2.rectangle(display, (0, 0), (w, bar_h), (40, 40, 40), -1)
        x_offset = 10
        for i, cls_name in enumerate(CLASSES):
            color = COLORS[i % len(COLORS)]
            text = f"{i+1}:{cls_name}"
            if i == self.current_class:
                cv2.putText(display, f"[{text}]", (x_offset, 22),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            else:
                cv2.putText(display, text, (x_offset, 22),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            x_offset += len(text) * 12 + 20

        # Draw status bar at bottom
        status = f"Image {self.current_idx+1}/{len(self.images)} | Boxes: {len(self.boxes)} | N=Next P=Prev S=Save Q=Quit"
        cv2.rectangle(display, (0, h-25), (w, h), (40, 40, 40), -1)
        cv2.putText(display, status, (10, h-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return display

    def run(self):
        """Main loop."""
        print("\n=== YOLO Annotation Tool ===")
        print("Controls: Left-click drag to draw, Right-click to undo")
        print("Keys: 1-5 select class, N/P navigate, S save, Q quit\n")

        while True:
            image_path = self.images[self.current_idx]
            print(f"\nLoading: {os.path.basename(image_path)}")

            img = cv2.imread(image_path)
            if img is None:
                print(f"Failed to load {image_path}")
                self.current_idx = (self.current_idx + 1) % len(self.images)
                continue

            self.load_annotations(image_path)

            # Scale large images to fit screen
            h, w = img.shape[:2]
            max_h, max_w = 900, 1600
            self.scale = min(max_w / w, max_h / h, 1.0)

            while True:
                display = self.draw_ui(img)

                # Resize for display
                if self.scale < 1.0:
                    display = cv2.resize(display, None, fx=self.scale, fy=self.scale)

                cv2.imshow("Annotate", display)
                key = cv2.waitKey(30) & 0xFF

                if key == ord('q') or key == 27:  # Q or ESC
                    self.save_annotations(image_path)
                    cv2.destroyAllWindows()
                    print("\nDone! Annotations saved.")
                    return
                elif key == ord('s'):
                    self.save_annotations(image_path)
                elif key == ord('n') or key == 83:  # N or Right arrow
                    self.save_annotations(image_path)
                    self.current_idx = (self.current_idx + 1) % len(self.images)
                    break
                elif key == ord('p') or key == 81:  # P or Left arrow
                    self.save_annotations(image_path)
                    self.current_idx = (self.current_idx - 1) % len(self.images)
                    break
                elif ord('1') <= key <= ord('9'):
                    cls = key - ord('1')
                    if cls < len(CLASSES):
                        self.current_class = cls
                        print(f"Selected class: {CLASSES[cls]}")

if __name__ == "__main__":
    image_dir = sys.argv[1] if len(sys.argv) > 1 else "images"
    tool = AnnotationTool(image_dir)
    tool.run()

