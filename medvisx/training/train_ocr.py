"""
Standalone OCR Training Runner
Usage:
    python training/train_ocr.py \
        --train_csv data/prescription_train.csv \
        --train_img_dir data/prescription_images/train \
        --val_csv data/prescription_val.csv \
        --val_img_dir data/prescription_images/val
"""
import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.stage1_ocr.train_ocr import train_ocr

parser = argparse.ArgumentParser(description="Train MedVis-X OCR Model")
parser.add_argument("--train_csv", required=True,
                    help="Path to training CSV (columns: filename, label)")
parser.add_argument("--train_img_dir", required=True,
                    help="Path to training images directory")
parser.add_argument("--val_csv", default=None,
                    help="Path to validation CSV (optional)")
parser.add_argument("--val_img_dir", default=None,
                    help="Path to validation images directory (optional)")
parser.add_argument("--save_path", default="models/ocr/crnn_best.pth",
                    help="Where to save the best model weights")
args = parser.parse_args()

print("=" * 50)
print("  MedVis-X OCR Training")
print("=" * 50)
train_ocr(args.train_csv, args.train_img_dir,
          args.val_csv, args.val_img_dir,
          args.save_path)
print("Training complete.")
