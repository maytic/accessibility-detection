"""Run the exported int8 tflite model on one or more images and save the
detections drawn on top, using the same MediaPipe Tasks API (anchor decoding
+ NMS via the model's embedded metadata) as evaluate_tflite.py and as
on-device (Android) inference will use.

Usage:
    python3 predict.py image1.jpg image2.jpg ...
    python3 predict.py path/to/folder

Annotated images are written next to the input as "<name>_pred.jpg", or into
--out-dir if given.
"""
import argparse
import os

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from PIL import Image, ImageDraw, ImageFont

EXPORT_DIR = "exported_model"
TFLITE_MODEL = os.path.join(EXPORT_DIR, "accessibility_detector.tflite")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def collect_image_paths(inputs):
    paths = []
    for p in inputs:
        if os.path.isdir(p):
            for name in sorted(os.listdir(p)):
                if os.path.splitext(name)[1].lower() in IMAGE_EXTS:
                    paths.append(os.path.join(p, name))
        else:
            paths.append(p)
    return paths


def draw_detections(pil_image, detections):
    draw = ImageDraw.Draw(pil_image)
    for detection in detections:
        box = detection.bounding_box
        category = detection.categories[0]
        label = f"{category.category_name} {category.score:.2f}"
        x0, y0 = box.origin_x, box.origin_y
        x1, y1 = x0 + box.width, y0 + box.height
        draw.rectangle([x0, y0, x1, y1], outline="red", width=3)
        text_y = max(0, y0 - 12)
        draw.rectangle([x0, text_y, x0 + 7 * len(label), text_y + 12], fill="red")
        draw.text((x0 + 1, text_y), label, fill="white")
    return pil_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="Image files and/or directories")
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument("--max-results", type=int, default=20)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    options = vision.ObjectDetectorOptions(
        base_options=BaseOptions(model_asset_path=TFLITE_MODEL),
        max_results=args.max_results,
        score_threshold=args.score_threshold,
    )
    detector = vision.ObjectDetector.create_from_options(options)

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)

    for path in collect_image_paths(args.inputs):
        pil_image = Image.open(path).convert("RGB")
        rgb_array = np.asarray(pil_image)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_array)

        result = detector.detect(mp_image)

        print(f"\n{path}: {len(result.detections)} detection(s)")
        for detection in result.detections:
            box = detection.bounding_box
            category = detection.categories[0]
            print(
                f"  {category.category_name:16s} score={category.score:.3f} "
                f"box=[{box.origin_x},{box.origin_y},{box.width},{box.height}]"
            )

        annotated = draw_detections(pil_image.copy(), result.detections)
        base, ext = os.path.splitext(os.path.basename(path))
        out_name = f"{base}_pred.jpg"
        out_path = os.path.join(args.out_dir, out_name) if args.out_dir else os.path.join(
            os.path.dirname(path), out_name
        )
        annotated.save(out_path)
        print(f"  saved: {out_path}")


if __name__ == "__main__":
    main()
