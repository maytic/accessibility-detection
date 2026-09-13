
import json
import os

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

EXPORT_DIR = "exported_model"
TFLITE_MODEL = os.path.join(EXPORT_DIR, "accessibility_detector.tflite")
VAL_DIR = "dataset/mm/val"
ANN_FILE = os.path.join(VAL_DIR, "labels.json")
IMAGES_DIR = os.path.join(VAL_DIR, "images")
MAX_DETS = 100

coco_gt = COCO(ANN_FILE)
name_to_cat_id = {c["name"]: c["id"] for c in coco_gt.dataset["categories"]}

options = vision.ObjectDetectorOptions(
    base_options=BaseOptions(model_asset_path=TFLITE_MODEL),
    max_results=MAX_DETS,
    score_threshold=0.0,
)
detector = vision.ObjectDetector.create_from_options(options)

results = []
image_ids = coco_gt.getImgIds()
for i, image_id in enumerate(image_ids):
    file_name = coco_gt.loadImgs(image_id)[0]["file_name"]
    # Some COCO images are grayscale/CMYK; create_from_file() keeps the
    # source channel count as-is, but the model expects 3-channel RGB input.
    rgb_array = np.asarray(Image.open(os.path.join(IMAGES_DIR, file_name)).convert("RGB"))
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_array)
    detection_result = detector.detect(mp_image)

    for detection in detection_result.detections:
        category = detection.categories[0]
        cat_id = name_to_cat_id.get(category.category_name)
        if cat_id is None:
            continue
        box = detection.bounding_box
        results.append({
            "image_id": image_id,
            "category_id": cat_id,
            "bbox": [box.origin_x, box.origin_y, box.width, box.height],
            "score": category.score,
        })

    if (i + 1) % 20 == 0 or (i + 1) == len(image_ids):
        print(f"\r{i + 1}/{len(image_ids)} images", end="", flush=True)
print()

if not results:
    raise SystemExit("No detections at all -- check the model/labels before scoring.")

with open("tflite_detections.json", "w") as f:
    json.dump(results, f)

coco_dt = coco_gt.loadRes(results)
coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
coco_eval.params.maxDets = [1, 10, MAX_DETS]
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

metric_names = [
    "AP", "AP50", "AP75", "APs", "APm", "APl",
    "ARmax1", "ARmax10", "ARmax100", "ARs", "ARm", "ARl",
]
print("int8 tflite model:", dict(zip(metric_names, coco_eval.stats)))


# DONE (t=0.36s).
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.192
#  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.366
#  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.192
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.050
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.223
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.335
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.183
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.267
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.280
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.084
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.317
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.452