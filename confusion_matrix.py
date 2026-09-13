
import csv
import time
from collections import Counter, defaultdict

import numpy as np
import tensorflow as tf

from mediapipe_model_maker import object_detector

TRAINED_SPEC = object_detector.SupportedModels.MOBILENET_V2_I320
EXPORT_DIR = "exported_model"
IOU_MATCH_THRESHOLD = 0.5
CONFIDENCE_THRESHOLD = 0.3
NO_DETECTION = "(no detection)"


def iou_matrix(boxes_a, boxes_b):
    y1a, x1a, y2a, x2a = [boxes_a[:, i:i + 1] for i in range(4)]
    y1b, x1b, y2b, x2b = [boxes_b[None, :, i] for i in range(4)]
    inter_y1 = np.maximum(y1a, y1b)
    inter_x1 = np.maximum(x1a, x1b)
    inter_y2 = np.minimum(y2a, y2b)
    inter_x2 = np.minimum(x2a, x2b)
    inter_area = np.maximum(0, inter_y2 - inter_y1) * np.maximum(0, inter_x2 - inter_x1)
    area_a = (y2a - y1a) * (x2a - x1a)
    area_b = (y2b - y1b) * (x2b - x1b)
    union = area_a + area_b - inter_area
    return np.where(union > 0, inter_area / union, 0.0)


validation_data = object_detector.Dataset.from_coco_folder(
    "dataset/mm/val",
    cache_dir="dataset/mm/val_cache",
)
label_names = validation_data.label_names  # index -> name, index 0 is "background"

spec = object_detector.SupportedModels.get(TRAINED_SPEC)
hparams = object_detector.HParams(export_dir=EXPORT_DIR)
model_options = object_detector.ModelOptions()
model = object_detector.ObjectDetector(
    model_spec=spec,
    label_names=label_names,
    hparams=hparams,
    model_options=model_options,
)
model.restore_float_ckpt()

dataset = validation_data.gen_tf_dataset(
    batch_size=1, is_training=False, preprocess=model._preprocessor
)
num_examples = validation_data.size


@tf.function
def run_inference(x, anchor_boxes, image_shape):
    return model._model(
        x,
        anchor_boxes=anchor_boxes,
        image_shape=image_shape,
        training=False,
    )


matrix = defaultdict(Counter)  # matrix[true_class_name][predicted_class_name_or_NO_DETECTION]

start = time.time()
for i, (x, y) in enumerate(dataset):
    y_pred = run_inference(x, y["anchor_boxes"], y["image_info"][:, 1, :])

    if (i + 1) % 20 == 0 or (i + 1) == num_examples:
        elapsed = time.time() - start
        print(f"\r{i + 1}/{num_examples} images ({elapsed:.1f}s elapsed)", end="", flush=True)

    gt_classes = y["groundtruths"]["classes"][0].numpy()
    gt_boxes = y["groundtruths"]["boxes"][0].numpy()
    valid_gt = gt_classes != -1
    gt_classes = gt_classes[valid_gt]
    gt_boxes = gt_boxes[valid_gt]
    if gt_classes.size == 0:
        continue

    # groundtruths['boxes'] is in original-image pixel space, but the model's
    # predicted boxes are decoded/clipped into the resized model-input pixel
    # space (image_info row 1). Rescale gt boxes into that same space before
    # computing IoU, using the scale/offset the preprocessor already computed
    # (image_info rows 2 and 3), matching official/vision's resize_and_crop_boxes.
    image_info = y["image_info"][0].numpy()
    scale = image_info[2]
    offset = image_info[3]
    gt_boxes = gt_boxes * np.tile(scale, 2) - np.tile(offset, 2)

    num_dets = int(y_pred["num_detections"][0].numpy())
    pred_boxes = y_pred["detection_boxes"][0].numpy()[:num_dets]
    pred_classes = y_pred["detection_classes"][0].numpy()[:num_dets]
    pred_scores = y_pred["detection_scores"][0].numpy()[:num_dets]
    keep = pred_scores >= CONFIDENCE_THRESHOLD
    pred_boxes = pred_boxes[keep]
    pred_classes = pred_classes[keep]

    if pred_boxes.shape[0] == 0:
        for c in gt_classes:
            matrix[label_names[c]][NO_DETECTION] += 1
        continue

    ious = iou_matrix(gt_boxes, pred_boxes)
    best_pred_idx = np.argmax(ious, axis=1)
    best_iou = np.max(ious, axis=1)

    for gt_i, c in enumerate(gt_classes):
        true_name = label_names[c]
        if best_iou[gt_i] >= IOU_MATCH_THRESHOLD:
            predicted_name = label_names[pred_classes[best_pred_idx[gt_i]]]
        else:
            predicted_name = NO_DETECTION
        matrix[true_name][predicted_name] += 1

# condensed, readable summary
print(f"\n{'class':16s} {'n_gt':>5s} {'correct':>8s} {'missed':>8s}  top confusion")
for true_name in sorted(matrix, key=lambda n: -sum(matrix[n].values())):
    row = matrix[true_name]
    n_gt = sum(row.values())
    correct = row.get(true_name, 0) / n_gt
    missed = row.get(NO_DETECTION, 0) / n_gt
    wrong = Counter({k: v for k, v in row.items() if k not in (true_name, NO_DETECTION)})
    top = wrong.most_common(1)
    top_str = f"{top[0][0]} ({top[0][1] / n_gt:.0%})" if top else "-"
    print(f"{true_name:16s} {n_gt:5d} {correct:8.0%} {missed:8.0%}  {top_str}")

# full matrix, for anything the condensed view hides
all_pred_labels = sorted({p for row in matrix.values() for p in row} | set(label_names[1:]))
out_path = "confusion_matrix.csv"
with open(out_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["true_class"] + all_pred_labels)
    for true_name in sorted(matrix):
        w.writerow([true_name] + [matrix[true_name].get(p, 0) for p in all_pred_labels])
print(f"\nfull matrix written to {out_path}")
