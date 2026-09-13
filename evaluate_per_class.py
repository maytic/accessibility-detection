"""Per-class AP/AR breakdown, working around a real gap in mediapipe-model-maker:
ObjectDetector.evaluate() sets per_category_metrics=True, but tf-models-official's
coco_evaluator only reads per-category numbers from a `category_stats` attribute
that vanilla pycocotools' COCOeval never sets -- so that flag is silently a no-op.

This patches pycocotools.cocoeval.COCOeval.summarize() to actually compute and
store category_stats (shape [12 metrics][K categories], matching what
tf-models-official's _retrieve_per_category_metrics() already expects), using
the same self.eval['precision'] / self.eval['recall'] arrays the aggregate
numbers are already computed from. Nothing about training or the aggregate
metrics changes -- this only fills in the missing per-category slice.
"""
import json

import numpy as np
from pycocotools import cocoeval as _cocoeval_module

from mediapipe_model_maker import object_detector

# Must match whichever spec produced the checkpoint in exported_model/float_ckpt.
TRAINED_SPEC = object_detector.SupportedModels.MOBILENET_V2_I320
EXPORT_DIR = "exported_model"

_orig_summarize = _cocoeval_module.COCOeval.summarize


def _summarize_with_categories(self):
    _orig_summarize(self)  # unchanged: prints + computes the aggregate self.stats

    p = self.params
    K = self.eval["precision"].shape[2]

    def cat_summarize(k, ap, iou_thr=None, area_rng="all", max_dets=100):
        aind = [i for i, a in enumerate(p.areaRngLbl) if a == area_rng]
        mind = [i for i, m in enumerate(p.maxDets) if m == max_dets]
        if ap:
            s = self.eval["precision"]
            if iou_thr is not None:
                t = np.where(np.isclose(iou_thr, p.iouThrs))[0]
                s = s[t]
            s = s[:, :, k, aind, mind]
        else:
            s = self.eval["recall"]
            if iou_thr is not None:
                t = np.where(np.isclose(iou_thr, p.iouThrs))[0]
                s = s[t]
            s = s[:, k, aind, mind]
        valid = s[s > -1]
        return float(np.mean(valid)) if valid.size else -1.0

    # order must match tf-models-official's coco_evaluator.py metrics_dict_keys
    specs = [
        (1, None, "all", 100), (1, 0.5, "all", 100), (1, 0.75, "all", 100),
        (1, None, "small", 100), (1, None, "medium", 100), (1, None, "large", 100),
        (0, None, "all", 1), (0, None, "all", 10), (0, None, "all", 100),
        (0, None, "small", 100), (0, None, "medium", 100), (0, None, "large", 100),
    ]
    self.category_stats = [
        np.array([cat_summarize(k, *spec) for k in range(K)]) for spec in specs
    ]


_cocoeval_module.COCOeval.summarize = _summarize_with_categories


validation_data = object_detector.Dataset.from_coco_folder(
    "dataset/mm/val",
    cache_dir="dataset/mm/val_cache",
)

spec = object_detector.SupportedModels.get(TRAINED_SPEC)
hparams = object_detector.HParams(export_dir=EXPORT_DIR)
model_options = object_detector.ModelOptions()
model = object_detector.ObjectDetector(
    model_spec=spec,
    label_names=validation_data.label_names,
    hparams=hparams,
    model_options=model_options,
)
model.restore_float_ckpt()

_, coco_metrics = model.evaluate(validation_data, batch_size=32)

cat_id_to_name = {
    c["id"]: c["name"]
    for c in json.loads(open("dataset/mm/val/labels.json").read())["categories"]
}

per_class = {}
for key, value in coco_metrics.items():
    if "ByCategory" not in key:
        continue
    metric_name, cat_id_str = key.rsplit("/", 1)
    cat_id = int(cat_id_str)
    per_class.setdefault(cat_id_to_name.get(cat_id, cat_id_str), {})[metric_name] = value

print(f"\n{'class':16s} {'AP':>7s} {'AP50':>7s} {'AR100':>7s}")
for name, metrics in sorted(
    per_class.items(),
    key=lambda kv: kv[1].get("Precision mAP ByCategory", -1),
    reverse=True,
):
    ap = metrics.get("Precision mAP ByCategory", float("nan"))
    ap50 = metrics.get("Precision mAP ByCategory@50IoU", float("nan"))
    ar100 = metrics.get("Recall AR@100 ByCategory", float("nan"))
    print(f"{name:16s} {ap:7.3f} {ap50:7.3f} {ar100:7.3f}")


# class                 AP    AP50   AR100
# stop sign          0.621   0.710   0.651
# fire hydrant       0.456   0.611   0.536
# bus                0.370   0.516   0.495
# motorcycle         0.275   0.571   0.424
# person             0.262   0.518   0.367
# car                0.171   0.332   0.301
# truck              0.158   0.252   0.398
# bicycle            0.136   0.286   0.225
# bench              0.098   0.218   0.236
# traffic light      0.049   0.115   0.125
# dog                0.000   0.000   0.000
# backpack           0.000   0.000   0.000
# suitcase           0.000   0.000   0.000
# chair              0.000   0.000   0.000
# potted plant       0.000   0.000   0.000
