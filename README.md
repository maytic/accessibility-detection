# Accessibility Detection

An on-device object detector for a navigation-assist use case: it flags people,
vehicles, and common street obstacles (stop signs, fire hydrants, benches,
etc.) so a mobile app can call them out to the user. The model is trained
with `mediapipe_model_maker` and exported as a quantized int8 TFLite file for
Android.

## Classes

15 COCO categories, chosen for relevance to street navigation:

```
person, bicycle, car, motorcycle, bus, truck, traffic light, fire hydrant,
stop sign, bench, backpack, suitcase, dog, chair, potted plant
```

## Pipeline

| Script | Purpose |
|---|---|
| `download_dataset.py` | Pulls a class-balanced subset of COCO 2017 (3000 train / 500 val images) for the 15 target classes |
| `dataset_restructure.py` | Symlinks/copies the subset into the `dataset/mm/{train,val}` layout `mediapipe_model_maker` expects |
| `train.py` | Fine-tunes `MOBILENET_MULTI_AVG_I384` (MobileNet-SSD), evaluates the float model, then exports an int8 TFLite file via post-training quantization |
| `evaluate_per_class.py` | Per-class AP/AR breakdown for the float checkpoint |
| `confusion_matrix.py` | Per-class confusion matrix for the float checkpoint |
| `evaluate_tflite.py` | Scores the exported `.tflite` file through the real MediaPipe Tasks inference path (anchor decoding + NMS), for numbers comparable to the float model |
| `predict.py` | Runs the exported `.tflite` model on arbitrary images and saves annotated output |

## Final metrics (validation set, 500 images)

**Float model** (`MOBILENET_MULTI_AVG_I384`, before quantization):

| AP | AP50 | AP75 | APs | APm | APl | ARmax1 | ARmax10 | ARmax100 | ARs | ARm | ARl |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.259 | 0.426 | 0.266 | 0.020 | 0.264 | 0.421 | 0.229 | 0.372 | 0.396 | 0.061 | 0.443 | 0.588 |

**Quantized int8 TFLite model** (what actually ships to Android, scored via `evaluate_tflite.py`):

| AP | AP50 | AP75 | APs | APm | APl | ARmax1 | ARmax10 | ARmax100 | ARs | ARm | ARl |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.192 | 0.366 | 0.192 | 0.050 | 0.223 | 0.335 | 0.183 | 0.267 | 0.280 | 0.084 | 0.317 | 0.452 |

Roughly a 25% relative AP drop from int8 quantization — expected, and the
quantized model is what's used for on-device inference.

Per-class AP/AP50/AR100 for the float model (`evaluate_per_class.py`):

| class | AP | AP50 | AR100 |
|---|---|---|---|
| stop sign | 0.626 | 0.730 | 0.663 |
| fire hydrant | 0.464 | 0.615 | 0.551 |
| dog | 0.441 | 0.689 | 0.586 |
| bus | 0.407 | 0.582 | 0.543 |
| motorcycle | 0.358 | 0.646 | 0.467 |
| person | 0.300 | 0.562 | 0.412 |
| car | 0.216 | 0.402 | 0.351 |
| truck | 0.193 | 0.343 | 0.435 |
| potted plant | 0.191 | 0.428 | 0.405 |
| bicycle | 0.188 | 0.368 | 0.298 |
| suitcase | 0.169 | 0.308 | 0.342 |
| chair | 0.130 | 0.243 | 0.303 |
| bench | 0.090 | 0.189 | 0.262 |
| traffic light | 0.066 | 0.180 | 0.163 |
| backpack | 0.052 | 0.100 | 0.157 |

## Quantization note

`train.py` exports int8 via standard post-training quantization rather than
`mediapipe_model_maker`'s quantization-aware training (QAT) API. QAT is
broken for the `MOBILENET_MULTI_AVG` model family in the current
`mediapipe_model_maker` release — it silently wrecks the detection head and
produces a near-zero-AP model. This is a known, unresolved upstream issue
(`mediapipe_model_maker` itself is deprecated and no longer actively
maintained), not something specific to this dataset or config. Post-training
quantization sidesteps the buggy retraining path entirely.

## Android usage

Only `exported_model/accessibility_detector.tflite` is needed — labels,
anchor boxes, and NMS config are embedded in the file. Add
`com.google.mediapipe:tasks-vision` and load it with the MediaPipe Tasks
`ObjectDetector` API.
