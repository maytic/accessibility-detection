import os

import tensorflow as tf

from mediapipe_model_maker import object_detector



train_data = object_detector.Dataset.from_coco_folder(
    'dataset/mm/train',
    cache_dir='dataset/mm/train_cache'
)

validation_data = object_detector.Dataset.from_coco_folder(
    'dataset/mm/val',
    cache_dir='dataset/mm/val_cache'
)

print(train_data.size, 'training images,', len(train_data.label_names), 'classes')
print(train_data.label_names)

print(validation_data.size, 'validation images,', len(validation_data.label_names), 'classes')
print(validation_data.label_names)


# get the model specification
spec = object_detector.SupportedModels.MOBILENET_MULTI_AVG_I384

hyper_params = object_detector.HParams(
    learning_rate=0.3,
    batch_size=32,
    epochs=30,
    export_dir='exported_model'
)

# default l2 weight decay
model_options = object_detector.ModelOptions()

options = object_detector.ObjectDetectorOptions(
    supported_model=spec,
    model_options=model_options,
    hparams=hyper_params
)

spec = object_detector.SupportedModels.get(options.supported_model)
model = object_detector.ObjectDetector(
    model_spec=spec,
    label_names=train_data.label_names,
    hparams=options.hparams,
    model_options=options.model_options,
)
model._callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_total_loss',
        patience=5,
        restore_best_weights=True,
    )
]

float_ckpt_path = os.path.join(hyper_params.export_dir, 'float_ckpt.index')
if os.path.exists(float_ckpt_path):
    print(f'found existing float checkpoint at {float_ckpt_path}, skipping training')
    model.restore_float_ckpt()
else:
    model._create_and_train_model(train_data, validation_data)


# evaluate the float model
loss, coco_metrics = model.evaluate(validation_data, batch_size=32)
print('float model:', coco_metrics)


# quantization-aware training, then export as an int8 tflite.
# QAT mutates model._model in place, so re-running QAT on this same model
# instance requires model.restore_float_ckpt() first (see its docstring).
qat_hparams = object_detector.QATHParams()
model.quantization_aware_training(train_data, validation_data, qat_hparams=qat_hparams)

qat_loss, qat_coco_metrics = model.evaluate(validation_data, batch_size=32)
print('qat int8 model:', qat_coco_metrics)

model.export_model(model_name='accessibility_detector.tflite')


# MOBILENET_V2 baseline
# Running per image evaluation...
# Evaluate annotation type *bbox*
# DONE (t=2.36s).
# Accumulating evaluation results...
# DONE (t=0.31s).
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.140
#  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.212
#  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.140
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.001
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.131
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.256
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.145
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.201
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.214
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.012
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.205
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.369
# {'AP': 0.13996129, 'AP50': 0.2119513, 'AP75': 0.13975106, 'APs': 0.00083572534, 'APm': 0.13124111, 'APl': 0.25602874, 'ARmax1': 0.14464103, 'ARmax10': 0.20130248, 'ARmax100': 0.21388994, 'ARs': 0.011894351, 'ARm': 0.20536315, 'ARl': 0.3691285}
#


# MOBILENET_V2 with more balanced dastaset
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.156
#  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.254
#  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.157
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.003
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.137
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.280
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.149
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.216
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.230
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.022
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.239
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.389
# {'AP': 0.15611348, 'AP50': 0.25418988, 'AP75': 0.1570107, 'APs': 0.0029100107, 'APm': 0.13736363, 'APl': 0.28030118, 'ARmax1': 0.14857952, 'ARmax10': 0.21599445, 'ARmax100': 0.22970934, 'ARs': 0.02214265, 'ARm': 0.23865238, 'ARl': 0.3886058}

# MOBILENET_V2_I320 scores
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.173
#  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.275
#  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.181
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.006
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.161
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.303
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.167
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.236
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.251
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.029
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.269
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.405
# {'AP': 0.17315406, 'AP50': 0.27531525, 'AP75': 0.1806985, 'APs': 0.0063945707, 'APm':

# MOBILENET_MULTI_AVG_I384
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.259
#  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.426
#  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.266
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.020
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.264
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.421
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.229
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.372
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.396
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.061
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.443
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.588
# {'AP': 0.2593608, 'AP50': 0.42559987, 'AP75': 0.26576367, 'APs': 0.019612327, 'APm': 0.26374725, 'APl': 0.42092347, 'ARmax1': 0.22932543, 'ARmax10': 0.3719258, 'ARmax100': 0.3958522, 'ARs': 0.060630042, 'ARm': 0.44323882, 'ARl': 0.58805615}