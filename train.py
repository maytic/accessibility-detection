import tensorflow as tf

from mediapipe_model_maker import object_detector
from mediapipe_model_maker import quantization



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
spec = object_detector.SupportedModels.MOBILENET_V2_I320

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
model._create_and_train_model(train_data, validation_data)


# evaluate the model
loss, coco_metrics = model.evaluate(validation_data, batch_size=32)

print(coco_metrics)


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