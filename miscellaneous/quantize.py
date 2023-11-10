import tensorflow as tf
from model import get_model
import tensorflow_model_optimization as tfmot
import os
import pandas as pd
pd.set_option('display.max_rows', None)

import tensorflow as tf
# tf.keras.backend.set_floatx('float16')
from data_loader import DATALOADER
from preprocess import get_dfs

import matplotlib.pyplot as plt
from utils import ohe_dict_func

import pickle
# Get DATA
train,test = get_dfs()
train['filename']= "pose_estimation/dataset-001/dataset/"+train['filename']
test['filename']= "pose_estimation/dataset-001/dataset/"+test['filename']
train.reset_index(drop=True,inplace=True)
test.reset_index(drop=True,inplace=True)

class_list = train['class'].value_counts().keys()[:70]
train = train[train["class"].isin(class_list)]
test = test[test["class"].isin(class_list)]

num_classes=74
ohe_dictionary = ohe_dict_func(class_list)
# print(ohe_dictionary)
# exit()
BATCH_SIZE = 8
train_dataset = DATALOADER(train,image_size=(240,240,3),batch_size=BATCH_SIZE, num_classes=num_classes,ohe_dictionary=ohe_dictionary)
valid_dataset =  DATALOADER(test,image_size=(240,240,3),batch_size=1,num_classes=num_classes,ohe_dictionary=ohe_dictionary)


model=get_model(None)

model.load_weights("./ckpt/model.h5")
# print(model.summary())

quantize_model = tfmot.quantization.keras.quantize_model(model)
quantize_model.compile(loss={'quant_bbox':tf.keras.losses.MeanSquaredError(),'quant_classes':"categorical_crossentropy"},
             optimizer=tf.keras.optimizers.Adam(lr = 0.00001),
             metrics={'quant_bbox':'accuracy',"quant_classes":'categorical_accuracy'})

history_quantized = quantize_model.fit(
    train_dataset,
    batch_size=BATCH_SIZE,
    validation_data=valid_dataset,
    verbose=1,
    epochs=1,
#     callbacks=callbacks
)

converter = tf.lite.TFLiteConverter.from_keras_model(quantize_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

quantized_tflite_model = converter.convert()

with open('./ckpt/quantized_model.tflite', 'wb') as f:
    f.write(quantized_tflite_model)

    import tempfile

# Create float TFLite model.
float_converter = tf.lite.TFLiteConverter.from_keras_model(model)
float_tflite_model = float_converter.convert()

# Measure sizes of models.
_, float_file = tempfile.mkstemp('.tflite')
_, quant_file = tempfile.mkstemp('.tflite')

with open(quant_file, 'wb') as f:
  f.write(quantized_tflite_model)

with open(float_file, 'wb') as f:
  f.write(float_tflite_model)

print("Float model in Mb:", os.path.getsize(float_file) / float(2**20))
print("Quantized model in Mb:", os.path.getsize(quant_file) / float(2**20))

