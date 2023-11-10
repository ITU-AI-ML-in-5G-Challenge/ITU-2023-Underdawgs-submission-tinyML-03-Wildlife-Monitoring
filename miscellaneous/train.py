import os
import pandas as pd
pd.set_option('display.max_rows', None)

import tensorflow as tf
# tf.keras.backend.set_floatx('float16')
from data_loader import DATALOADER
from preprocess import get_dfs
from model import get_model
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

# Get Model
model=get_model(num_classes)
# print(mode.summary())
# for i in train_dataset:
#     print(i[2])
#     break
model.compile(loss={'bbox':tf.keras.losses.MeanSquaredError(),'classes':"categorical_crossentropy"},
             optimizer=tf.keras.optimizers.Adam(lr = 0.00001),
             metrics={'bbox':'accuracy',"classes":'categorical_accuracy'})

earlystopping = tf.keras.callbacks.EarlyStopping(monitor =  'val_loss',
                                                 min_delta = 1e-4,
                                                 patience = 5,
                                                 restore_best_weights = True,
                                                 verbose = 1,
                                                mode='min',)

checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath = f"./ckpt/model2.h5",
                                                  monitor = 'val_loss',
                                                  verbose = 1, 
                                                  save_best_only = True,
                                                  save_weights_only = True,
                                                 mode='min')

# scheduler = tf.keras.optimizers.schedules.CosineDecay(0.01,100)
# schedule = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=0)
callbacks = [earlystopping, checkpointer,
#              reduce_lr, 
#              schedule
            ]

history = model.fit(train_dataset,
    epochs=50,
    validation_data=valid_dataset,
    callbacks=callbacks, verbose=True
)
plt.plot(history.history['bbox_accuracy'])
plt.plot(history.history['val_bbox_accuracy'])
plt.title('Bounding Box Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig("Bounding Box Accuracy.jpg")
# plt.show()

plt.plot(history.history['classes_categorical_accuracy'])
plt.plot(history.history['val_classes_categorical_accuracy'])
plt.title('Categorical Class Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig("Classes Categorical Accuracy.jpg")

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig("Model Loss.jpg")

plt.plot(history.history['bbox_loss'])
plt.plot(history.history['val_bbox_loss'])
plt.title('Bounding Box Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig("Bounding Box Loss.jpg")


plt.plot(history.history['classes_loss'])
plt.plot(history.history['val_classes_loss'])
plt.title('Class Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig("Class Loss.jpg")


plt.plot(history.history['classes_categorical_accuracy'])
plt.plot(history.history['val_classes_categorical_accuracy'])
plt.title('Class Categorical Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig("Class Categorical Loss.jpg")


with open('./trainHistoryDict', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

print(history.history.keys())
