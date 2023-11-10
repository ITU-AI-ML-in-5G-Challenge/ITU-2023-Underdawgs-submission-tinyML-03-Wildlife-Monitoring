import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
BATCH_SIZE = 256
num_classes=4
image_size = (240,240,3)


class DATALOADER(tf.keras.utils.Sequence):
    def __init__(self, df,image_size, batch_size, num_classes, ohe_dictionary,shuffle=True):
        self.df = df
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_classes=num_classes
        self.image_size=image_size
        self.ohe_dictionary=ohe_dictionary
        self.on_epoch_end()
    
    def __len__(self):
        return len(self.df) // self.batch_size
    
    def __getitem__(self, index):
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        X, y,z = self.__data_generation(indices)
        return X, [y,z]
    
    def on_epoch_end(self):
        self.indices = np.arange(len(self.df))
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def load(self,input_path):

        input_image = cv2.imread(input_path)
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        input_image = cv2.resize(input_image, (image_size[0],image_size[1]))
        input_image=input_image/255.0
        input_image = tf.image.convert_image_dtype(input_image, tf.float32)
        
        return input_image
    
    def __data_generation(self, indices):
        X = np.empty((self.batch_size, self.image_size[0], self.image_size[1],3), dtype=np.float32)
        y = np.empty((self.batch_size, 4), dtype=np.float32)
        z = np.empty((self.batch_size,70),dtype=np.float32)
        for i, idx in enumerate(indices):
            # Load audio file and compute MFCC
            file_path = self.df['filename'].iloc[idx]

            # exit()
            X[i,] = self.load(file_path)
            # y[i,] = np.concatenate((self.df[['xmin', 'ymin', 'xmax', 'ymax']].iloc[idx].values,  np.array(self.ohe_dictionary[self.df['class'].iloc[idx]])))
            y[i,]=self.df[['xmin', 'ymin', 'xmax', 'ymax']].iloc[idx].values
            z[i,]=self.ohe_dictionary[self.df['class'].iloc[idx]]
            # print(z)
        return X, y,z

    
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# import librosa
# import cv2
# BATCH_SIZE = 256
# num_classes=4
# image_size = (240,240,3)


# class DATALOADER(tf.keras.utils.Sequence):
#     def __init__(self, df,image_size, batch_size, num_classes, ohe_dictionary,shuffle=True):
#         self.df = df
#         self.batch_size = batch_size
#         self.shuffle = shuffle
#         self.num_classes=num_classes
#         self.image_size=image_size
#         self.ohe_dictionary=ohe_dictionary
#         self.on_epoch_end()
    
#     def __len__(self):
#         return len(self.df) // self.batch_size
    
#     def __getitem__(self, index):
#         indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
#         X, y = self.__data_generation(indices)
#         return X, y
    
#     def on_epoch_end(self):
#         self.indices = np.arange(len(self.df))
#         if self.shuffle:
#             np.random.shuffle(self.indices)
    
#     def load(self,input_path):

#         input_image = cv2.imread(input_path)
#         input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
#         input_image = cv2.resize(input_image, (image_size[0],image_size[1]))
#         input_image = tf.image.convert_image_dtype(input_image, tf.float16)

#         return input_image
    
#     def __data_generation(self, indices):
#         X = np.empty((self.batch_size, self.image_size[0], self.image_size[1],3), dtype=np.float32)
#         y = np.empty((self.batch_size, self.num_classes), dtype=np.float32)
#         z = np.empty()
#         for i, idx in enumerate(indices):
#             # Load audio file and compute MFCC
#             file_path = self.df['filename'].iloc[idx]

#             # exit()
#             X[i,] = self.load(file_path)
#             # y[i,] = np.concatenate((self.df[['xmin', 'ymin', 'xmax', 'ymax']].iloc[idx].values,  np.array(self.ohe_dictionary[self.df['class'].iloc[idx]])))

#         return X, y

    
