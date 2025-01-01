# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 11:49:24 2024

@author: user
"""
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from glob import glob
from tqdm.auto import tqdm

from tensorflow import keras
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import plot_model

IMAGE_LIB = 'D:/Face_Off/Face_A/*.jpg' #input
MASK_LIB = 'D:/Face_Off/Face_B/*.jpg'  #output

IMG_SIZEx = 320 #320  312
IMG_SIZEy = 256
# SEED = 5566
# NUM_CLASSES = 2
print(sorted(glob(IMAGE_LIB)))


img_paths = sorted(glob(IMAGE_LIB))
mask_paths = sorted(glob(MASK_LIB))

x_data = np.empty((len(img_paths), IMG_SIZEy, IMG_SIZEx, 3))
y_data = np.empty((len(img_paths), IMG_SIZEy, IMG_SIZEx, 3))

for i, path in enumerate(tqdm(img_paths)):
    print(path)
    # read input image
    img = cv2.imread(path) 
    img = cv2.resize(img, (IMG_SIZEx, IMG_SIZEy))
    print(img.shape)
    img = img / 255
    x_data[i] = img
    filename_img = os.path.split(path)[1] # filename_img: seg1.png
    filename = filename_img.split('.')[0] # filename: seg1
    print(filename)
    # read output mask image
    mask = cv2.imread(os.path.join('D:/Face_Off/Face_B', f'{filename}_mask_0_r.jpg')) # seg1_mask_0.png, seg1_mask_1.png, seg1_mask_2.png
    print(mask.shape)
    # mask = mask / 255.
    # mask[mask>=0.5] = 1.
    # mask[mask<0.5] = 0
    mask = cv2.resize(mask, (IMG_SIZEx, IMG_SIZEy))
    y_data[i] = mask / 255
print(y_data.shape)


fig, ax = plt.subplots(1,4, figsize = (16,4))
ax[0].imshow(x_data[0])
ax[1].imshow(x_data[1])
ax[2].imshow(y_data[0])
ax[3].imshow(y_data[1])
plt.show()
print(x_data[0])
print("=============")
print(y_data[0])
# print(y_data[1][200][270])
# print(y_data[1][200][50])

print('x_data.shape:', x_data.shape, ' ', 'y_data.shape:', y_data.shape)


# Unet with Conv2DTranspose
import matplotlib.pyplot as plt
f = 4#2
'''
input_layer = Input(shape=(IMG_SIZEy, IMG_SIZEx, 3))
l = Conv2D(filters=8*f, kernel_size=(3,3), activation='relu', padding='same')(input_layer)
c1 = Conv2D(filters=8*f, kernel_size=(3,3), activation='relu', padding='same')(l)
l = MaxPool2D(strides=(2,2))(c1)
l = Conv2D(filters=16*f, kernel_size=(3,3), activation='relu', padding='same')(l)
c2 = Conv2D(filters=16*f, kernel_size=(3,3), activation='relu', padding='same')(l)
l = MaxPool2D(strides=(2,2))(c2)
l = Conv2D(filters=32*f, kernel_size=(3,3), activation='relu', padding='same')(l)
c3 = Conv2D(filters=32*f, kernel_size=(3,3), activation='relu', padding='same')(l)
l = MaxPool2D(strides=(2,2))(c3)
l = Conv2D(filters=64*f, kernel_size=(3,3), activation='relu', padding='same')(l)
c4 = Conv2D(filters=64*f, kernel_size=(3,3), activation='relu', padding='same')(l)
l = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c4), c3], axis=-1)
l = Conv2D(filters=32*f, kernel_size=(3,3), activation='relu', padding='same')(l)
l = Conv2D(filters=32*f, kernel_size=(2,2), activation='relu', padding='same')(l)
l = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(l), c2], axis=-1)
l = Conv2D(filters=16*f, kernel_size=(3,3), activation='relu', padding='same')(l)
l = Conv2D(filters=16*f, kernel_size=(3,3), activation='relu', padding='same')(l)
l = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(l), c1], axis=-1)
l = Conv2D(filters=8*f, kernel_size=(3,3), activation='relu', padding='same')(l)
l = Conv2D(filters=8*f, kernel_size=(3,3), activation='relu', padding='same')(l)
output_layer = Conv2D(filters=3, kernel_size=(1,1), activation='sigmoid')(l)
'''

input_layer = Input(shape=(IMG_SIZEy, IMG_SIZEx, 3))
l = Conv2D(filters=16*f, kernel_size=(3,3), activation='relu', padding='same')(input_layer)
c1 = Conv2D(filters=16*f, kernel_size=(3,3), activation='relu', padding='same')(l)
l = MaxPool2D(strides=(2,2))(c1)
l = Conv2D(filters=16*f, kernel_size=(3,3), activation='relu', padding='same')(l)
c2 = Conv2D(filters=16*f, kernel_size=(3,3), activation='relu', padding='same')(l)
l = MaxPool2D(strides=(2,2))(c2)
l = Conv2D(filters=32*f, kernel_size=(3,3), activation='relu', padding='same')(l)
c3 = Conv2D(filters=32*f, kernel_size=(3,3), activation='relu', padding='same')(l)
l = MaxPool2D(strides=(2,2))(c3)
l = Conv2D(filters=64*f, kernel_size=(3,3), activation='relu', padding='same')(l)
c4 = Conv2D(filters=64*f, kernel_size=(3,3), activation='relu', padding='same')(l)
l = MaxPool2D(strides=(2,2))(c4)
l = Conv2D(filters=64*f, kernel_size=(3,3), activation='relu', padding='same')(l)
c5 = Conv2D(filters=64*f, kernel_size=(3,3), activation='relu', padding='same')(l)


l = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c5), c4], axis=-1)
l = Conv2D(filters=64*f, kernel_size=(3,3), activation='relu', padding='same')(l)
l = Conv2D(filters=64*f, kernel_size=(2,2), activation='relu', padding='same')(l)
l = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(l), c3], axis=-1)
l = Conv2D(filters=32*f, kernel_size=(3,3), activation='relu', padding='same')(l)
l = Conv2D(filters=32*f, kernel_size=(3,3), activation='relu', padding='same')(l)
l = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(l), c2], axis=-1)
l = Conv2D(filters=16*f, kernel_size=(3,3), activation='relu', padding='same')(l)
l = Conv2D(filters=16*f, kernel_size=(3,3), activation='relu', padding='same')(l)
l = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(l), c1], axis=-1)
l = Conv2D(filters=8*f, kernel_size=(3,3), activation='relu', padding='same')(l)
l = Conv2D(filters=8*f, kernel_size=(3,3), activation='relu', padding='same')(l)
output_layer = Conv2D(filters=3, kernel_size=(1,1), activation='sigmoid')(l)

'''
#U+
input_layer = Input(shape=(IMG_SIZEy, IMG_SIZEx, 3))
conv0_0 = Conv2D(filters=f, kernel_size=(3,3), activation='relu', padding='same')(input_layer)
conv0_0 = Conv2D(filters=f, kernel_size=(3,3), activation='relu', padding='same')(conv0_0)
pool1 = MaxPool2D(strides=(2,2))(conv0_0)

conv1_0 = Conv2D(filters=f*2, kernel_size=(3,3), activation='relu', padding='same')(pool1)
conv1_0 = Conv2D(filters=f*2, kernel_size=(3,3), activation='relu', padding='same')(conv1_0)
pool12 = MaxPool2D(strides=(2,2))(conv1_0)

merge00_10 = concatenate([Conv2DTranspose(f, (2, 2), strides=(2, 2), padding='same')(conv1_0), conv0_0], axis=-1)
conv0_1 = Conv2D(filters=f, kernel_size=(3,3), activation='relu', padding='same')(merge00_10)
conv0_1 = Conv2D(filters=f, kernel_size=(3,3), activation='relu', padding='same')(conv0_1)

conv2_0 = Conv2D(filters=f*4, kernel_size=(3,3), activation='relu', padding='same')(pool12)
conv2_0 = Conv2D(filters=f*4, kernel_size=(3,3), activation='relu', padding='same')(conv2_0)
pool13 = MaxPool2D(strides=(2,2))(conv2_0)

merge10_20 = concatenate([Conv2DTranspose(f, (2, 2),  strides=(2, 2), padding='same')(conv2_0), conv1_0], axis=-1)
conv1_1 = Conv2D(filters=f, kernel_size=(3,3), activation='relu', padding='same')(merge10_20)
conv1_1 = Conv2D(filters=f, kernel_size=(3,3), activation='relu', padding='same')(conv1_1)

merge01_11 = concatenate([Conv2DTranspose(f, (2, 2),  strides=(2, 2), padding='same')(conv1_1), conv0_1], axis=-1)
conv0_2 = Conv2D(filters=f, kernel_size=(3,3), activation='relu', padding='same')(merge01_11)
conv0_2 = Conv2D(filters=f, kernel_size=(3,3), activation='relu', padding='same')(conv0_2)

conv3_0 = Conv2D(filters=f*8, kernel_size=(3,3), activation='relu', padding='same')(pool13)
conv3_0 = Conv2D(filters=f*8, kernel_size=(3,3), activation='relu', padding='same')(conv3_0)

merge20_30 = concatenate([Conv2DTranspose(f, (2, 2),  strides=(2, 2), padding='same')(conv3_0), conv2_0], axis=-1)
conv2_1 = Conv2D(filters=f, kernel_size=(3,3), activation='relu', padding='same')(merge20_30)
conv2_1 = Conv2D(filters=f, kernel_size=(3,3), activation='relu', padding='same')(conv2_1)

merge11_21 = concatenate([Conv2DTranspose(f, (2, 2),  strides=(2, 2), padding='same')(conv2_1), conv1_1], axis=-1)
conv1_2 = Conv2D(filters=f, kernel_size=(3,3), activation='relu', padding='same')(merge11_21)
conv1_2 = Conv2D(filters=f, kernel_size=(3,3), activation='relu', padding='same')(conv1_2)

merge02_12 = concatenate([Conv2DTranspose(f, (2, 2),  strides=(2, 2), padding='same')(conv1_2), conv0_2], axis=-1)
conv0_3 = Conv2D(filters=f, kernel_size=(3,3), activation='relu', padding='same')(merge02_12)
conv0_3 = Conv2D(filters=f, kernel_size=(3,3), activation='relu', padding='same')(conv0_3)

output_layer = Conv2D(filters=3, kernel_size=(1,1), activation='sigmoid')(conv0_3)
'''
'''
#U+
input_layer = Input(shape=(IMG_SIZEy, IMG_SIZEx, 3))
conv0_0 = Conv2D(filters=f*4, kernel_size=(3,3), activation='relu', padding='same')(input_layer)
conv0_0 = Conv2D(filters=f*4, kernel_size=(3,3), activation='relu', padding='same')(conv0_0)
pool1 = MaxPool2D(strides=(2,2))(conv0_0)

conv1_0 = Conv2D(filters=f*4, kernel_size=(3,3), activation='relu', padding='same')(pool1)
conv1_0 = Conv2D(filters=f*4, kernel_size=(3,3), activation='relu', padding='same')(conv1_0)
pool12 = MaxPool2D(strides=(2,2))(conv1_0)

merge00_10 = concatenate([Conv2DTranspose(f, (2, 2), strides=(2, 2), padding='same')(conv1_0), conv0_0], axis=-1)
conv0_1 = Conv2D(filters=f, kernel_size=(3,3), activation='relu', padding='same')(merge00_10)
conv0_1 = Conv2D(filters=f, kernel_size=(3,3), activation='relu', padding='same')(conv0_1)

conv2_0 = Conv2D(filters=f*4, kernel_size=(3,3), activation='relu', padding='same')(pool12)
conv2_0 = Conv2D(filters=f*4, kernel_size=(3,3), activation='relu', padding='same')(conv2_0)
pool13 = MaxPool2D(strides=(2,2))(conv2_0)

merge10_20 = concatenate([Conv2DTranspose(f, (2, 2),  strides=(2, 2), padding='same')(conv2_0), conv1_0], axis=-1)
conv1_1 = Conv2D(filters=f, kernel_size=(3,3), activation='relu', padding='same')(merge10_20)
conv1_1 = Conv2D(filters=f, kernel_size=(3,3), activation='relu', padding='same')(conv1_1)

merge01_11 = concatenate([Conv2DTranspose(f, (2, 2),  strides=(2, 2), padding='same')(conv1_1), conv0_1], axis=-1)
conv0_2 = Conv2D(filters=f, kernel_size=(3,3), activation='relu', padding='same')(merge01_11)
conv0_2 = Conv2D(filters=f, kernel_size=(3,3), activation='relu', padding='same')(conv0_2)

conv3_0 = Conv2D(filters=f*8, kernel_size=(3,3), activation='relu', padding='same')(pool13)
conv3_0 = Conv2D(filters=f*8, kernel_size=(3,3), activation='relu', padding='same')(conv3_0)
pool14 = MaxPool2D(strides=(2,2))(conv3_0)

merge20_30 = concatenate([Conv2DTranspose(f, (2, 2),  strides=(2, 2), padding='same')(conv3_0), conv2_0], axis=-1)
conv2_1 = Conv2D(filters=f, kernel_size=(3,3), activation='relu', padding='same')(merge20_30)
conv2_1 = Conv2D(filters=f, kernel_size=(3,3), activation='relu', padding='same')(conv2_1)

merge11_21 = concatenate([Conv2DTranspose(f, (2, 2),  strides=(2, 2), padding='same')(conv2_1), conv1_1], axis=-1)
conv1_2 = Conv2D(filters=f, kernel_size=(3,3), activation='relu', padding='same')(merge11_21)
conv1_2 = Conv2D(filters=f, kernel_size=(3,3), activation='relu', padding='same')(conv1_2)

merge02_12 = concatenate([Conv2DTranspose(f, (2, 2),  strides=(2, 2), padding='same')(conv1_2), conv0_2], axis=-1)
conv0_3 = Conv2D(filters=f, kernel_size=(3,3), activation='relu', padding='same')(merge02_12)
conv0_3 = Conv2D(filters=f, kernel_size=(3,3), activation='relu', padding='same')(conv0_3)

conv4_0 = Conv2D(filters=f*16, kernel_size=(3,3), activation='relu', padding='same')(pool14)
conv4_0 = Conv2D(filters=f*16, kernel_size=(3,3), activation='relu', padding='same')(conv4_0)

merge30_40 = concatenate([Conv2DTranspose(f, (2, 2),  strides=(2, 2), padding='same')(conv4_0), conv3_0], axis=-1)
conv3_1 = Conv2D(filters=f, kernel_size=(3,3), activation='relu', padding='same')(merge30_40)
conv3_1 = Conv2D(filters=f, kernel_size=(3,3), activation='relu', padding='same')(conv3_1)

merge21_31 = concatenate([Conv2DTranspose(f, (2, 2),  strides=(2, 2), padding='same')(conv3_1), conv2_1], axis=-1)
conv2_2 = Conv2D(filters=f, kernel_size=(3,3), activation='relu', padding='same')(merge21_31)
conv2_2 = Conv2D(filters=f, kernel_size=(3,3), activation='relu', padding='same')(conv2_2)

merge12_22 = concatenate([Conv2DTranspose(f, (2, 2),  strides=(2, 2), padding='same')(conv2_2), conv1_2], axis=-1)
conv1_3 = Conv2D(filters=f, kernel_size=(3,3), activation='relu', padding='same')(merge12_22)
conv1_3 = Conv2D(filters=f, kernel_size=(3,3), activation='relu', padding='same')(conv1_3)

merge03_13 = concatenate([Conv2DTranspose(f, (2, 2),  strides=(2, 2), padding='same')(conv1_3), conv0_3], axis=-1)
conv0_4 = Conv2D(filters=f, kernel_size=(3,3), activation='relu', padding='same')(merge03_13)
conv0_4 = Conv2D(filters=f, kernel_size=(3,3), activation='relu', padding='same')(conv0_4)

output_layer = Conv2D(filters=3, kernel_size=(1,1), activation='sigmoid')(conv0_4)
'''
                                                
model = Model(input_layer, output_layer)
model.summary()
plot_model(model)


# model.compile(optimizer=Adam(), loss=keras.losses.categorical_crossentropy)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
# callbacks = [
#     ModelCheckpoint("Unet_Person.h5", save_best_only=True),
#     EarlyStopping(patience=100, restore_best_weights=True)
# ]
# model.fit(x_data,
#      y_data,
#      validation_split=0,
#      batch_size=2,
#      epochs=600,
#      verbose=2,
#      callbacks=callbacks)


hist = model.fit(x_data, y_data, batch_size=2,
                epochs=550, verbose=2)

model.save('face_off_in_house_2508_U_4layers_v1.h5')


img_input = x_data[0:2]
y_pred = model.predict(img_input)
print(y_pred)
# y_pread = y_pred*1.5345
# print(y_pread[1][200][270])
# print(y_pread[1][200][50])

# input img
fig, ax = plt.subplots(1,2,figsize=(12,4))
ax[0].imshow(img_input[0, :, :])  
ax[1].imshow(img_input[1, :, :])  
plt.show()

# pred
fig, ax = plt.subplots(1,2,figsize=(12,4))
ax[0].imshow(y_pred[0, :, :])  
ax[1].imshow(y_pred[1, :, :])  
plt.show()

# Truth
fig, ax = plt.subplots(1,2,figsize=(12,4))
ax[0].imshow(y_data[0, :, :])
ax[1].imshow(y_data[1, :, :])   
plt.show()



