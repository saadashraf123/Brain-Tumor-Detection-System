from ctypes.wintypes import RGB
import enum
from pickletools import optimize
from statistics import mode
from sklearn import metrics
from sklearn.utils import shuffle
import tensorflow as tf
import cv2
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import normalize
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils.np_utils import to_categorical


image_directory = 'datasets/'
no_tumor_detection = os.listdir(image_directory + 'no/')
yes_tumor_detection = os.listdir(image_directory + 'yes/')

dataset = []
label = []
INPUT_SIZE = 64


for i, image_name in enumerate(no_tumor_detection):
    if(image_name.split('.')[1] == "jpg"):
        image = cv2.imread(image_directory + "no/"+ image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((INPUT_SIZE,INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(0)

for i, image_name in enumerate(yes_tumor_detection):
    if(image_name.split('.')[1] == "jpg"):
        image = cv2.imread(image_directory + "yes/"+ image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((INPUT_SIZE,INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(1)


dataset = np.array(dataset)
label = np.array(label)

# print(len(dataset))
# print(len(label))

x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=0)


# print(x_train.shape)
# print(y_train.shape)

x_train =  normalize(x_train, axis=1)
x_test =  normalize(x_test, axis=1)

# y_train = to_categorical(y_train, num_classes=2)
# y_test = to_categorical(y_test, num_classes=2)


#Model Building

model= Sequential()

model.add(Conv2D(32,(3,3), input_shape= (INPUT_SIZE,INPUT_SIZE,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Conv2D(32,(3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Conv2D(INPUT_SIZE,(3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Flatten())
model.add(Dense(INPUT_SIZE))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



# model.add(Dense(2))
# model.add(Activation('softmax'))
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


model.fit(x_train, y_train, batch_size=16, verbose=1, epochs=10, validation_data=(x_test,y_test), shuffle=False )

model.save('BrainTumorEpochs.h5')
# model.save('BrainTumorCategorical5Epochs.h5')
# model.save('BrainTumorCategoricalEpochs.h5')