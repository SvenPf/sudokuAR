# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"    

import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import normalize
import cv2

# Here Code for CNN

DATADIR = "../../data/"

pickle_in = open(DATADIR + "train_x.pickle", "rb")
train_x = pickle.load(pickle_in)

pickle_in = open(DATADIR + "train_y.pickle", "rb")
train_y = np.array(pickle.load(pickle_in))  # needs to be np array

train_x = normalize(train_x, axis=-1) # normalize pixels

model = Sequential()  # feed forward network

# first layer (conv + pooling)
model.add(Conv2D(32, (3, 3), input_shape=train_x.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# second layer (conv + pooling)
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# fully connected layer
model.add(Flatten())  # converts 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('tanh'))

# output layer (9 classes)
model.add(Dense(9))
model.add(Activation('softmax'))  # better then sigmoid for multiple classes

# sparse_categorical_crossentropy for multiple classes with labels [0], [1], [2], [3], [4], [5], [6], [7], [8]
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(train_x, train_y, batch_size=32, epochs=10, validation_split=0.1)

model.save("num_classifier.model")
