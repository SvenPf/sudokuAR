# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import pickle
import numpy as np
import cv2
from sudoku_ar.dictionary.locations import X_TRAIN_DATA, Y_TRAIN_DATA, MODEL_DIR
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

# because of error with gpu
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

MODEL = MODEL_DIR + "num_classifier.model"
IMG_SIZE = 28
BATCH_SIZE = 16
NUM_CLASSES = 9
EPOCHS = 12


def prepare(input_train_set):
    # normalize pixels
    input_train_set = input_train_set.astype('float32')
    input_train_set /= 255
    # reshap because tensorflow expects this shape
    return np.array(input_train_set).reshape(-1, IMG_SIZE, IMG_SIZE, 1)


def train():

    # similar model to mnist model proposed by F. Chollet to deal with handwritten numbers
    # https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py

    pickle_in = open(X_TRAIN_DATA, "rb")
    x_train = pickle.load(pickle_in)

    pickle_in = open(Y_TRAIN_DATA, "rb")
    y_train = np.array(pickle.load(pickle_in))  # needs to be np array

    x_train = x_train.astype('float32')
    x_train /= 255  # normalize pixels

    model = Sequential()  # feed forward network

    # first layer (conv + pooling)
    model.add(Conv2D(32, (3, 3), input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    # second layer (conv + pooling)
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # fully connected layer
    model.add(Flatten())  # converts 3D feature maps to 1D feature vectors
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # output layer (9 classes)
    model.add(Dense(NUM_CLASSES))
    # better then sigmoid for multiple classes
    model.add(Activation('softmax'))

    # sparse_categorical_crossentropy for multiple classes with labels [0], [1], [2], [3], [4], [5], [6], [7], [8]
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',  # Adadelta
                  metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.1)

    print("Saving model in " + MODEL)
    model.save(MODEL)


def predict(image):
    # returns predicted number and the related confidence

    new_model = load_model(MODEL)

    resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

    predictions = new_model.predict([prepare(resized)])

    predicted_class = np.argmax(predictions)

    return (predicted_class + 1), predictions[0][predicted_class]

