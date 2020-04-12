import os
import time
import pickle
import cv2
import numpy as np
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
from dictionary.locations import X_TRAIN_DATA, Y_TRAIN_DATA, MODEL_DIR, TEST_MODEL_DIR, LOG_DIR

# allow gpu growth (solves error: Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR)
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

IMG_SIZE = 28
CATEGORIES = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # possible numbers on sudoku grid


def train():

    TEST_MODEL_NAME = "num_classifier_{}".format(int(time.time()))
    TEST_MODEL_PATH = TEST_MODEL_DIR / TEST_MODEL_NAME
    LOG_PATH = LOG_DIR / TEST_MODEL_NAME

    BATCH_SIZE = 16
    NUM_CLASSES = len(CATEGORIES)
    EPOCHS = 13
    SPLIT = 0.2  # validation split

    # similar to mnist model proposed by F. Chollet to deal with handwritten numbers
    # https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py

    tensorboard = TensorBoard(log_dir=LOG_PATH)

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

    # Optimizer
    adadelta = optimizers.Adadelta(learning_rate=1.0, rho=0.95)

    # sparse_categorical_crossentropy for multiple classes with labels [0], [1], [2], [3], [4], [5], [6], [7], [8]
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=adadelta,  # Adadelta or adam
                  metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=BATCH_SIZE,
              epochs=EPOCHS, validation_split=SPLIT, callbacks=[tensorboard])

    print("Saving model to " + str(TEST_MODEL_PATH) + ".h5")
    model.save(str(TEST_MODEL_PATH) + ".h5")  # save does not support pathlib paths


class NumberClassifier:

    MODEL_NAME = "num_classifier.h5"
    MODEL_PATH = MODEL_DIR / MODEL_NAME

    def __init__(self):
        # load_model does not support pathlib paths
        self.model = load_model(str(self.MODEL_PATH))

        # pre prediction to fire loading of cuda and cudnn
        self.model.predict(np.zeros((1, IMG_SIZE, IMG_SIZE, 1)))

    def __prepare(self, x_input_set):
        prepared_set = []

        for x_input in x_input_set:
            x_input = cv2.resize(x_input, (IMG_SIZE, IMG_SIZE))
            # normalize pixels
            x_input = x_input.astype('float32')
            x_input /= 255
            # reshape because keras expects this shape
            x_input = x_input.reshape(IMG_SIZE, IMG_SIZE, 1)
            prepared_set.append(x_input)

        return np.asarray(prepared_set)

    def predict(self, images):
        # returns predicted number and the related confidence

        predictions_list = []

        # predict digit for all images
        y_output_set = self.model.predict(self.__prepare(images))

        for y_output in y_output_set:
            predicted_class = np.argmax(y_output)
            predicted_digit = CATEGORIES[predicted_class]
            confidence = y_output[predicted_class]
            predictions_list.append((predicted_digit, confidence))

        return predictions_list
