import pickle
import numpy as np
import cv2
import time
from sudoku_ar.dictionary.locations import X_TRAIN_DATA, Y_TRAIN_DATA, MODEL_DIR, TEST_MODEL_DIR, LOG_DIR
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


IMG_SIZE = 28
CATEGORIES = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # possible numbers on sudoku grid


def train():

    TEST_MODEL_NAME = "num_classifier_{}.model".format(int(time.time()))
    TEST_MODEL_PATH = TEST_MODEL_DIR / TEST_MODEL_NAME
    LOG_PATH = LOG_DIR / TEST_MODEL_NAME

    BATCH_SIZE = 16
    NUM_CLASSES = len(CATEGORIES)
    EPOCHS = 8
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

    # sparse_categorical_crossentropy for multiple classes with labels [0], [1], [2], [3], [4], [5], [6], [7], [8]
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',  # Adadelta or adam
                  metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=BATCH_SIZE,
              epochs=EPOCHS, validation_split=SPLIT, callbacks=[tensorboard])

    print("Saving model in " + str(TEST_MODEL_PATH))
    model.save(str(TEST_MODEL_PATH))  # save does not support pathlib paths


class NumberClassifier:

    MODEL_NAME = "num_classifier.model"
    MODEL_PATH = MODEL_DIR / MODEL_NAME

    def __init__(self):
        # load_model does not support pathlib paths
        self.model = load_model(str(self.MODEL_PATH))
        # change config because of error with gpu
        config = ConfigProto()
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)

    def __prepare(self, x_input_set):

        prepared_set = []

        for x_input in x_input_set:
            x_input = cv2.resize(x_input, (IMG_SIZE, IMG_SIZE))
            # normalize pixels
            x_input = x_input.astype('float32')
            x_input /= 255
            # reshape because keras expects this shape
            # x_input = np.array(x_input)
            x_input = x_input.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
            prepared_set.append(x_input)

        return prepared_set

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
