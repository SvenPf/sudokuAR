import os

FILE_DIR = os.path.dirname(os.path.abspath(__file__)) + "/"

# General paths
RESOURCES_DIR = os.path.abspath(FILE_DIR + "../resources/") + "/"
CLASSIFIER_DIR = os.path.abspath(FILE_DIR + "../classifier/") + "/"

# Resource folder
TRAIN_DATA_DIR = RESOURCES_DIR + "train_data/"
RAW_SAMPLES_DIR = TRAIN_DATA_DIR + "raw/"

# Training/Generating
X_TRAIN_DATA = TRAIN_DATA_DIR + "x_train.pickle"
Y_TRAIN_DATA = TRAIN_DATA_DIR + "y_train.pickle"

# Training
MODEL_DIR = CLASSIFIER_DIR + "model/"

# Generating
SAMPLES_TAR = RAW_SAMPLES_DIR + "samples.tar"
