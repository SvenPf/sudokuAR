from pathlib import Path

SRC_DIR = Path(__file__).parent.parent.absolute()

# General paths
RESOURCES_DIR = SRC_DIR / "resources/"
CLASSIFIER_DIR = SRC_DIR / "classifier/"

# Resource folder
TRAIN_DATA_DIR = RESOURCES_DIR / "train_data/"
RAW_SAMPLES_DIR = TRAIN_DATA_DIR / "raw/"

# Training/Generating
X_TRAIN_DATA = TRAIN_DATA_DIR / "x_train.pickle"
Y_TRAIN_DATA = TRAIN_DATA_DIR / "y_train.pickle"

# Training
MODEL_DIR = CLASSIFIER_DIR / "model/"
TEST_MODEL_DIR = MODEL_DIR / "test/"
LOG_DIR = CLASSIFIER_DIR / "logs/"

# Generating
SAMPLES_TAR = RAW_SAMPLES_DIR / "samples.tar"
