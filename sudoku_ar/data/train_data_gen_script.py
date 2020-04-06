import tarfile
import numpy as np
import os
import cv2
import pickle
import random
from sudoku_ar.dictionary.locations import X_TRAIN_DATA, Y_TRAIN_DATA, SAMPLES_TAR

CATEGORIES = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # possible numbers on sudoku grid
SAMPLENAMES = ["Sample0" + str("%.2d" % (i+1))
               for i in CATEGORIES]  # names of directories

IMG_SIZE = 28

def createTrainingData():

    training_data = []

    # can't handle pathlib paths
    tar = tarfile.open(str(SAMPLES_TAR), "r:")

    # iterator for all members (directories, files) in tar
    iter_tar = iter(tar)
    member = next(iter_tar)  # get first member

    # give every sample its matching label
    for sample_name in SAMPLENAMES:

        # class 0 -> number 1, class 1 -> number 2, ...
        label = SAMPLENAMES.index(sample_name)

        print("Creating training set for " +
              sample_name + " (Class " + str(label) + ")")

        while member:
            # check if member is a file and matching to current label
            if member.isfile() & (sample_name in os.path.dirname(member.name)):
                # extract member (image) from tar
                image = tar.extractfile(member).read()
                # load image in numpy array
                image_np = np.frombuffer(image, dtype=np.uint8)
                # convert to opencv graysacle image
                image_array = cv2.imdecode(image_np, cv2.IMREAD_GRAYSCALE)
                # resize
                resize = cv2.resize(image_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([resize, label])

                member = next(iter_tar, None)  # next member

            elif member.isdir():
                member = next(iter_tar, None)  # next member
            else:
                # first sample not matching current label -> save it for next label (don't skip it!)
                break

    tar.close()

    return training_data


def main():
    training_data = createTrainingData()
    print("Shuffle training data")
    random.shuffle(training_data)  # shuffle so that network dosen't overfit

    x_train = []  # input set
    y_train = []  # output set

    for image, label in training_data:
        x_train.append(image)
        y_train.append(label)

    # reshap because tensorflow expects this shape
    x_train = np.array(x_train).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    # save inputs
    print("Saving set of inputs in " + str(X_TRAIN_DATA))
    pickle_out = open(X_TRAIN_DATA, "wb")
    pickle.dump(x_train, pickle_out)
    pickle_out.close()

    # save outputs
    print("Saving set of labels in " + str(Y_TRAIN_DATA))
    pickle_out = open(Y_TRAIN_DATA, "wb")
    pickle.dump(y_train, pickle_out)
    pickle_out.close()


if __name__ == "__main__":
    main()
