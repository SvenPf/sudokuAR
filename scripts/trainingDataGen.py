import tarfile
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pickle
import random

DATADIR = "../data/"
TARPATH = DATADIR + "samples.tar"

CATEGORIES = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # possible numbers on sudoku grid
SAMPLENAMES = ["Sample0" + str("%.2d" % (i+1))
               for i in CATEGORIES]  # names of directories

IMG_SIZE = 50


def createTrainingData():

    training_data = []

    tar = tarfile.open(TARPATH, "r:")

    # iterator for all members (directories, files) in tar
    iter_tar = iter(tar)
    member = next(iter_tar)  # get first member

    # give every sample its matching label
    for sample_name in SAMPLENAMES:

        label = SAMPLENAMES.index(sample_name) + 1  # class

        print("Creating training set for " +
              sample_name + " (Class " + str(label) + ")")

        while member:
            # check if member is file and matching to current label
            if member.isfile() & (sample_name in os.path.dirname(member.name)):
                # extract member (image) from tar
                image = tar.extractfile(member).read()
                # load image in numpy array
                image_np = np.frombuffer(image, dtype=np.uint8)
                # convert to opencv graysacle image
                image_array = cv2.imdecode(image_np, cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(
                    image_array, (IMG_SIZE, IMG_SIZE))  # resize image
                # save pair of image and its label
                training_data.append([new_array, label])

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

    train_x = []  # input set
    train_y = []  # output set

    for image, label in training_data:
        train_x.append(image)
        train_y.append(label)

    # reshap because tensorflow expects this shape
    train_x = np.array(train_x).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    # save inputs
    print("Saving set of inputs in " +
          os.path.abspath(DATADIR + "train_x.pickle"))
    pickle_out = open(DATADIR + "train_x.pickle", "wb")
    pickle.dump(train_x, pickle_out)
    pickle_out.close()

    # save outputs
    print("Saving set of labels in " +
          os.path.abspath(DATADIR + "train_y.pickle"))
    pickle_out = open(DATADIR + "train_y.pickle", "wb")
    pickle.dump(train_y, pickle_out)
    pickle_out.close()


if __name__ == "__main__":
    main()
