import cv2
import numpy as np
from multiprocessing import Process
from time import sleep
from classifier.number_classifier import NumberClassifier

UPDATE_RATE = 1  # in seconds


class GridConverterProcess(Process):

    def __init__(self, sudoku_shape, grid_image_height, grid_image_width, shared_grid_image, shared_grid_array):

        Process.__init__(self)

        self.sudoku_shape = sudoku_shape

        self.grid_image_height = grid_image_height
        self.grid_image_width = grid_image_width
        self.cell_image_height = int(grid_image_height / sudoku_shape[0])
        self.cell_image_width = int(grid_image_width / sudoku_shape[1])
        self.cell_image_center_x = int(self.cell_image_width / 2)
        self.cell_image_center_y = int(self.cell_image_height / 2)

        self.shared_grid_image = shared_grid_image
        self.shared_grid_array = shared_grid_array

    def __pad_digit_image(self, digit_image):

        padded_digit = []

        # height and width of digit
        digit_height, digit_width = digit_image.shape

        extra_pad = 2
        pad_lr = 0
        pad_tb = 0
        pad_lr_corr = 0  # correction padding
        pad_tb_corr = 0  # correction padding

        # calculate padding size left/right, top/bottom
        if digit_width > digit_height:
            pad_tb = int((digit_width - digit_height) / 2)
            # int(...) rounds down so we may miss a pixel
            pad_tb_corr = digit_width - (digit_height + pad_tb * 2)
        elif digit_height > digit_width:
            pad_lr = int((digit_height - digit_width) / 2)
            # int(...) rounds down so we may miss a pixel
            pad_lr_corr = digit_height - (digit_width + pad_lr * 2)

        # pad digit image
        padded_digit = cv2.copyMakeBorder(digit_image, pad_tb + extra_pad, pad_tb + pad_tb_corr + extra_pad,
                                          pad_lr + extra_pad, pad_lr + pad_lr_corr + extra_pad, cv2.BORDER_CONSTANT, value=255)

        # DEBUG ---------------------------
        # cv2.imshow("padding", padded_digit)
        # cv2.waitKey(0)
        # ---------------------------------
        return padded_digit

    # TODO maybe centering then floodfill from sides is better ?!
    def __cut_out_digit(self, cell_image):
        # expects thresholded cell_image
        # check for emtpy cell and erase unwanted grid parts and center digit
        # start scan at the center of the cell
        # scan row to top/bottom with scan width of percentage of cell_width
        # scan column to left/right with scan width of percentage of cell_height
        # if sum of scan vector is less then threshold -> border found

        cut_digit = []

        threshold = 150
        scan_border_x = int(self.cell_image_center_x * 0.65)
        scan_border_y = int(self.cell_image_center_y * 0.65)

        # scanning needs white digit on black background (for sum)
        scan_image = cv2.bitwise_not(cell_image)

        # scanned borders
        # top edge of scan border on y axis
        row_top = self.cell_image_center_y - scan_border_y
        # bottom edge of scan border on y axis
        row_bottom = self.cell_image_center_y + scan_border_y
        # left edge of scan border on x axis
        col_left = self.cell_image_center_x - scan_border_x
        # right edge of scan border on x axis
        col_right = self.cell_image_center_x + scan_border_x

        top_b_found = False
        bottom_b_found = False
        left_b_found = False
        right_b_found = False

        for y in range(scan_border_y + 1):
            # scan to top
            if (not top_b_found) & (np.sum(scan_image[self.cell_image_center_y - y, (self.cell_image_center_x - scan_border_x):(self.cell_image_center_x + scan_border_x)]) < threshold):
                row_top = self.cell_image_center_y - y
                top_b_found = True
            # scan to bottom
            if (not bottom_b_found) & (np.sum(scan_image[self.cell_image_center_y + y, (self.cell_image_center_x - scan_border_x):(self.cell_image_center_x + scan_border_x)]) < threshold):
                row_bottom = self.cell_image_center_y + y
                bottom_b_found = True
            if top_b_found & bottom_b_found:
                break

        for x in range(scan_border_x + 1):
            # scan to left
            if (not left_b_found) & (np.sum(scan_image[(self.cell_image_center_y - scan_border_y):(self.cell_image_center_y + scan_border_y), self.cell_image_center_x - x]) < threshold):
                col_left = self.cell_image_center_x - x
                left_b_found = True
            # scan to right
            if (not right_b_found) & (np.sum(scan_image[(self.cell_image_center_y - scan_border_y):(self.cell_image_center_y + scan_border_y), self.cell_image_center_x + x]) < threshold):
                col_right = self.cell_image_center_x + x
                right_b_found = True
            if left_b_found & right_b_found:
                break

        # TODO check when right_b was not found if digit was shifted to left

        # non empty cell
        if (row_top != self.cell_image_center_y) & (row_bottom != self.cell_image_center_y) & (col_left != self.cell_image_center_x) & (col_right != self.cell_image_center_x):
            # get bounded image of digit
            cut_digit = cell_image[row_top:row_bottom, col_left:col_right]

            # DEBUG : visualizes scanning grid -------------------
            # test_image = cv2.rectangle(cell_image.copy(), (col_left, row_top), (col_right, row_bottom), (0, 0, 0), 1)
            # cv2.line(test_image, (self.cell_image_center_x, self.cell_image_center_y - scan_border_y), (self.cell_image_center_x, self.cell_image_center_y + scan_border_y), (0, 0, 0), 1)
            # cv2.line(test_image, (self.cell_image_center_x - scan_border_x, self.cell_image_center_y), (self.cell_image_center_x + scan_border_x, self.cell_image_center_y), (0, 0, 0), 1)
            # cv2.imshow("input", cv2.resize(test_image, (test_image.shape[1] * 3, test_image.shape[0] * 3)))
            # cv2.waitKey(0)
            # ----------------------------------------------------

        return cut_digit

    def __get_digit_images(self, grid_image):
        # TODO parallelisieren

        digit_images = []

        for i in range(self.sudoku_shape[0]):
            for j in range(self.sudoku_shape[1]):
                # TODO check if size of cells must be adjusted (center of lower cells get distorted)
                # cut out cells
                pt_x = self.cell_image_width * j
                pt_y = self.cell_image_height * i
                cell_image = grid_image[pt_y:pt_y +
                                        self.cell_image_height, pt_x:pt_x + self.cell_image_width]

                digit_image = self.__cut_out_digit(cell_image)

                # check if cell wasn't empty
                if len(digit_image) > 0:

                    padded_digit = self.__pad_digit_image(digit_image)

                    full_digit = cv2.resize(
                        padded_digit, (self.cell_image_width, self.cell_image_height))

                    # DEBUG ---------------------------
                    # cv2.imshow("full digit", full_digit)
                    # cv2.waitKey(0)
                    # ---------------------------------

                    # add digit image and its location on the sudoku grid to the list
                    digit_images.append((full_digit, i, j))

        return digit_images

    def __get_as_array(self, classifier, grid_digit_images):

        assert len(grid_digit_images) > 0

        # empty sudoku grid array
        sudoku_array = np.zeros(self.sudoku_shape, np.uint8)

        # DEBUG :1 show all detected digit and their prediction --
        # text_height = 30
        # w_height = len(grid_digit_images) * text_height
        # w_width = 210
        # y_pos = int(text_height / 2)
        # win = np.zeros((w_height, w_width), np.uint8)

        # font = cv2.FONT_HERSHEY_SIMPLEX
        # font_scale = 0.45
        # font_color = (255, 255, 255)
        # --------------------------------------------------------

        # create Batch of digit image
        digit_images_batch = [grid_digit_image[0]
                              for grid_digit_image in grid_digit_images]

        predictions = classifier.predict(digit_images_batch)

        predications_iterator = iter(predictions)
        for (_, i, j) in grid_digit_images:
            # get predicted digit
            sudoku_array[i, j] = next(predications_iterator)[0]

        # DEBUG :2 show all detected digit and their prediction -----------------------
        #     resize = cv2.resize(grid_digit_image, (text_height - 2, text_height - 2))
        #     y_offset = y_pos - int(text_height / 2)
        #     win[y_offset + 1:y_offset + text_height - 1, 1:text_height - 1] = resize
        #     text = "(" + str(i) + ", " + str(j) + ") " + \
        #         str(predictions[0][0]) + " (" + "{0:.1%}".format(predictions[0][1]) + ")"
        #     cv2.putText(win, text, (text_height + 5, y_pos + 2),
        #                 font, font_scale, font_color)
        #     y_pos += text_height

        # cv2.imshow("prediction", win)
        # ------------------------------------------------------------------------------

        # DEBUG : show sudoku array --
        # print(sudoku_array)
        # ----------------------------

        return sudoku_array


    def run(self):

        classifier = NumberClassifier()

        while True:
            sleep(UPDATE_RATE)

            if not self.shared_grid_image.receive_signal:
                continue

            grid_image = self.shared_grid_image.read()

            digit_images = self.__get_digit_images(grid_image)

            if len(digit_images) > 0:
                grid_array = self.__get_as_array(classifier, digit_images)

                print(grid_array)
                self.shared_grid_array.write(grid_array)
