import cv2
import numpy as np


class GridConverter:

    def __init__(self, classifier, sudoku_shape, grid_image_height, grid_image_width):
        self.classifier = classifier
        self.sudoku_shape = sudoku_shape

        self.grid_image_height = grid_image_height
        self.grid_image_width = grid_image_width
        self.cell_image_height = int(grid_image_height / sudoku_shape[0])
        self.cell_image_width = int(grid_image_width / sudoku_shape[1])
        self.cell_image_center_x = int(self.cell_image_width / 2)
        self.cell_image_center_y = int(self.cell_image_height / 2)

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
        row_top = self.cell_image_center_y - scan_border_y  # top edge of scan border on y axis
        row_bottom = self.cell_image_center_y + scan_border_y  # bottom edge of scan border on y axis
        col_left = self.cell_image_center_x - scan_border_x  # left edge of scan border on x axis
        col_right = self.cell_image_center_x + scan_border_x  # right edge of scan border on x axis

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
            if(top_b_found & bottom_b_found):
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

    def __get_as_array(self, grid_digit_images):

        assert(len(grid_digit_images) > 0)

        # empty sudoku grid
        sudoku_array = np.zeros(self.sudoku_shape, np.uint8)

        # DEBUG :1 show all detected digit and their prediction --
        # text_height = 30
        # w_height = len(digit_images) * text_height
        # w_width = 210
        # y_pos = int(text_height / 2)
        # win = np.zeros((w_height, w_width), np.uint8)

        # font = cv2.FONT_HERSHEY_SIMPLEX
        # font_scale = 0.45
        # font_color = (255, 255, 255)
        # --------------------------------------------------------

        digit_batch = []

        for (grid_digit_image, i, j) in grid_digit_images:
            digit_batch.append(grid_digit_image)

        predictions = self.classifier.predict(digit_batch)
        # print(predictions)

        # TODO predicting digits is bottleneck !!!! -> maybe predict digits as batch
        index = 0
        for (grid_digit_image, i, j) in grid_digit_images:
            sudoku_array[i, j] = predictions[index][0] # get predicted digit
            index += 1

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

    def convert_image_to_array(self, grid_image):
        """
        Expects preprocessed binary image of sudoku grid
        """

        grid_array = []

        digit_images = self.__get_digit_images(grid_image)

        if(len(digit_images) > 0):
            grid_array = self.__get_as_array(digit_images)

        return grid_array

    def convert_array_to_image(self, grid_array):

        padding_x = int(self.cell_image_width * 0.3)
        padding_y = int(self.cell_image_height * 0.3)

        image = np.zeros((self.grid_image_height, self.grid_image_width, 3), np.uint8)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_color = (0, 255, 0)

        for i in range(self.sudoku_shape[0]):
            for j in range(self.sudoku_shape[1]):
                digit = grid_array[i][j]
                if digit > 0:
                    y_start = self.cell_image_height * (i + 1) - padding_y
                    x_start = self.cell_image_width * j + padding_x
                    cv2.putText(image, str(digit), (x_start, y_start),
                                font, font_scale, font_color, 2)

        return image
