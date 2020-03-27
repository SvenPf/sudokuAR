import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt
from sudoku_ar.classifier.number_classifier import NumberClassifier
from sudoku_ar.solver.sudoku_solver import SudokuSolver


def getHoughLines(image):

    found = False
    # create blank black image with only one color channel
    hough_lines = np.zeros(image.shape, dtype=np.uint8)

    # get hough transform
    lines = cv2.HoughLinesP(image, 1, np.pi/180, 50, minLineLength=5)

    # check if any lines were found
    if lines is not None:
        found = True
        # draw all found lines in source image
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # draw white line
            cv2.line(hough_lines, (x1, y1), (x2, y2), (255, 255, 255), 2)

        # DEBUG ---------------
        # cv2.imshow("hough lines", hough_lines)
        # ---------------------

    return found, hough_lines


def getMaxRectangle(image):

    found = False
    max_rectangle = np.zeros((4, 2), dtype="float32")

    # get all contours in given image (contours must be white)
    contours, hierarchy = cv2.findContours(
        image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # atleast one contour must be found
    if len(contours) != 0:

        # find the biggest countour by area
        max_area_contour = max(contours, key=cv2.contourArea)

        # approximate polygon of given contour
        epsilon = 0.01 * cv2.arcLength(max_area_contour, True)
        poly_approx = cv2.approxPolyDP(max_area_contour, epsilon, True)

        # rectangle needs only 4 points
        if len(poly_approx) >= 4:
            found = True
            # convex hull for better approximation of rectangle
            hull = cv2.convexHull(poly_approx)

            # reshape for convenience
            points = hull.reshape(-1, 2)

            # the top-left point has the smallest sum
            # whereas the bottom-right has the largest sum
            sum_of = points.sum(axis=1)
            max_rectangle[0] = points[np.argmin(sum_of)]
            max_rectangle[2] = points[np.argmax(sum_of)]

            # compute the difference between the points -- the top-right
            # will have the minumum difference and the bottom-left will
            # have the maximum difference
            diff_of = np.diff(points, axis=1)
            max_rectangle[1] = points[np.argmin(diff_of)]
            max_rectangle[3] = points[np.argmax(diff_of)]

    return found, max_rectangle


def getSudokuGridImage(image, height, width):

    found = False
    warp = None
    perspective_trans = None

    # gray scale
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gaussian blur
    blur = cv2.GaussianBlur(grey, (5, 5), 0)
    # adaptive thresholding
    threshold = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 2)
    # invert black/white so that contours are white
    invert = cv2.bitwise_not(threshold)

    # maybe better than adaptive thresholding
    # edge = cv2.Canny(grey, 50, 150, apertureSize=3)

    found_hough, hough_lines = getHoughLines(invert)

    if found_hough:
        found_rec, max_rectangle = getMaxRectangle(hough_lines)

        if found_rec:
            found = True  # Sudoku grid was probably found

            # DEBUG : draw max rectangle -------------------
            # cv2.imshow('Max Area Rectangle', cv2.polylines(
            #     image.copy(), [np.int32(max_rectangle)], True, (0, 255, 0), 2))
            # ----------------------------------------------

            # rectangle if you look direcly from above
            reference_rectangle = np.array(
                [[0, 0], [width, 0], [width, height], [0, height]], dtype="float32")

            # get inverse transformation of current prespectiv and apply it on given image
            perspective_trans = cv2.getPerspectiveTransform(
                max_rectangle, reference_rectangle)
            warp = cv2.warpPerspective(
                threshold, perspective_trans, (width, height))

    return found, warp, perspective_trans


# TODO maybe centering then floodfill from sides is better ?!
def cutOutDigit(cell_image, center_x, center_y):
    # expects thresholded cell_image
    # check for emtpy cell and erase unwanted grid parts and center digit
    # start scan at the center of the cell
    # scan row to top/bottom with scan width of percentage of cell_width
    # scan column to left/right with scan width of percentage of cell_height
    # if sum of scan vector is less then threshold -> border found

    cut_digit = []

    threshold = 150
    scan_border_x = int(center_x * 0.65)
    scan_border_y = int(center_y * 0.65)

    # scanning needs white digit on black background (for sum)
    scan_image = cv2.bitwise_not(cell_image)

    # scanned borders
    row_top = center_y - scan_border_y  # top edge of scan border on y axis
    row_bottom = center_y + scan_border_y  # bottom edge of scan border on y axis
    col_left = center_x - scan_border_x  # left edge of scan border on x axis
    col_right = center_x + scan_border_x  # right edge of scan border on x axis

    top_b_found = False
    bottom_b_found = False
    left_b_found = False
    right_b_found = False

    for y in range(scan_border_y + 1):
        # scan to top
        if (not top_b_found) & (np.sum(scan_image[center_y - y, (center_x - scan_border_x):(center_x + scan_border_x)]) < threshold):
            row_top = center_y - y
            top_b_found = True
        # scan to bottom
        if (not bottom_b_found) & (np.sum(scan_image[center_y + y, (center_x - scan_border_x):(center_x + scan_border_x)]) < threshold):
            row_bottom = center_y + y
            bottom_b_found = True
        if(top_b_found & bottom_b_found):
            break

    for x in range(scan_border_x + 1):
        # scan to left
        if (not left_b_found) & (np.sum(scan_image[(center_y - scan_border_y):(center_y + scan_border_y), center_x - x]) < threshold):
            col_left = center_x - x
            left_b_found = True
        # scan to right
        if (not right_b_found) & (np.sum(scan_image[(center_y - scan_border_y):(center_y + scan_border_y), center_x + x]) < threshold):
            col_right = center_x + x
            right_b_found = True
        if left_b_found & right_b_found:
            break

    # non empty cell
    if (row_top != center_y) & (row_bottom != center_y) & (col_left != center_x) & (col_right != center_x):
        # get bounded image of digit
        cut_digit = cell_image[row_top:row_bottom, col_left:col_right]

        # DEBUG : visualizes scanning grid -------------------
        # test_image = cv2.rectangle(cell_image.copy(), (col_left, row_top), (col_right, row_bottom), (0, 0, 0), 1)
        # cv2.line(test_image, (center_x, center_y - scan_border_y), (center_x, center_y + scan_border_y), (0, 0, 0), 1)
        # cv2.line(test_image, (center_x - scan_border_x, center_y), (center_x + scan_border_x, center_y), (0, 0, 0), 1)
        # cv2.imshow("input", cv2.resize(test_image, (test_image.shape[1] * 3, test_image.shape[0] * 3)))
        # cv2.waitKey(0)
        # ----------------------------------------------------

    return cut_digit


def padDigitImage(digit_image, digit_height, digit_width):

    padded_digit = []

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


def getDigitImages(grid, cell_height, cell_width):

    digit_images = []

    center_x = int(cell_width / 2)
    center_y = int(cell_height / 2)

    for i in range(9):
        for j in range(9):
            # TODO check if size of cells must be adjusted (center of lower cells get distorted)
            # cut out cells
            pt_x = cell_width * j
            pt_y = cell_height * i
            cell_image = grid[pt_y:pt_y +
                              cell_height, pt_x:pt_x + cell_width]

            digit_image = cutOutDigit(cell_image, center_x, center_y)

            # check if cell wasn't empty
            if len(digit_image) > 0:
                # height and width of digit
                digit_height, digit_width = digit_image.shape

                padded_digit = padDigitImage(
                    digit_image, digit_height, digit_width)

                full_digit = cv2.resize(
                    padded_digit, (cell_width, cell_height))

                # DEBUG ---------------------------
                # cv2.imshow("full digit", full_digit)
                # cv2.waitKey(0)
                # ---------------------------------

                # add digit image and its location on the sudoku grid to the list
                digit_images.append((full_digit, i, j))

    return digit_images


def getSudokuGridAsArray(num_classifier, digit_images, sudoku_shape):

    assert(len(digit_images) > 0)

    # empty sudoku grid
    sudoku_array = np.zeros(sudoku_shape, np.uint8)

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

    for (digit_image, i, j) in digit_images:
        predicted_digit = num_classifier.predict([digit_image])[0][0]
        sudoku_array[i, j] = predicted_digit

    # DEBUG :2 show all detected digit and their prediction -----------------------
    #     resize = cv2.resize(digit_image, (text_height - 2, text_height - 2))
    #     y_offset = y_pos - int(text_height / 2)
    #     win[y_offset + 1:y_offset + text_height - 1, 1:text_height - 1] = resize
    #     predictions = num_classifier.predict([digit_image])
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


def convert_to_image(sudoku_grid_array, height, width, grid_shape):

    cell_height = int(height / grid_shape[0])
    cell_width = int(width / grid_shape[1])
    padding_x = int(cell_width * 0.3)
    padding_y = int(cell_height * 0.3)

    image = np.zeros((height, width, 3), np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_color = (0, 255, 0)

    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            digit = sudoku_grid_array[i][j]
            if digit > 0:
                y_start = cell_height * (i + 1) - padding_y
                x_start = cell_width * j + padding_x
                cv2.putText(image, str(digit), (x_start, y_start),
                            font, font_scale, font_color, 2)

    return image


def run(capture_device):

    SUDOKU_GRID_HEIGHT = 450
    SUDOKU_GRID_WIDTH = 450
    SUDOKU_SHAPE = (9, 9)

    num_classifier = NumberClassifier()
    sudoku_solver = SudokuSolver()

    # get webcam feed
    capture = cv2.VideoCapture(capture_device)

    # check if feed could be captured
    if not capture.isOpened():
        sys.exit(1)

    while True:
        # TODO later probably needs parallelization

        # wait 1 ms or quit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # get frame of webcam feed
        ret, frame = capture.read()

        if ret == False:
            print("Problem with capture device (press any key to close)")
            cv2.waitKey(0)
            sys.exit(1)

        # show webcam frame
        cv2.imshow("Webcam", frame)

        found_grid, sudoku_grid_image, transform = getSudokuGridImage(
            frame, SUDOKU_GRID_HEIGHT, SUDOKU_GRID_WIDTH)

        if not found_grid:
            continue

        # show converted frame
        cv2.imshow("Perspective Transformed", sudoku_grid_image)

        digit_images = getDigitImages(sudoku_grid_image, int(
            SUDOKU_GRID_HEIGHT / SUDOKU_SHAPE[0]), int(SUDOKU_GRID_WIDTH / SUDOKU_SHAPE[1]))

        if(len(digit_images) > 0):
            sudoku_grid_array = getSudokuGridAsArray(
                num_classifier, digit_images, SUDOKU_SHAPE)

            solved_sudoku = sudoku_solver.solve_array(sudoku_grid_array)

            if len(solved_sudoku) > 0:
                solved_sudoku_image = convert_to_image(
                    solved_sudoku - sudoku_grid_array, SUDOKU_GRID_HEIGHT, SUDOKU_GRID_HEIGHT, SUDOKU_SHAPE)

                height, width, _ = frame.shape
                wraped_solved_sudoku_image = cv2.warpPerspective(
                    solved_sudoku_image, transform, (width, height), flags=cv2.WARP_INVERSE_MAP)

                cv2.imshow("Solution", cv2.addWeighted(
                    frame, 0.8, wraped_solved_sudoku_image, 0.5, 0.0))

        # TODO check if new Sudoku grid was found, otherwise show old sudoku solution (only calculate it once!)

    # clean up
    capture.release()
    cv2.destroyAllWindows()
