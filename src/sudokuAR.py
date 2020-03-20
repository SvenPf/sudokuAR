import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt

DATADIR = "../data/"


def getHoughLines(image):

    # get hough transform
    lines = cv2.HoughLinesP(image, 1, np.pi/180, 50, minLineLength=5)

    # create blank black image with only one color channel
    hough_lines = np.zeros(image.shape, dtype=np.uint8)

    # check if any lines were found
    if lines is not None:
        # draw all found lines in source image
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # draw white line
            cv2.line(hough_lines, (x1, y1), (x2, y2), (255, 255, 255), 2)

    return hough_lines


def getMaxRectangle(image):

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

    return max_rectangle


def getSudokuGrid(image, height, width):

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
    # edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    hough_lines = getHoughLines(invert)
    max_rectangle = getMaxRectangle(hough_lines)
    # draw max rectangle
    cv2.imshow('Max Area Rectangle', cv2.polylines(
        image.copy(), [np.int32(max_rectangle)], True, (0, 255, 0), 2))

    # rectangle if you look direcly from above
    reference_rectangle = np.array(
        [[0, 0], [width, 0], [width, height], [0, height]], dtype="float32")

    # get inverse transformation of current prespectiv and apply it on given image
    perspective_trans = cv2.getPerspectiveTransform(
        max_rectangle, reference_rectangle)
    warp = cv2.warpPerspective(threshold, perspective_trans, (width, height))

    return warp


def getDigitImages(grid, cell_height, cell_width):

    digit_images = []

    center_x = int(cell_width / 2)
    center_y = int(cell_height / 2)

    threshold = 20
    scan_border_x = int(center_x * 0.65)
    scan_border_y = int(center_y * 0.65)

    for i in range(9):
        for j in range(9):
            # cut out cells
            pt_x = cell_width * j
            pt_y = cell_height * i
            cell_image = grid[pt_y:pt_y +
                              cell_height, pt_x:pt_x + cell_width]

            # check for emtpy cell and erase unwanted grid parts and center digit
            # start scan at the center of the cell
            # scan row to top/bottom with scan width of percentage of cell_width
            # scan column to left/right with scan width of percentage of cell_height
            # if sum of scan vector is less then threshold -> border found

            # scanned borders
            row_top = center_y - scan_border_y # top edge of scan border on y axis
            row_bottom = center_y + scan_border_y # bottom edge of scan border on y axis
            col_left = center_x - scan_border_x # left edge of scan border on x axis
            col_right = center_x + scan_border_x # right edge of scan border on x axis
            scan_image = cv2.bitwise_not(cell_image)

            top_b_found = False
            bottom_b_found = False
            left_b_found = False
            right_b_found = False

            for y in range(scan_border_y + 1):
                # scan to top
                if((not top_b_found) & (np.sum(scan_image[center_y - y, (center_x - scan_border_x):(center_x + scan_border_x)]) < threshold)):
                    row_top = center_y - y
                    top_b_found = True
                # scan to bottom
                if((not bottom_b_found) & (np.sum(scan_image[center_y + y, (center_x - scan_border_x):(center_x + scan_border_x)]) < threshold)):
                    row_bottom = center_y + y
                    bottom_b_found = True
                if(top_b_found & bottom_b_found):
                    break

            for x in range(scan_border_x + 1):
                # scan to left
                if((not left_b_found) & (np.sum(scan_image[(center_y - scan_border_y):(center_y + scan_border_y), center_x - x]) < threshold)):
                    col_left = center_x - x
                    left_b_found = True
                # scan to right
                if((not right_b_found) & (np.sum(scan_image[(center_y - scan_border_y):(center_y + scan_border_y), center_x + x]) < threshold)):
                    col_right = center_x + x
                    right_b_found = True
                if(left_b_found & right_b_found):
                    break

            # non empty cell
            if((row_top != center_y) & (row_bottom != center_y) & (col_left != center_x) & (col_right != center_x)):

                # get bounded image of digit
                digit_image = cell_image[row_top:row_bottom,
                                         col_left:col_right]

                # height and width of digit
                digit_width = int(col_right - col_left)
                digit_height = int(row_bottom - row_top)

                # calculate padding size left/right, top/bottom
                pad_lr = int((cell_width - digit_width) / 2)
                pad_tb = int((cell_height - digit_height) / 2)

                # int(...) may round down so we need to add the lost pixel
                pad_lr_correction = cell_width - (digit_width + pad_lr * 2)
                pad_tb_correction = cell_height - (digit_height + pad_tb * 2)

                # pad digit image
                padded_digit = cv2.copyMakeBorder(digit_image, pad_tb, pad_tb + pad_tb_correction,
                                                  pad_lr, pad_lr + pad_lr_correction, cv2.BORDER_CONSTANT, None, value=255)

                # add digit image and its location on the sudoku grid to the list
                digit_images.append((padded_digit, i, j))

                # DEBUG ----------------------------------------------
                # print(row_top, row_bottom, col_left, col_right)
                # cv2.imshow("input", scan_image)
                # cv2.waitKey(0)
                # print(padded_digit.shape)
                # cv2.imshow("scan", padded_digit)
                # cv2.waitKey(0)
                # ----------------------------------------------------

    # DEBUG ---------------------------------------
    from classifier.numberClassifier import predict

    for (digit_image, i, j) in digit_images:
        print(predict(digit_image))
        cv2.imshow("prediction", digit_image)
        cv2.waitKey(0)
    # ---------------------------------------------

    return digit_images


def resize(image, fx, fy):
    # scale image
    return cv2.resize(image, None, fx=fx, fy=fy,
                      interpolation=cv2.INTER_CUBIC)


def main():

    SUDOKU_GRID_HEIGHT = 450
    SUDOKU_GRID_WIDTH = 450

    # get webcam feed
    capture = cv2.VideoCapture(DATADIR + "photo.jpg", 0)
    # capture = cv2.VideoCapture(
    #     'https://test:test123@192.168.178.70:8080/video')

    # check if feed could be captured
    if not capture.isOpened():
        sys.exit(1)

    while True:
        # get frame of webcam feed
        ret, frame = capture.read()

        if ret == False:
            print('Problem with capture device')
            cv2.waitKey(0)  # delete for video feed
            sys.exit(1)

        # show webcam frame
        cv2.imshow('Webcam', frame)

        sudoku_grid = getSudokuGrid(
            frame, SUDOKU_GRID_HEIGHT, SUDOKU_GRID_WIDTH)

        # show converted frame
        cv2.imshow('Perspective Transformed', sudoku_grid)

        digit_images = getDigitImages(sudoku_grid, int(
            SUDOKU_GRID_HEIGHT / 9), int(SUDOKU_GRID_WIDTH / 9))

        # show all cell images
        # for i in range(9):
        #     for j in range(9):
        #         cv2.imshow(str(i) + "," + str(j), cell_images[i][j])
        #         cv2.resizeWindow(str(i) + "," + str(j), 200, 200)

        # wait 1 ms or quit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # clean up
    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
