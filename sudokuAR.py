import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt


def getHoughLines(image):

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

    # get hough transform
    lines = cv2.HoughLinesP(invert, 1, np.pi/180, 50, minLineLength=5)

    # create blank black image with only one channel
    hough_lines = np.zeros(invert.shape, dtype=np.uint8)

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

    # get all contours in given image
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


def getSudokuGrid(image, width, height):

    hough_lines = getHoughLines(image)
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
    warp = cv2.warpPerspective(image, perspective_trans, (width, height))

    return warp


def resize(image):
    # scale image
    return cv2.resize(image, None, fx=0.8, fy=0.8,
                      interpolation=cv2.INTER_CUBIC)


def main():
    # get webcam feed
    capture = cv2.VideoCapture('photo.jpg', 0)
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

        # show converted frame
        cv2.imshow('Perspective Transformed',
                   getSudokuGrid(frame, 400, 400))

        # wait 1 ms or quit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # clean up
    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
