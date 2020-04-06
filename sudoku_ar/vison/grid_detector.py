import cv2
import numpy as np


class GridDetecor:

    def __init__(self):
        pass

    def __get_max_rectangle(self, image):

        found = False
        max_rectangle = np.zeros((4, 2), dtype="float32")

        # get all contours in given image (contours must be white)
        contours, _ = cv2.findContours(
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

        return max_rectangle if found else None

    def __get_hough_lines(self, image):

        found = False
        # create blank black image with only one color channel
        hough_lines = np.zeros(image.shape, dtype=np.uint8)

        # get hough transform
        lines = cv2.HoughLinesP(image, 1, np.pi/180, 50, minLineLength=5)

        # check if any lines were found
        if len(lines) != 0:
            found = True
            # draw all found lines in source image
            for line in lines:
                x_1, y_1, x_2, y_2 = line[0]
                # draw white line
                cv2.line(hough_lines, (x_1, y_1), (x_2, y_2), (255, 255, 255), 2)

            # DEBUG ---------------
            # cv2.imshow("hough lines", hough_lines)
            # ---------------------

        return hough_lines if found else None

    def get_grid_location(self, image):

        grid_location = None

        # TODO wohin mit preprocessing <-> perspective_transformer
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

        hough_lines = self.__get_hough_lines(invert)

        if hough_lines is not None:
            grid_location = self.__get_max_rectangle(hough_lines)

            # DEBUG : draw max rectangle -------------------
            # cv2.imshow('Max Area Rectangle', cv2.polylines(
            #     image.copy(), [np.int32(max_rectangle)], True, (0, 255, 0), 2))
            # ----------------------------------------------

        return grid_location
