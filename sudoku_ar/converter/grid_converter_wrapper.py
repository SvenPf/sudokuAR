import cv2
import os
import numpy as np
from multiprocessing import Process, Value, Lock, shared_memory
from converter.grid_converter_process import GridConverterProcess
from helper.shared_numpy import SharedNumpy


class GridConverterWrapper:

    def __init__(self, sudoku_shape, grid_image_height, grid_image_width):

        self.sudoku_shape = sudoku_shape
        self.grid_image_height = grid_image_height
        self.grid_image_width = grid_image_width
        self.cell_image_height = int(grid_image_height / sudoku_shape[0])
        self.cell_image_width = int(grid_image_width / sudoku_shape[1])
        self.cell_image_center_x = int(self.cell_image_width / 2)
        self.cell_image_center_y = int(self.cell_image_height / 2)

        self.shared_grid_image = SharedNumpy(
            (grid_image_height, grid_image_width), np.uint8, sig_on_write=True)
        self.shared_grid_array = SharedNumpy(sudoku_shape, np.uint8)

        self.process = GridConverterProcess(
            sudoku_shape, grid_image_height, grid_image_width, self.shared_grid_image, self.shared_grid_array)
        self.process.daemon = True  # terminates all child processes on parent exit
        self.process.start()

    def __del__(self):
        print("delete Converter", os.getpid())

        if self.process is not None:
            self.process.terminate()
            self.process.join()

    def convert_image_to_array(self, grid_image):
        """
        Expects preprocessed binary image of sudoku grid
        """

        # write next image to convert
        self.shared_grid_image.write(grid_image)

        # get current converted image
        grid_array = self.shared_grid_array.read()

        # print(grid_array)

        return grid_array if np.any(grid_array) else []

    def convert_array_to_image(self, grid_array):

        padding_x = int(self.cell_image_width * 0.3)
        padding_y = int(self.cell_image_height * 0.3)

        image = np.zeros(
            (self.grid_image_height, self.grid_image_width, 3), np.uint8)

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
