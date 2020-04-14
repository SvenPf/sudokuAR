import cv2
import numpy as np
from vison.grid_detector import GridDetecor
from vison.perspective_transformer import PerspectiveTransformer
from converter.grid_converter import GridConverter
from classifier.number_classifier import NumberClassifier
from solver.sudoku_solver import SudokuSolver
from helper.umat_video_stream import UMatVideoStream


class App:

    def __init__(self, capture_device):

        SUDOKU_GRID_HEIGHT = 450
        SUDOKU_GRID_WIDTH = 450
        SUDOKU_SHAPE = (9, 9)
        SELECTION_RATE = 128

        self.grid_detector = GridDetecor()
        self.perspective_transformer = PerspectiveTransformer(
            SUDOKU_GRID_HEIGHT, SUDOKU_GRID_WIDTH)
        self.umat_video_stream = UMatVideoStream(
            capture_device, SELECTION_RATE)
        self.num_classifier = NumberClassifier()
        self.grid_converter = GridConverter(
            self.num_classifier, SUDOKU_SHAPE, SUDOKU_GRID_HEIGHT, SUDOKU_GRID_WIDTH)
        self.sudoku_solver = SudokuSolver()

    def __preprocess(self, image):
        # gray scale
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # gaussian blur
        blur = cv2.GaussianBlur(grey, (5, 5), 0)
        # adaptive thresholding
        threshold = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 2)
        # TODO noice canceling

        return threshold

    def run(self):

        frame = cv2.UMat(self.umat_video_stream.height, self.umat_video_stream.width, cv2.CV_8UC3)
        solved_sudoku_image = None
        old_sudoku_grid_array = []
        # get webcam feed
        video = self.umat_video_stream.start()

        while not video.stopped:
            # TODO later probably needs parallelization
            # maybe pipe lining is better

            # wait 1 ms or quit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                video.stop()
                break

            # get frame of webcam feed
            frame = video.read()

            if frame is None:
                break

            # show webcam frame
            cv2.imshow("Webcam", frame)

            # preprocess frame
            preprocessed_frame = self.__preprocess(frame)
            grid_location = self.grid_detector.get_grid_location(preprocessed_frame, video.height, video.width)

            if grid_location is None:
                solved_sudoku_image = None
                continue

            sudoku_grid_image = self.perspective_transformer.transform_image_perspective(
                preprocessed_frame, grid_location)

            # show converted frame
            cv2.imshow("Perspective Transformed", sudoku_grid_image)

            # TODO convert to UMat
            sudoku_grid_array = self.grid_converter.convert_image_to_array(
                sudoku_grid_image.get())

            if len(sudoku_grid_array) > 0:

                if not np.array_equal(sudoku_grid_array, old_sudoku_grid_array):

                    solved_sudoku_array = self.sudoku_solver.solve_array(
                        sudoku_grid_array)

                    if len(solved_sudoku_array) > 0:

                        solved_sudoku_image = self.grid_converter.convert_array_to_image(
                            solved_sudoku_array - sudoku_grid_array)

                    old_sudoku_grid_array = sudoku_grid_array

            if solved_sudoku_image is not None:

                wraped_solved_sudoku_image = self.perspective_transformer.inverse_transform_image_perspective(
                    solved_sudoku_image, video.height, video.width)

                cv2.imshow("Solution", cv2.addWeighted(
                    frame, 0.8, wraped_solved_sudoku_image, 0.5, 0.0))

            # TODO check if new Sudoku grid was found, otherwise show old sudoku solution (only calculate it once!)

        print("Video stream was stopped (Press any Key to close)")
        cv2.waitKey(0)
        # clean up
        cv2.destroyAllWindows()
