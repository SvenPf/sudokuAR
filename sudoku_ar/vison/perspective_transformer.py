import cv2
import numpy as np


class PerspectiveTransformer:

    transform_matrix = None
    grid_image_shape = None
    goal_perspective = None

    def __init__(self, grid_image_height, grid_image_width):

        self.grid_image_shape = (grid_image_width, grid_image_height)

        # perspective if you look direcly from above
        self.goal_perspective = np.array([[0, 0], [grid_image_width, 0],
                                          [grid_image_width, grid_image_height], [0, grid_image_height]], dtype=np.float32)

    def transform_image_perspective(self, image, current_perspective):
        # get inverse transformation of current prespectiv and apply it on given image
        self.transform_matrix = cv2.getPerspectiveTransform(
            current_perspective, self.goal_perspective)

        return cv2.warpPerspective(image, self.transform_matrix, self.grid_image_shape)

    def inverse_transform_image_perspective(self, image, full_image_height, full_image_width):
        assert self.transform_matrix is not None
        return cv2.warpPerspective(image, self.transform_matrix, (full_image_width, full_image_height), flags=cv2.WARP_INVERSE_MAP)
