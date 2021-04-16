import random

import cv2
import numpy as np

class Image:
    """To instantiate this class, you need to provide two params.
    'path' param is necessary, it is the path of the image.
    'noise' param is optional. If you want to add noise to the image, you could input for example, 'sp' or 'gaussian'.
    You could use get_gray_image method to get original gray image.
    """
    def __init__(self, path, noise=None):
        self.gray_image = cv2.imread(path, 0)
        if noise is not None:
            self.__add_noise(noise)

    def __add_noise(self, noise):
        if noise == 'sp':
            self.__add_sp_noise()
        elif noise == 'gaussian':
            self.__add_gaussian_noise()

    def __add_sp_noise(self, prob=0.01):
        thres = 1 - prob
        for i in range(self.gray_image.shape[0]):
            for j in range(self.gray_image.shape[1]):
                rdn = random.random()
                if rdn < prob:
                    self.gray_image[i][j] = 0
                elif rdn > thres:
                    self.gray_image[i][j] = 255

    def __add_gaussian_noise(self, sigma=5):
        hight, width = self.gray_image.shape[0], self.gray_image.shape[1]
        self.gray_image += np.uint8(np.random.randn(hight, width) * sigma)

    def get_gray_image(self):
        return self.gray_image