import os

import cv2
import numpy as np
import matplotlib.pyplot as plt

from image import Image

class Detection:
    """Many edge detection mothods are biult into this class.
    You could use method like use_roberts to use Robert edge detection algorithm to a image(Image object).
    You could use draw method to intuitively look the results of all the edge detection algorithm.
    """
    def __init__(self):
        pass

    def use_all_operator(self, image):
        det_images = []
        det_images.append(self.use_roberts(image))
        det_images.append(self.use_sobel(image))
        det_images.append(self.use_prewitt(image))
        det_images.append(self.use_laplacian(image))
        det_images.append(self.use_kirsch(image))
        return det_images

    def use_roberts(self, image):
        kernel_x, kernel_y = self.__get_roberts_kernels()
        x = cv2.filter2D(image.gray_image, cv2.CV_16S, kernel_x)
        y = cv2.filter2D(image.gray_image, cv2.CV_16S, kernel_y)
        scale_abs_x = cv2.convertScaleAbs(x)
        scale_abs_y = cv2.convertScaleAbs(y)
        roberts_image = cv2.addWeighted(scale_abs_x, 0.5, scale_abs_y, 0.5, 0)
        return roberts_image

    @staticmethod
    def __get_roberts_kernels():
        kernel_x = np.array([[-1, 0], [0, 1]], dtype=int)
        kernel_y = np.array([[0, -1], [1, 0]], dtype=int)
        return kernel_x, kernel_y

    def use_sobel(self, image):
        kernel_x, kernel_y = self.__get_sobel_kernels()
        x = cv2.filter2D(image.gray_image, cv2.CV_16S, kernel_x)
        y = cv2.filter2D(image.gray_image, cv2.CV_16S, kernel_y)
        scale_abs_x = cv2.convertScaleAbs(x)
        scale_abs_y = cv2.convertScaleAbs(y)
        sobel_image = cv2.addWeighted(scale_abs_x, 0.5, scale_abs_y, 0.5, 0)
        return sobel_image

    @staticmethod
    def __get_sobel_kernels():
        kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=int)
        kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=int)
        return kernel_x, kernel_y

    def use_prewitt(self, image):
        kernel_x, kernel_y = self.__get_prewitt_kernels()
        x = cv2.filter2D(image.gray_image, cv2.CV_16S, kernel_x)
        y = cv2.filter2D(image.gray_image, cv2.CV_16S, kernel_y)
        scale_abs_x = cv2.convertScaleAbs(x)
        scale_abs_y = cv2.convertScaleAbs(y)
        prewitt_image = cv2.addWeighted(scale_abs_x, 0.5, scale_abs_y, 0.5, 0)
        return prewitt_image

    @staticmethod
    def __get_prewitt_kernels():
        kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)
        kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=int)
        return kernel_x, kernel_y

    def use_laplacian(self, image):
        kernel = self.__get_laplacian_kernel()
        laplacian_image = cv2.convertScaleAbs(cv2.filter2D(image.gray_image, cv2.CV_16S, kernel))
        return laplacian_image

    @staticmethod
    def __get_laplacian_kernel():
        kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=int)
        return kernel

    def use_kirsch(self, image):
        kernels = self.__get_kirsch_kernels()
        kirsch_image = cv2.filter2D(image.gray_image, cv2.CV_16S, kernels[0])
        for k in kernels[1:]:
            kirsch_image = np.dstack([kirsch_image, cv2.filter2D(image.gray_image, cv2.CV_16S, k)])
        kirsch_image = cv2.convertScaleAbs(np.max(np.abs(kirsch_image), axis=2))
        return kirsch_image

    @staticmethod
    def __get_kirsch_kernels():
        kernel_1 = np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]], dtype=int)
        kernel_2 = np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]], dtype=int)
        kernel_3 = np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]], dtype=int)
        kernel_4 = np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]], dtype=int)
        kernel_5 = np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]], dtype=int)
        kernel_6 = np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]], dtype=int)
        kernel_7 = np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]], dtype=int)
        kernel_8 = np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]], dtype=int)
        return [kernel_1, kernel_2, kernel_3, kernel_4, kernel_5, kernel_6, kernel_7, kernel_8]

    def use_canny(self, image):
        print(cv2.Canny(image.gray_image, 50, 150))
        return cv2.Canny(image.gray_image, 50, 150)

    def draw(self, image):
        lena_init = image.gray_image
        lena_roberts = self.use_roberts(image)
        lena_sobel = self.use_sobel(image)
        lena_prewitt = self.use_prewitt(image)
        lena_laplacian = self.use_laplacian(image)
        lena_kirsch = self.use_kirsch(image)
        self.__imshow_r2_l3(lena_init, lena_roberts, lena_sobel, lena_prewitt, lena_laplacian, lena_kirsch)

    def __imshow_r2_l3(self, lena_init, lena_roberts, lena_sobel, lena_prewitt, lena_laplacian, lena_kirsch):
        fig, ax = plt.subplots(2, 3)
        ax[0, 0].imshow(lena_init, 'gray')
        ax[0, 1].imshow(lena_roberts, 'gray')
        ax[0, 2].imshow(lena_sobel, 'gray')
        ax[1, 0].imshow(lena_prewitt, 'gray')
        ax[1, 1].imshow(lena_laplacian, 'gray')
        ax[1, 2].imshow(lena_kirsch, 'gray')
        self.__plot_r2_l3(ax)

    @staticmethod
    def __plot_r2_l3(ax):
        idx_to_title = [['Original Image', 'Robert', 'Sobel'], ['Prewitt', 'Laplacian', 'Kirsch']]
        for i in range(2):
            for j in range(3):
                ax[i, j].set_title(idx_to_title[i][j])
                ax[i, j].xaxis.set_ticks([])
                ax[i, j].yaxis.set_ticks([])
        plt.show()

if __name__ == '__main__':
    detection = Detection()
    image = Image('./images/lena.tif')
    canny = detection.use_canny(image)
    plt.imshow(canny, 'gray')
    plt.show()