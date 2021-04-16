import os

import matplotlib.pyplot as plt

from detection import Detection
from evaluate import Evaluate
from image import Image

class MultiProcess:
    """This class could handle multiple images.
    You could use multi_images_evaluate method to get all images' pnsr and ssim.
    You could use draw method to directly show all images and all operator.
    """
    def __init__(self):
        self.detection = Detection()
        self.evaluate = Evaluate()

    def multi_images_evaluate(self, path, noise=None):
        name2psnr, name2ssim = {}, {}
        for name_path in os.listdir(path):
            name = name_path.split('.')[0]
            image = Image(os.path.join(path, name_path), noise)
            psnrs, ssims = self.__get_image_psnrs_ssims(image)
            name2psnr[name] = psnrs
            name2ssim[name] = ssims
        return name2psnr, name2ssim

    def __get_image_psnrs_ssims(self, image):
        det_images = self.detection.use_all_operator(image)
        psnrs, ssims = [], []
        for img in det_images:
            psnrs.append(self.evaluate.compute_psnr(image.gray_image, img))
            ssims.append(self.evaluate.compute_sk_ssim(image.gray_image, img))
        return psnrs, ssims

    def draw(self, path, n_operator=5, noise=None):
        paths = os.listdir(path)
        fig, ax = plt.subplots(len(paths), n_operator+1)
        for i, name_path in enumerate(paths):
            image = Image(os.path.join(path, name_path), noise)
            self.__draw_single_image(ax, image, i)
        self.__set_plot(fig, ax, len(paths), n_operator+1)

    def __draw_single_image(self, ax, image, num):
        original = image.gray_image
        roberts = self.detection.use_roberts(image)
        sobel = self.detection.use_sobel(image)
        prewitt = self.detection.use_prewitt(image)
        laplacian = self.detection.use_laplacian(image)
        kirsch = self.detection.use_kirsch(image)
        self.__imshow(ax, num, original, roberts, sobel, prewitt, laplacian, kirsch)

    def __imshow(self, ax, num, *ops):
        for i, op in enumerate(ops):
            ax[num, i].imshow(op, 'gray')

    def __set_plot(self, fig, ax, row, col):
        idx2title = ['Original', 'Robert', 'Sobel', 'Prewitt', 'Laplacian', 'kirsch']
        for i in range(row):
            for j in range(col):
                ax[i, j].xaxis.set_ticks([])
                ax[i, j].yaxis.set_ticks([])
                if i == 0:
                    ax[i, j].set_title(idx2title[j])
        fig.subplots_adjust(wspace=-0.8)
        plt.show()

    def draw_line_chart(self, path, noise=None):
        psnrs, ssims = self.multi_images_evaluate(path, noise=noise)
        ops = ['Robert', 'Sobel', 'Prewitt', 'Laplacian', 'kirsch']
        markers = ['.', '^', 's', '*', 'x']
        self.__draw_line(ops, psnrs, type='psnr', markers=markers)
        self.__draw_line(ops, ssims, type='ssim', markers=markers)

    def __draw_line(self, ops, name2values, type, markers):
        for i, op in enumerate(ops):
            values = []
            for val in name2values.values():
                values.append(val[i])
            plt.plot(name2values.keys(), values, marker=markers[i], label=op)
        plt.ylabel(type)
        plt.legend(loc="upper left")
        plt.show()

    def draw_all_canny(self, path, noise=None):
        paths = os.listdir(path)
        fig, ax = plt.subplots(1, len(paths))
        for i, name_path in enumerate(paths):
            image = Image(os.path.join(path, name_path), noise)
            canny = self.detection.use_canny(image)
            ax[i].imshow(canny, 'gray')
        plt.show()

if __name__ == '__main__':
    multi_process = MultiProcess()
    multi_process.draw_all_canny('./images/')