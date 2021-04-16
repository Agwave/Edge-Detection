
import numpy as np
from skimage.metrics import structural_similarity as ssim

class Evaluate:
    """The class realized evaluation methods of edge detection algorithm.
    You could use compute_psnr method to compute PSNR.
    You could use compute_sk_ssim method to compute ssim.
    """
    def __init__(self):
        pass

    @staticmethod
    def compute_psnr(img1, img2):
        mse = np.mean(np.square(img1-img2))
        psnr = 10 * np.log10(255 * 255 / mse)
        return psnr

    @staticmethod
    def compute_sk_ssim(img1, img2):
        ssim_score = ssim(img1, img2, data_range=255)
        return ssim_score

    def compute_my_ssim(self, img1, img2, k1=0.01, k2=0.03, win_size=11, l=255):
        img1, img2, m, n = self.__get_check_image(img1, img2, win_size)
        ssim_sum, win_cnt, c_1, c_2 = 0., 0, (k1 * l) ** 2, (k2 * l) ** 2
        for i in range(m-win_size):
            for j in range(n-win_size):
                win_1 = img1[i:i+win_size, j:j+win_size]
                win_2 = img2[i:i+win_size, j:j+win_size]
                ssim_sum += self.__get_win_ssim(win_1, win_2, c_1, c_2)
                win_cnt += 1
        return ssim_sum / win_cnt

    @staticmethod
    def __get_check_image(img1, img2, win_size):
        if img1.dtype == np.uint8:
            img1 = np.double(img1)
        if img2.dtype == np.uint8:
            img2 = np.double(img2)
        m, n = img1.shape
        assert m >= win_size and n >= win_size
        return img1, img2, m, n

    def __get_win_ssim(self, win_1, win_2, c_1, c_2):
        mu_1, var_1 = self.__get_mu_var(win_1)
        mu_2, var_2 = self.__get_mu_var(win_2)
        convar = np.mean(np.matmul(win_1-mu_1, win_2-mu_2))
        ssim = ((2 * mu_1 * mu_2 + c_1) * (2 * convar + c_2)) / (
                    (mu_1 ** 2 + mu_2 ** 2 + c_1) * (var_1 + var_2 + c_2))
        return ssim

    @staticmethod
    def __get_mu_var(win):
        mu = np.mean(win)
        var = np.mean(np.square(win - mu))
        return mu, var