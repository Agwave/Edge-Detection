import time

from image import Image
from evaluate import Evaluate
from detection import Detection

if __name__ == '__main__':
    lena = Image('./images/lena_gray_256.tif')
    eval = Evaluate()
    detection = Detection()
    lena_roberts = detection.use_roberts(lena)
    lena_sobel = detection.use_sobel(lena)
    lena_prewitt = detection.use_prewitt(lena)
    lena_laplacian = detection.use_laplacian(lena)
    lena_kirsch = detection.use_kirsch(lena)
    start = time.time()
    robert_psnr = eval.compute_psnr(lena.gray_image, lena_roberts)
    robert_ssim = eval.compute_sk_ssim(lena.gray_image, lena_roberts)
    sobel_psnr = eval.compute_psnr(lena.gray_image, lena_sobel)
    sobel_ssim = eval.compute_sk_ssim(lena.gray_image, lena_sobel)
    prewitt_psnr = eval.compute_psnr(lena.gray_image, lena_prewitt)
    prewitt_ssim = eval.compute_sk_ssim(lena.gray_image, lena_prewitt)
    print('robert_psnr, sobel_psnr, prewitt_psnr')
    print(robert_psnr, sobel_psnr, prewitt_psnr)
    print()
    print('robert_ssim, sobel_ssim, prewitt_ssim')
    print(robert_ssim, sobel_ssim, prewitt_ssim)
    end = time.time()
    print('耗时： {}'.format(end-start))

