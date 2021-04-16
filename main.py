import time

from multi_process import MultiProcess
from image import Image
from evaluate import Evaluate
from detection import Detection

if __name__ == '__main__':
    multi_process = MultiProcess()
    multi_process.draw('./images', noise=None)

