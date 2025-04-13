# AUTHOR: Jakob G.
# Last Edit: 4/12/2025

import torch
import numpy as np
from PIL import Image
import cv2

###################################################################################################
####################################### New Model Section #########################################
###################################################################################################

class CannyFRCNN(torch.nn.Module):
    """
    Sequential-like Pytorch model utilizing Canny edge detection to detect target edges and remove noise
    prior to engaging FRCNN for target detection.

    Args:
        - frcnn: Main Faster-RCNN model from Pytorch
        - device: String name of the device to move modified images onto.
        - kernel: Kernel size of the Gaussian Blur function.
        - stdev: Standard Deviation of Gaussian Blur
        - threshold1: Initial Edge threshold of Canny algorithm
        - threshold2: Secondary Edge threshold of Canny algorithm
    """

    def __init__(self,
                 frcnn,
                 device: str,
                 kernel: tuple = (3, 3),
                 stdev: float = 1.0,
                 threshold1: float = 200,
                 threshold2: float = 100):
        super().__init__()

        self.frcnn = frcnn
        self.device = device
        self.kernel = kernel
        self.stdev = stdev
        self.threshold1 = threshold1
        self.threshold2 = threshold2

    def forward(self, X, Y):

        # Convert the input tuple into a list for editing
        X = list(X)

        x_1 = X[0].cpu().permute(1, 2, 0).numpy()
        # First convert tensor image to numpy array
        x_1 = np.uint8(X[0].cpu().permute(1, 2, 0).numpy())

        # Use Gaussian Blur to reduce Noise in image, run Canny, and convert the image back into the corret array shape.
        x_1 = cv2.GaussianBlur(x_1, ksize=self.kernel, sigmaX=self.stdev)
        x_1 = np.array(Image.fromarray(cv2.Canny(x_1, threshold1=self.threshold1, threshold2=self.threshold2)).convert('RGB'), dtype=np.float32).reshape((3, 640, 640))


        # Convert edited image back to Pytorch format and feed into FRCNN
        X[0] = torch.FloatTensor(torch.from_numpy(x_1)).to(self.device)

        del x_1 # Delete for memory conservation

        X = self.frcnn(X, Y)
        return X, Y