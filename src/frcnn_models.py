# AUTHOR: Jakob G.
# Last Edit: 4/12/2025

import torch
import numpy as np
from PIL import Image
import cv2

###################################################################################################
####################################### Utilities Section #########################################
###################################################################################################


class RobinsonCompass:
    """
    Robinson Compass is an image masking method that uses a specified Kernel for each
    significant compass direction.  It first uses the Sobel edge detection to extract
    the vertical and horizontal gradient of an image's pixel intensities. It then
    applies the 8 kernels on the gradient images and uses Non-Maximum Suppression
    to remove fake edges.

    Args:
        - kernel: Gaussian Blur Kernel
        - stdev: Gaussian Blur standard deviation
        - threshold: Threshold of the edge detection algorithm
    """

    kernels = [
        np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
        np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]),
        np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]),
        np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]]),
        np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]),
        np.array([[0, -1, -2], [1, 0, -1], [2, 1, 0]]),
        np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
        np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]]),
    ]

    def __init__(self, kernel: tuple, stdev: float, threshold: float = 1.0):
        self.gauss_kernel = kernel
        self.stdev = stdev
        self._threshold = threshold

    def get_detections(self, image: np.ndarray):
        """
        Return a single image show all edges detected by Robinson Compass.

        Args:
            - image: Numpy array representation of a Pytorch image.
        Returns:
            - edges: Numpy array containing all detected edges.
        """

        # First, convert the image to Grayscale.  Masking will not work properly otherwise.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Remove some noise through blurring.
        gaussian_image = cv2.GaussianBlur(image, self.gauss_kernel, self.stdev)

        # Do Sobel to get Gradient images.
        grad_x = cv2.Sobel(gaussian_image, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gaussian_image, cv2.CV_32F, 0, 1, ksize=3)

        # Get the detected edges from each direction.
        edges = []
        for kernel in self.kernels:
            edges.append(cv2.filter2D(grad_x, cv2.CV_32F, kernel) + cv2.filter2D(grad_y, cv2.CV_32F, kernel))

        # Non-Maximum Suppression to remove fake edges.
        edges = np.max(edges, axis=0)

        # Threshold all remaining edges.
        edges = edges > self._threshold

        return edges


###################################################################################################
####################################### New Model Section #########################################
###################################################################################################

class RobinsonFRCNN(torch.nn.Module):
    """
    Sequential-like Pytorch model utilizing Robinson Compass edge detection to detect target edges and remove
    excess noise prior to engaging FRCNN for target detection.

    Args:
        - frcnn: Main Faster-RCNN model from Pytorch
        - device: String name of the device to move modified images onto.
        - kernel: Kernel size of the Gaussian Blur function.
        - stdev: Standard Deviation of Gaussian Blur
        - threshold: Minimum pixel intensity for an edge during Robinson Compass
    """

    def __init__(self, frcnn, device: str, kernel: tuple = (3, 3), stdev: float = 1.0, threshold: float = 400):
        super().__init__()
        self.frcnn = frcnn
        self.device = device
        self.rob_comp = RobinsonCompass(kernel, stdev, threshold)

    def forward(self, X, Y):
        # Convert the input tuple into a list for editing
        X = list(X)

        # First convert tensor image to numpy array of type uint8_t. This is what OpenCV expects the individual elements to be.
        x_1 = np.uint8(X[0].cpu().permute(1, 2, 0).numpy())

        x_1 = self.rob_comp.get_detections(x_1)

        # Convert back to the expected shape.
        x_1 = np.array(Image.fromarray(x_1).convert('RGB'), dtype=np.float32).reshape((3, 640, 640))

        # Convert edited image back to Pytorch format and feed into FRCNN
        X[0] = torch.FloatTensor(torch.from_numpy(x_1)).to(self.device)

        del x_1

        X = self.frcnn(X, Y)

        return X, Y

class CannyFRCNN(torch.nn.Module):
    """
    Sequential-like Pytorch model utilizing Canny edge detection to detect target edges and remove
    excess noise prior to engaging FRCNN for target detection.

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

        # First convert tensor image to numpy array of type uint8_t. This is what OpenCV expects the individual elements to be.
        x_1 = np.uint8(X[0].cpu().permute(1, 2, 0).numpy())

        # Use Gaussian Blur to reduce Noise in image, run Canny, and convert the image back into the corret array shape.
        x_1 = cv2.GaussianBlur(x_1, ksize=self.kernel, sigmaX=self.stdev)
        x_1 = np.array(Image.fromarray(cv2.Canny(x_1, threshold1=self.threshold1, threshold2=self.threshold2)).convert('RGB'), dtype=np.float32).reshape((3, 640, 640))

        # Convert edited image back to Pytorch format and feed into FRCNN
        X[0] = torch.FloatTensor(torch.from_numpy(x_1)).to(self.device)

        del x_1 # Delete for memory conservation

        X = self.frcnn(X, Y)
        return X, Y