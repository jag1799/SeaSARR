# AUTHOR: Jakob G.
# Last Edited: 04/12/2025

from torchvision.datasets import CocoDetection
import matplotlib.pyplot as plt
import numpy as np
import random
from pycocotools.coco import COCO

class SeaSarFRCNN(CocoDetection):
    """`MS Coco Detection <https://cocodataset.org/#detection-2016>`_ Dataset.

    It requires the `COCO API to be installed <https://github.com/pdollar/coco/tree/master/PythonAPI>`_.

    Custom Dataset class for SARscope dataset on Kaggle:
    <https://www.kaggle.com/datasets/kailaspsudheer/sarscope-unveiling-the-maritime-landscape/data>

    Pytorch dataset subclass configured using the CocoDetection subclass for compatability with Torch
    Dataloader objects.

    Args:
        root (str or ``pathlib.Path``): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        root: str ,
        annFile: str,
        transform = None,
        target_transform = None,
        transforms = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)


        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.num_images = len(self.ids)

    def show_edited_subplot(self, images: list, titles: list, xlabel: str, ylabel: str):
        """
        Shows multiple edited versions of the same images. Used for comparing the differences a layer makes
        to an image.

        NOTE: User MUST add plt.show() after they call this function.  This is to allow the user
        to add any additional edits to the plot if they wish.

        Args:
            - images: List of images to show.
            - titles: Titles of each image.
            - xlabel: Label for the figure x-axis.  Input "" to keep blank.
            - ylabel: Label for the figure y-axis. Input "" to keep blank.
        """

        fig, ax = plt.subplots(1, len(images), figsize=(10, 10))

        for i, image in enumerate(images):
            ax[i].imshow(image)
            ax[i].set_title(titles[i])
            ax[i].set_xlabel(xlabel)
            ax[i].set_ylabel(ylabel)

    def _random_image(self, get_image: bool = False, show_image: bool = True):
        """
        Shows and/or returns a random image from the dataset with its annotations.

        NOTE: User MUST add plt.show() after they call this function.  This is to allow the user
        to add any additional edits to the plot if they wish.

        Args:
            - get_image: Returns a randomly selected image from the dataset if true
            - show_image: Displays a randomly selected image from the dataset if true
        Returns:
            - image: PIL.Image object containing an random image if get_image is true.
            - target: List containing annotations for each target in the corresponding image.
        """
        idx = random.randint(0, self.num_images - 1)

        if get_image:
            image, target = self._get_image(idx, get_image, show_image)
            return image, target
        else:
            self._get_image(idx)


    def _get_image(self, index, get_image: bool = False, show_image: bool = True):
        """
        Show the image at the specified index with its annotations.

        NOTE: User MUST add plt.show() after they call this function.  This is to allow the user
        to add any additional edits to the plot if they wish.

        Args:
            - index: Index of the image to be shown.
            - get_image: Returns an image and its annotations if true.
            - show_image: Displays an image and its annotations if true.
        Returns:
            - image: PIL Image containing an image at the current index.
            - target: List of target annotations of image.
        """

        id = self.ids[index]

        image = self._load_image(id)
        target = self._load_target(id)

        if show_image:
            plt.imshow(np.asarray(image))
            self.coco.showAnns(target, draw_bbox=True)
            plt.title(f"Image Id: {id}")

        if get_image:
            return image, target

    def __getitem__(self, index: int):

        if not isinstance(index, int):
            raise ValueError(f"Index must be of type integer, got {type(index)} instead.")

        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)

        if self.transform is not None:
            image = self.transform(image)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def _packager(self, batch):
        return tuple(zip(*batch))