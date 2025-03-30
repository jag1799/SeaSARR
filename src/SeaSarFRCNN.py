from torchvision.datasets import CocoDetection
import matplotlib.pyplot as plt
import numpy as np
import random

class SeaSarFRCNN(CocoDetection):
    """`MS Coco Detection <https://cocodataset.org/#detection-2016>`_ Dataset.

    It requires the `COCO API to be installed <https://github.com/pdollar/coco/tree/master/PythonAPI>`_.

    Custom Dataset class for SARscope dataset on Kaggle:
    <https://www.kaggle.com/datasets/kailaspsudheer/sarscope-unveiling-the-maritime-landscape/data>

    Code taken directly from Pytorch's COCODetection class to overwrite existing methods.

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
        from pycocotools.coco import COCO

        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.num_images = len(self.ids)

    def show_random_image(self):
        """
        Show a random image from the dataset with its annotations.
        """
        idx = random.randint(0, self.num_images - 1)

        self._show_image(idx)

    def _show_image(self, index):
        """
        Show the image at the specified index with its annotations.
        Args:
            index: Index of the image to be shown.
        """

        id = self.ids[index]

        image = self._load_image(id)
        target = self._load_target(id)

        plt.imshow(np.asarray(image))
        self.coco.showAnns(target, draw_bbox=True)

        plt.title(f"Image Id: {id}")
        plt.show()

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