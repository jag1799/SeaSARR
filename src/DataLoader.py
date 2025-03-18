from PIL import Image
from pycocotools.coco import COCO
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import os

class SAR_DataLoader(torchvision.datasets.VisionDataset):

    """
    Custom Data Loader object inspired from Pytorch's CocoDetection dataset loader.

    Args:
        - root (str): Root directory for an image set. (i.e. /train, /valid, /test)
        - annotation_file (str): Path to the set's image annotations.
        - transforms: Function/Transform that takes a sample of images
                      and their targets as input, and returns transformed versions.
        - transform: Function/Transform that intakes a PIL image and returns the transformed image.
        - target_transform: Function/Transform that intakes a target and transforms it.
    """

    def __init__(self, root, annotation_file, transforms = None,
                 transform = None, target_transform = None):
        super().__init__(root, transforms, transform, target_transform)

        self._sarScope = COCO(annotation_file=annotation_file)
        self._ids = list(sorted(self._sarScope.imgs.keys()))

    def _load_image(self, id: int):
        img_path = self._sarScope.loadImgs(id)[0]['file_name']
        return Image.open(os.path.join(self.root, img_path)).convert("RGB")

    def _load_target(self, id: int):
        return self._sarScope.loadAnns(self._sarScope.getAnnIds(id))

    def _show_index(self, index):
        id = self._ids[index]

        image = self._load_image(id)
        target = self._load_target(id)

        plt.imshow(np.asarray(image))
        self._sarScope.showAnns(target, draw_bbox=True)

        plt.title(f"Image Id: {id}")
        plt.show()

    def _show_random_idx(self):
        import random
        num_images = self.__len__()
        idx = random.randint(0, num_images-1)
        self._show_index(idx)

    def __getitem__(self, index):

        id = self._ids[index]
        image = self._load_image(id)
        target = self._load_target(id)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        return len(self._ids)
