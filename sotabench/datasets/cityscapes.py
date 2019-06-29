import json
import os
import zipfile
from collections import namedtuple

from torchvision.datasets.vision import VisionDataset
from PIL import Image

SEMANTIC_SEGMENTATION_IGNORE_LABEL = 255


class Cityscapes(VisionDataset):
    """`Cityscapes <http://www.cityscapes-dataset.com/>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory ``leftImg8bit``
            and ``gtFine`` or ``gtCoarse`` are located.
        split (string, optional): The image split to use, ``train``, ``test`` or ``val`` if mode="gtFine"
            otherwise ``train``, ``train_extra`` or ``val``
        mode (string, optional): The quality mode to use, ``gtFine`` or ``gtCoarse``
        target_type (string or list, optional): Type of target to use, ``instance``, ``semantic``, ``polygon``
            or ``color``. Can also be a list to output a tuple with all specified target types.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    Examples:
        Get semantic segmentation target
        .. code-block:: python
            dataset = Cityscapes('./data/cityscapes', split='train', mode='fine',
                                 target_type='semantic')
            img, smnt = dataset[0]
        Get multiple targets
        .. code-block:: python
            dataset = Cityscapes('./data/cityscapes', split='train', mode='fine',
                                 target_type=['instance', 'color', 'polygon'])
            img, (inst, col, poly) = dataset[0]
        Validate on the "coarse" set
        .. code-block:: python
            dataset = Cityscapes('./data/cityscapes', split='val', mode='coarse',
                                 target_type='semantic')
            img, smnt = dataset[0]
    """

    # Based on https://github.com/mcordts/cityscapesScripts
    CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])

    classes = [
        CityscapesClass('unlabeled', 0, SEMANTIC_SEGMENTATION_IGNORE_LABEL, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('ego vehicle', 1, SEMANTIC_SEGMENTATION_IGNORE_LABEL, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('rectification border', 2, SEMANTIC_SEGMENTATION_IGNORE_LABEL, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('out of roi', 3, SEMANTIC_SEGMENTATION_IGNORE_LABEL, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('static', 4, SEMANTIC_SEGMENTATION_IGNORE_LABEL, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('dynamic', 5, SEMANTIC_SEGMENTATION_IGNORE_LABEL, 'void', 0, False, True, (111, 74, 0)),
        CityscapesClass('ground', 6, SEMANTIC_SEGMENTATION_IGNORE_LABEL, 'void', 0, False, True, (81, 0, 81)),
        CityscapesClass('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
        CityscapesClass('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
        CityscapesClass('parking', 9, SEMANTIC_SEGMENTATION_IGNORE_LABEL, 'flat', 1, False, True, (250, 170, 160)),
        CityscapesClass('rail track', 10, SEMANTIC_SEGMENTATION_IGNORE_LABEL, 'flat', 1, False, True, (230, 150, 140)),
        CityscapesClass('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
        CityscapesClass('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
        CityscapesClass('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
        CityscapesClass('guard rail', 14, SEMANTIC_SEGMENTATION_IGNORE_LABEL, 'construction', 2, False, True, (180, 165, 180)),
        CityscapesClass('bridge', 15, SEMANTIC_SEGMENTATION_IGNORE_LABEL, 'construction', 2, False, True, (150, 100, 100)),
        CityscapesClass('tunnel', 16, SEMANTIC_SEGMENTATION_IGNORE_LABEL, 'construction', 2, False, True, (150, 120, 90)),
        CityscapesClass('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
        CityscapesClass('polegroup', 18, SEMANTIC_SEGMENTATION_IGNORE_LABEL, 'object', 3, False, True, (153, 153, 153)),
        CityscapesClass('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
        CityscapesClass('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
        CityscapesClass('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
        CityscapesClass('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
        CityscapesClass('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
        CityscapesClass('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
        CityscapesClass('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
        CityscapesClass('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
        CityscapesClass('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
        CityscapesClass('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
        CityscapesClass('caravan', 29, SEMANTIC_SEGMENTATION_IGNORE_LABEL, 'vehicle', 7, True, True, (0, 0, 90)),
        CityscapesClass('trailer', 30, SEMANTIC_SEGMENTATION_IGNORE_LABEL, 'vehicle', 7, True, True, (0, 0, 110)),
        CityscapesClass('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
        CityscapesClass('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
        CityscapesClass('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
        CityscapesClass('license plate', -1, -1, 'vehicle', 7, False, True, (0, 0, 142)),
    ]

    def __init__(self, root, split='train', mode='fine', target_type='instance',
                 transform=None, target_transform=None, transforms=None):
        super(Cityscapes, self).__init__(root, transforms, transform, target_transform)
        self.transform = transform
        self.target_transform = target_transform
        self.mode = 'gtFine' if mode == 'fine' else 'gtCoarse'
        self.images_dir = os.path.join(self.root, 'leftImg8bit', split)
        self.targets_dir = os.path.join(self.root, self.mode, split)
        self.target_type = target_type
        self.split = split
        self.images = []
        self.targets = []

        if mode not in ['fine', 'coarse']:
            raise ValueError('Invalid mode! Please use mode="fine" or mode="coarse"')

        if mode == 'fine' and split not in ['train', 'test', 'val']:
            raise ValueError('Invalid split for mode "fine"! Please use split="train", split="test"'
                             ' or split="val"')
        elif mode == 'coarse' and split not in ['train', 'train_extra', 'val']:
            raise ValueError('Invalid split for mode "coarse"! Please use split="train", split="train_extra"'
                             ' or split="val"')

        if not isinstance(target_type, list):
            self.target_type = [target_type]

        if not all(t in ['instance', 'semantic', 'polygon', 'color'] for t in self.target_type):
            raise ValueError('Invalid value for "target_type"! Valid values are: "instance", "semantic", "polygon"'
                             ' or "color"')

        if not os.path.isdir(self.images_dir) or not os.path.isdir(self.targets_dir):
            image_dir_zip = os.path.join(self.root, 'leftImg8bit') + '_trainvaltest.zip'

            if self.mode == 'gtFine':
                target_dir_zip = os.path.join(self.root, self.mode) + '_trainvaltest.zip'
            elif self.mode == 'gtCoarse':
                target_dir_zip = os.path.join(self.root, self.mode)

            if os.path.isfile(image_dir_zip) and os.path.isfile(target_dir_zip):
                extract_cityscapes_zip(zip_location=image_dir_zip, root=self.root)
                extract_cityscapes_zip(zip_location=target_dir_zip, root=self.root)
            else:
                raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders for the'
                                   ' specified "split" and "mode" are inside the "root" directory')

        for city in os.listdir(self.images_dir):
            img_dir = os.path.join(self.images_dir, city)
            target_dir = os.path.join(self.targets_dir, city)
            for file_name in os.listdir(img_dir):
                target_types = []
                for t in self.target_type:
                    target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0],
                                                 self._get_target_suffix(self.mode, t))
                    target_types.append(os.path.join(target_dir, target_name))

                self.images.append(os.path.join(img_dir, file_name))
                self.targets.append(target_types)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """

        image = Image.open(self.images[index]).convert('RGB')

        targets = []
        for i, t in enumerate(self.target_type):
            if t == 'polygon':
                target = self._load_json(self.targets[index][i])
            else:
                target = Image.open(self.targets[index][i])

            targets.append(target)

        target = tuple(targets) if len(targets) > 1 else targets[0]

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        return len(self.images)

    def extra_repr(self):
        lines = ["Split: {split}", "Mode: {mode}", "Type: {target_type}"]
        return '\n'.join(lines).format(**self.__dict__)

    def _load_json(self, path):
        with open(path, 'r') as file:
            data = json.load(file)
        return data

    def _get_target_suffix(self, mode, target_type):
        if target_type == 'instance':
            return '{}_instanceIds.png'.format(mode)
        elif target_type == 'semantic':
            return '{}_labelIds.png'.format(mode)
        elif target_type == 'color':
            return '{}_color.png'.format(mode)
        else:
            return '{}_polygons.json'.format(mode)


def extract_cityscapes_zip(zip_location, root):
    zip_file = zipfile.ZipFile(zip_location, 'r')
    zip_file.extractall(root)
    zip_file.close()
