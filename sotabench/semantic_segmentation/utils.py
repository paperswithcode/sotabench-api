import cv2
import numpy as np
import torch
from torchvision.transforms import transforms

import albumentations as albu
from albumentations.core.transforms_interface import DualTransform

from PIL import Image


def minmax_normalize(img, norm_range=(0, 1), orig_range=(0, 255)):
    # range(0, 1)
    norm_img = (img - orig_range[0]) / (orig_range[1] - orig_range[0])
    # range(min_value, max_value)
    norm_img = norm_img * (norm_range[1] - norm_range[0]) + norm_range[0]
    return norm_img


class FlipChannels(object):
    def __call__(self, img):
        img = np.array(img)[:, :, ::-1]
        return Image.fromarray(img.astype(np.uint8))


class PadIfNeededRightBottom(DualTransform):
    def __init__(self, min_height=769, min_width=769, border_mode=cv2.BORDER_CONSTANT,
                 value=0, ignore_index=255, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.min_height = min_height
        self.min_width = min_width
        self.border_mode = border_mode
        self.value = value
        self.ignore_index = ignore_index

    def apply(self, img, **params):
        img_height, img_width = img.shape[:2]
        pad_height = max(0, self.min_height-img_height)
        pad_width = max(0, self.min_width-img_width)
        return np.pad(img, ((0, pad_height), (0, pad_width), (0, 0)), 'constant', constant_values=self.value)

    def apply_to_mask(self, img, **params):
        img_height, img_width = img.shape[:2]
        pad_height = max(0, self.min_height-img_height)
        pad_width = max(0, self.min_width-img_width)
        return np.pad(img, ((0, pad_height), (0, pad_width)), 'constant', constant_values=self.ignore_index)


class DefaultPascalTransform(object):
    """Applies standard Pascal Transforms"""

    def __init__(self, target_size, ignore_index):
        assert isinstance(target_size, tuple)
        self.target_size = target_size
        self.ignore_index = ignore_index

    def __call__(self, img, target):
        img = np.array(img)
        target = np.array(target)
        target[target == self.ignore_index] = 0
        # target[target == 255] = 0

        resizer = albu.Compose([PadIfNeededRightBottom(min_height=self.target_size[0], min_width=self.target_size[1],
                                                       value=0,
                                                       ignore_index=self.ignore_index, p=1.0),
                                albu.Crop(x_min=0, x_max=self.target_size[1],
                                          y_min=0, y_max=self.target_size[0])])

        img = minmax_normalize(img, norm_range=(-1, 1))
        resized = resizer(image=img, mask=target)
        img = resized['image']
        target = resized['mask']
        img = img.transpose(2, 0, 1)

        img = torch.FloatTensor(img)
        target = torch.LongTensor(target)

        return img, target


class CityscapesMaskConversion(object):
    """Converts Cityscape masks - adds an ignore index"""

    def __init__(self, ignore_index):
        self.ignore_index = ignore_index

        self.id_to_trainid = {-1: ignore_index, 0: ignore_index, 1: ignore_index, 2: ignore_index, 3: ignore_index,
                              4: ignore_index, 5: ignore_index, 6: ignore_index, 7: 0, 8: 1, 9: ignore_index,
                              10: ignore_index, 11: 2, 12: 3, 13: 4, 14: ignore_index, 15: ignore_index, 16: ignore_index,
                              17: 5, 18: ignore_index, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                              28: 15, 29: ignore_index, 30: ignore_index, 31: 16, 32: 17, 33: 18}

    def __call__(self, img):

        mask = np.array(img, dtype=np.int32)
        mask_copy = mask.copy()

        for k, v in self.id_to_trainid.items():
            mask_copy[mask == k] = v

        return torch.from_numpy(mask_copy).long()


class JointCompose(transforms.Compose):

    def __call__(self, img, target):
        for t in self.transforms:
            img, target = t(img, target)
        return img, target


def _fast_hist(label_pred, label_true, num_classes):
    mask = (label_true >= 0) & (label_true < num_classes)
    hist = np.bincount(
        num_classes * label_true[mask].astype(int) +
        label_pred[mask], minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return hist

def evaluate_segmentation(predictions, gts, num_classes):
    hist = np.zeros((num_classes, num_classes))
    for lp, lt in zip(predictions, gts):
        hist += _fast_hist(lp.flatten(), lt.flatten(), num_classes)
    # axis 0: gt, axis 1: prediction
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

    return {
        'Accuracy (overall)': acc,
        'Accuracy (class)': acc_cls,
        'Mean IoU': mean_iu,
        'Frequency Weighted Average Accuracy': fwavacc
    }

def get_segmentation_metrics(model, model_output_transform, test_loader, is_cuda=True):
    pred_all = []
    mask_all = []

    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):

            if is_cuda:
                images = images.cuda(non_blocking=True)
                labels = labels.cuda()

            # compute output
            output = model(images)

            if model_output_transform is not None:
                output = model_output_transform(output, labels)

            pred = output.data.max(1)[1].cpu().numpy()

            pred_all.append(pred)
            mask_all.append(labels.data.cpu().numpy())

        mask_all = np.concatenate(mask_all)
        pred_all = np.concatenate(pred_all)

    return evaluate_segmentation(pred_all, mask_all, test_loader.no_classes)