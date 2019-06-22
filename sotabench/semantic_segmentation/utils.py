import numpy as np
import torch
from torchvision.transforms import transforms

import albumentations as albu
from albumentations.core.transforms_interface import DualTransform


def minmax_normalize(img, norm_range=(0, 1), orig_range=(0, 255)):
    # range(0, 1)
    norm_img = (img - orig_range[0]) / (orig_range[1] - orig_range[0])
    # range(min_value, max_value)
    norm_img = norm_img * (norm_range[1] - norm_range[0]) + norm_range[0]
    return norm_img


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

    def __call__(self, sample):
        print(sample)
        img, target = sample[0], sample[1]
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


class DefaultCityscapesTransform(object):
    """Applies standard Pascal Transforms"""

    def __init__(self, target_size, ignore_index):
        self.target_size = target_size
        self.ignore_index = ignore_index

    def __call__(self, img, target):
        img = np.array(img)
        target = np.array(target)
        target[target == self.ignore_index] = 0

        img = minmax_normalize(img, norm_range=(-1, 1))
        img = img.transpose(2, 0, 1)

        img = torch.FloatTensor(img)
        target = torch.LongTensor(target)

        return img, target


class JointCompose(transforms.Compose):

    def __call__(self, img, target):
        for t in self.transforms:
            img, target = t(img, target)
        return img, target


def compute_ious(pred, label, classes, ignore_index=255, only_present=True):
    pred[label == ignore_index] = 0
    ious = []
    for c in classes:
        label_c = label == c
        if only_present and np.sum(label_c) == 0:
            ious.append(np.nan)
            continue
        pred_c = pred == c
        intersection = np.logical_and(pred_c, label_c).sum()
        union = np.logical_or(pred_c, label_c).sum()
        if union != 0:
            ious.append(intersection / union)
    return ious if ious else [1]


def compute_iou_batch(preds, labels, classes=None):
    iou = np.nanmean([np.nanmean(compute_ious(pred, label, classes)) for pred, label in zip(preds, labels)])
    return iou


def get_segmentation_metrics(model, model_output_transform, test_loader, is_cuda=True):

    train_ious = []

    with torch.no_grad():
        for i, (image, labels) in enumerate(test_loader):

            images = images.numpy().transpose(0, 2, 3, 1)
            labels = labels.numpy()

            if is_cuda:
                images = images.cuda(non_blocking=True)
                labels = labels.cuda()

            # compute output
            output = model(images)

            if model_output_transform is not None:
                output = model_output_transform(output, labels)

            output = output.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
            iou = compute_iou_batch(np.argmax(output, axis=1), labels, test_loader.no_classes)
            train_ious.append(iou)

    return {
        'Mean IOU': np.mean(train_ious)
    }
