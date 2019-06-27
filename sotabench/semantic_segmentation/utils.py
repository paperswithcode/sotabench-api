import cv2
import numpy as np
import torch
import albumentations as albu
from albumentations.core.transforms_interface import DualTransform

from PIL import Image
from collections import deque

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


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class ConfusionMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, a, b):
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=a.device)
        with torch.no_grad():
            k = (a >= 0) & (a < n)
            inds = n * a[k].to(torch.int64) + b[k]
            self.mat += torch.bincount(inds, minlength=n**2).reshape(n, n)

    def reset(self):
        self.mat.zero_()

    def compute(self):
        h = self.mat.float()
        acc_global = torch.diag(h).sum() / h.sum()
        acc = torch.diag(h) / h.sum(1)
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return acc_global, acc, iu

    def reduce_from_all_processes(self):
        if not torch.distributed.is_available():
            return
        if not torch.distributed.is_initialized():
            return
        torch.distributed.barrier()
        torch.distributed.all_reduce(self.mat)

    def __str__(self):
        acc_global, acc, iu = self.compute()
        return (
            'global correct: {:.1f}\n'
            'average row correct: {}\n'
            'IoU: {}\n'
            'mean IoU: {:.1f}').format(
                acc_global.item() * 100,
                ['{:.1f}'.format(i) for i in (acc * 100).tolist()],
                ['{:.1f}'.format(i) for i in (iu * 100).tolist()],
                iu.mean().item() * 100)


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs


def collate_fn(batch):
    images, targets = list(zip(*batch))
    batched_imgs = cat_list(images, fill_value=0)
    batched_targets = cat_list(targets, fill_value=255)
    return batched_imgs, batched_targets

def evaluate_segmentation(model, model_output_transform, test_loader, device='cuda'):

    confmat = ConfusionMatrix(test_loader.num_classes)

    with torch.no_grad():
        print(len(test_loader))
        for i, (input, target) in enumerate(test_loader):

            target = target.to(device=device, non_blocking=True)
            input = input.to(device=device, non_blocking=True)

            # compute output
            output = model(input)

            if model_output_transform is not None:
                output = model_output_transform(output, target)

            confmat.update(target.flatten(), output.argmax(1).flatten())
            print(i)

    acc_global, acc, iu = confmat.compute()

    return {
        'Accuracy': acc_global.item() * 100,
        'Mean IOU': iu.mean().item() * 100
    }