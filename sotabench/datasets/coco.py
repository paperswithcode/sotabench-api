import torchvision


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, root, annFile, transforms, **kwargs):
        super(CocoDetection, self).__init__(root, annFile, **kwargs)
        self._transforms = transforms

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = dict(image_id=image_id, annotations=target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target
