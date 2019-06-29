import numpy as np
import torch
import torchvision

from pycocotools import mask as coco_mask
from pycocotools.coco import COCO

from .coco_eval import CocoEvaluator


def collate_fn(batch):
    return tuple(zip(*batch))


def convert_to_coco_api(ds):
    coco_ds = COCO()
    ann_id = 0
    dataset = {'images': [], 'categories': [], 'annotations': []}
    categories = set()
    for img_idx in range(len(ds)):
        # find better way to get target
        # targets = ds.get_annotations(img_idx)
        img, targets = ds[img_idx]
        image_id = targets["image_id"].item()
        img_dict = {}
        img_dict['id'] = image_id
        img_dict['height'] = img.shape[-2]
        img_dict['width'] = img.shape[-1]
        dataset['images'].append(img_dict)
        bboxes = targets["boxes"]
        bboxes[:, 2:] -= bboxes[:, :2]
        bboxes = bboxes.tolist()
        labels = targets['labels'].tolist()
        areas = targets['area'].tolist()
        iscrowd = targets['iscrowd'].tolist()
        if 'masks' in targets:
            masks = targets['masks']
            # make masks Fortran contiguous for coco_mask
            masks = masks.permute(0, 2, 1).contiguous().permute(0, 2, 1)
        if 'keypoints' in targets:
            keypoints = targets['keypoints']
            keypoints = keypoints.reshape(keypoints.shape[0], -1).tolist()
        num_objs = len(bboxes)
        for i in range(num_objs):
            ann = {}
            ann['image_id'] = image_id
            ann['bbox'] = bboxes[i]
            ann['category_id'] = labels[i]
            categories.add(labels[i])
            ann['area'] = areas[i]
            ann['iscrowd'] = iscrowd[i]
            ann['id'] = ann_id
            if 'masks' in targets:
                ann["segmentation"] = coco_mask.encode(masks[i].numpy())
            if 'keypoints' in targets:
                ann['keypoints'] = keypoints[i]
                ann['num_keypoints'] = sum(k != 0 for k in keypoints[i][2::3])
            dataset['annotations'].append(ann)
            ann_id += 1
    dataset['categories'] = [{'id': i} for i in sorted(categories)]
    coco_ds.dataset = dataset
    coco_ds.createIndex()
    return coco_ds


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        if isinstance(dataset, torchvision.datasets.CocoDetection):
            break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco
    return convert_to_coco_api(dataset)


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


def get_coco_metrics(coco_evaluator):

    metrics = {'AP (BB)': None, 'AP (BB) 50': None, 'AP (BB) 75': None, 'AP (BB) S': None, 'AP (BB) M': None,
               'AP (BB) L': None}
    iouThrs = [None, .5, .75, None, None, None]
    maxDets = [100] + [coco_evaluator.coco_eval['bbox'].params.maxDets[2]] * 5
    areaRngs = ['all', 'all', 'all', 'small', 'medium', 'large']
    bounding_box_params = coco_evaluator.coco_eval['bbox'].params

    for metric_no, metric in enumerate(metrics):
        aind = [i for i, aRng in enumerate(bounding_box_params.areaRngLbl) if aRng == areaRngs[metric_no]]
        mind = [i for i, mDet in enumerate(bounding_box_params.maxDets) if mDet == maxDets[metric_no]]

        s = coco_evaluator.coco_eval['bbox'].eval['precision']

        # IoU
        if iouThrs[metric_no] is not None:
            t = np.where(iouThrs[metric_no] == bounding_box_params.iouThrs)[0]
            s = s[t]
        s = s[:, :, :, aind, mind]

        if len(s[s > -1]) == 0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s > -1])

        metrics[metric] = mean_s

    return metrics


def evaluate_detection_coco(model, model_output_transform, test_loader, device='cuda'):

    coco = get_coco_api_from_dataset(test_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
            input = list(inp.to(device=device, non_blocking=True) for inp in input)
            target = [{k: v.to(device=device, non_blocking=True) for k, v in t.items()} for t in target]

            # compute output
            output = model(input)

            if model_output_transform is not None:
                output = model_output_transform(output, target)
            elif test_loader.no_classes == 91:  # COCO
                output = [{k: v.to('cpu') for k, v in t.items()} for t in output]  # default torchvision extraction

            result = {tar["image_id"].item(): out for tar, out in zip(target, output)}
            coco_evaluator.update(result)

    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    return get_coco_metrics(coco_evaluator)
