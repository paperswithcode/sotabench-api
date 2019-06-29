import time
import torch

from sotabench.utils import AverageMeter, accuracy


def evaluate_classification(model, model_output_transform, test_loader,
                            device='cuda'):
    batch_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()

    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):

            target = target.to(device=device, non_blocking=True)
            input = input.to(device=device, non_blocking=True)

            # compute output
            output = model(input)

            if model_output_transform is not None:
                output = model_output_transform(output, target)

            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

    return {
        'Top 1 Accuracy': top1.avg / 100,
        'Top 5 Accuracy': top5.avg / 100
    }
