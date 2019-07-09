import time
import torch

from sotabench.utils import AverageMeter, accuracy, send_data_to_device

def evaluate_classification(model, test_loader, model_output_transform, send_data_to_device, device='cuda'):
    batch_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()

    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):

            input, target = send_data_to_device(input, target, device=device)
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
