import time

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from sotabench.utils import AverageMeter, accuracy

def evaluate_cifar10(
        model,
        input_transform=None,
        target_transform=None,
        is_cuda=True,
        num_workers=4,
        batch_size=128):

    # with torch.no_grad():

    if is_cuda:
        model = model.cuda()

    # switch model to evaluation model
    model.eval()

    # load input transforms
    if not input_transform:
        input_transform = transforms.Compose([transforms.ToTensor()])

    # load the dataset
    test_dataset = datasets.CIFAR10('/', train=False, transform=input_transform, target_transform=target_transform, download=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # set up criterion
    criterion = nn.CrossEntropyLoss()

    # set up average meters for metric calculation
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for i, (input, target) in enumerate(test_loader):
        target = target.cuda(non_blocking=True)

        # compute output
        output = model(input.cuda())
        loss = criterion(output, target)
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return {
        'top_1_accuracy': top1.avg,
        'top_5_accuracy': top5.avg
        }
