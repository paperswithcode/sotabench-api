import time

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from sotalearn.utils import AverageMeter, accuracy

def evaluate(
        model,
        input_transform=None,
        target_transform=None,
        is_cuda=True,
        num_workers=4,
        batch_size=128,
        print_freq=100):

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

        if i % print_freq == 0:
            print('Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(test_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

        return top1.avg, top5.avg
