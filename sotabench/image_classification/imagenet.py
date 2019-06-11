import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from .utils import get_classification_metrics

def benchmark(
        model,
        input_transform=None,
        target_transform=None,
        is_cuda=True,
        num_workers=4,
        batch_size=128):

    if is_cuda:
        model = model.cuda()

    model.eval()

    if not input_transform:
        input_transform = transforms.Compose([transforms.ToTensor()])

    test_dataset = datasets.ImageNet('./data', split='train', transform=input_transform, target_transform=target_transform, download=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    criterion = nn.CrossEntropyLoss()

    metrics = get_classification_metrics(model=model, test_loader=test_loader, criterion=criterion, is_cuda=is_cuda)

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=metrics['top_1_accuracy'], top5=metrics['top_5_accuracy']))

    return metrics
