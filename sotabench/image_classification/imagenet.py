import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from sotabench.core import BenchmarkResult, evaluate

from .utils import get_classification_metrics

@evaluate
def benchmark(
        model,
        input_transform=None,
        target_transform=None,
        is_cuda: bool = True,
        num_workers: int = 4,
        batch_size: int = 128,
        data_root: str = './data') -> BenchmarkResult:

    if is_cuda:
        model = model.cuda()

    model.eval()

    if not input_transform:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        input_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

    test_dataset = datasets.ImageNet(data_root, split='val', transform=input_transform, target_transform=target_transform, download=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    criterion = nn.CrossEntropyLoss()

    metrics = get_classification_metrics(model=model, test_loader=test_loader, criterion=criterion, is_cuda=is_cuda)

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=metrics['top_1_accuracy'], top5=metrics['top_5_accuracy']))

    return BenchmarkResult(task="Image Classification", dataset=test_dataset, metrics=metrics)
