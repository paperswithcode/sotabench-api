import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from sotabench.core import BenchmarkResult, evaluate

from .utils import get_classification_metrics

@evaluate
def benchmark(
        model,
        input_transform=None, target_transform=None,
        is_cuda: bool = True,
        data_root: str = './data',
        num_workers: int = 4, batch_size: int = 128, num_gpu: int = 2,
        paper_model_name: str = None, paper_arxiv_id: str = None, paper_pwc_id: str = None,
        pytorch_hub_url: str = None) -> BenchmarkResult:

    if num_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(num_gpu)))
    else:
        model = model

    if is_cuda:
        model = model.cuda()

    model.eval()

    if not input_transform:
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [109.9, 109.7, 113.8]],
                                         std=[x / 255.0 for x in [50.1, 50.6, 50.8]])
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    test_dataset = datasets.SVHN(data_root, split='test', transform=input_transform, target_transform=target_transform, download=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    criterion = nn.CrossEntropyLoss()

    metrics = get_classification_metrics(model=model, test_loader=test_loader, criterion=criterion, is_cuda=is_cuda)

    print(' * Acc@1 {top1:.3f} Acc@5 {top5:.3f}'.format(top1=metrics['top_1_accuracy'], top5=metrics['top_5_accuracy']))

    return BenchmarkResult(
        task="Image Classification", dataset=test_dataset,
        metrics=metrics,
        pytorch_hub_url=pytorch_hub_url,
        paper_model_name=paper_model_name, paper_arxiv_id=paper_arxiv_id, paper_pwc_id=paper_pwc_id)