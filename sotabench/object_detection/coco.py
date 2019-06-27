import torch
import torchvision

from sotabench.datasets.coco import COCODataset
from sotabench.core import BenchmarkResult, evaluate


@evaluate
def benchmark(
        model,
        input_transform=None, target_transform=None, transforms=None,
        prediction_transform=None, model_output_transform=None,
        is_cuda: bool = True,
        data_root: str = './.data',
        num_workers: int = 4, batch_size: int = 8, num_gpu: int = 1,
        paper_model_name: str = None, paper_arxiv_id: str = None, paper_pwc_id: str = None,
        pytorch_hub_url: str = None) -> BenchmarkResult:

    if num_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(num_gpu)))
    else:
        model = model

    if is_cuda:
        model = model.cuda()

    model.eval()

    if not input_transform or target_transform or transforms:
        mean_std = ([103.939, 116.779, 123.68], [1.0, 1.0, 1.0])

        input_transform = torchvision.transforms.Compose([
            FlipChannels(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(lambda x: x.mul_(255)),
            torchvision.transforms.Normalize(*mean_std)
        ])

        target_transform = torchvision.transforms.Compose([
            CityscapesMaskConversion(ignore_index=255)])

    test_dataset = Cityscapes(root=data_root, split='val', target_type='semantic', transform=input_transform,
                                       target_transform=target_transform, transforms=transforms)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader.no_classes = 19 # Number of classes for Cityscapes

    metrics = get_segmentation_metrics(model=model, model_output_transform=model_output_transform, test_loader=test_loader, is_cuda=is_cuda)

    print(metrics)

    return BenchmarkResult(
        task="Semantic Segmentation", dataset=test_dataset,
        metrics=metrics,
        pytorch_hub_url=pytorch_hub_url,
        paper_model_name=paper_model_name, paper_arxiv_id=paper_arxiv_id, paper_pwc_id=paper_pwc_id)

