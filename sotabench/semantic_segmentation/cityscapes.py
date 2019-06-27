from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from sotabench.core import BenchmarkResult
from sotabench.datasets.cityscapes import Cityscapes
from sotabench.utils import send_model_to_device

from .utils import evaluate_segmentation, FlipChannels, CityscapesMaskConversion


class Cityscapes:

    dataset = Cityscapes
    normalize = transforms.Normalize(*([103.939, 116.779, 123.68], [1.0, 1.0, 1.0]))
    input_transform = transforms.Compose([FlipChannels(), transforms.ToTensor(), transforms.Lambda(lambda x: x.mul_(255)), normalize])
    target_transform = transforms.Compose([CityscapesMaskConversion(ignore_index=255)])

    @classmethod
    def benchmark(cls, model, input_transform=None, target_transform=None, transforms=None, model_output_transform=None,
                  device: str = 'cuda', data_root: str = './.data', num_workers: int = 4, batch_size: int = 128,
                  num_gpu: int = 1, paper_model_name: str = None, paper_arxiv_id: str = None, paper_pwc_id: str = None,
                  pytorch_hub_url: str = None) -> BenchmarkResult:

        config = locals()
        model, device = send_model_to_device(model, device=device, num_gpu=num_gpu)
        model.eval()

        if not input_transform or target_transform or transforms:
            input_transform = cls.input_transform
            target_transform = cls.target_transform

        test_dataset = cls.dataset(root=data_root, split='val', target_type='semantic', transform=input_transform,
                                   target_transform=target_transform, transforms=transforms)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        test_loader.no_classes = 19  # Number of classes for Cityscapes
        test_results = evaluate_segmentation(model=model, model_output_transform=model_output_transform, test_loader=test_loader, device=device)

        print(test_results)

        return BenchmarkResult(task="Semantic Segmentation", benchmark=cls, config=config, dataset=test_dataset,
                               results=test_results, pytorch_hub_url=pytorch_hub_url, paper_model_name=paper_model_name,
                               paper_arxiv_id=paper_arxiv_id, paper_pwc_id=paper_pwc_id)