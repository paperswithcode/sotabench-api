from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from sotabench.core import BenchmarkResult
from sotabench.utils import send_model_to_device

from .utils import evaluate_image_generation_gan


class ImageNet:

    dataset = datasets.ImageNet
    norm_mean = [2 * p - 1 for p in [0.485, 0.456, 0.406]]
    norm_std = [2 * p for p in [0.229, 0.224, 0.225]]
    normalize = transforms.Normalize(mean=norm_mean, std=norm_std)
    input_transform = transforms.Compose([transforms.ToTensor(), normalize])

    @classmethod
    def benchmark(cls, model, input_transform=None, target_transform=None, model_output_transform=None,
                  device: str = 'cuda', data_root: str = './.data/vision/imagenet', num_workers: int = 4,
                  batch_size: int = 8, num_gpu: int = 1, paper_model_name: str = None, paper_arxiv_id: str = None,
                  paper_pwc_id: str = None, pytorch_hub_url: str = None) -> BenchmarkResult:

        config = locals()
        model, device = send_model_to_device(model, device=device, num_gpu=num_gpu)

        if hasattr(model, 'eval'):
            model.eval()

        if not input_transform:
            input_transform = cls.input_transform

        test_dataset = cls.dataset(data_root, train=False, transform=input_transform, target_transform=target_transform, download=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        test_results = evaluate_image_generation_gan(model=model, model_output_transform=model_output_transform, test_loader=test_loader, device=device)

        print(test_results)

        return BenchmarkResult(task="Image Generation", benchmark=cls, config=config, dataset=test_dataset,
                               results=test_results, pytorch_hub_url=pytorch_hub_url, paper_model_name=paper_model_name,
                               paper_arxiv_id=paper_arxiv_id, paper_pwc_id=paper_pwc_id)
