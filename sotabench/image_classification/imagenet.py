from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from sotabench.core import BenchmarkResult
from sotabench.utils import send_model_to_device

from .utils import evaluate_classification


class ImageNet:

    dataset = datasets.ImageNet
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    input_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                                          transforms.ToTensor(), normalize])

    @classmethod
    def benchmark(cls, model, input_transform=None, target_transform=None, model_output_transform=None,
                  device: str = 'cuda', data_root: str = './.data/vision/imagenet', num_workers: int = 4, batch_size: int = 128,
                  num_gpu: int = 1, paper_model_name: str = None, paper_arxiv_id: str = None, paper_pwc_id: str = None,
                  pytorch_hub_url: str = None) -> BenchmarkResult:

        config = locals()
        model, device = send_model_to_device(model, device=device, num_gpu=num_gpu)
        model.eval()

        if not input_transform:
            input_transform = cls.input_transform

        try:
            test_dataset = cls.dataset(data_root, split='val', transform=input_transform, target_transform=target_transform, download=True)
        except:
            test_dataset = cls.dataset(data_root, split='val', transform=input_transform, target_transform=target_transform, download=False)

        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        test_results = evaluate_classification(model=model, model_output_transform=model_output_transform, test_loader=test_loader, device=device)

        print(' * Acc@1 {top1:.3f} Acc@5 {top5:.3f}'.format(top1=test_results['Top 1 Accuracy'], top5=test_results['Top 5 Accuracy']))

        return BenchmarkResult(task="Image Classification", benchmark=cls, config=config, dataset=test_dataset,
                               results=test_results, pytorch_hub_url=pytorch_hub_url, paper_model_name=paper_model_name,
                               paper_arxiv_id=paper_arxiv_id, paper_pwc_id=paper_pwc_id)
