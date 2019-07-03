from torch.utils.data import DataLoader

from sotabench.core import BenchmarkResult
from sotabench.datasets import ADE20K
from sotabench.utils import send_model_to_device

from .transforms import Normalize, ToTensor, Compose
from .utils import collate_fn, evaluate_segmentation


class ADE20K:

    dataset = ADE20K
    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transforms = Compose([ToTensor(), normalize])

    @classmethod
    def benchmark(cls, model, input_transform=None, target_transform=None, transforms=None, model_output_transform=None,
                  device: str = 'cuda', data_root: str = './.data/vision/ade20k', num_workers: int = 4, batch_size: int = 32,
                  num_gpu: int = 1, paper_model_name: str = None, paper_arxiv_id: str = None, paper_pwc_id: str = None,
                  pytorch_hub_url: str = None) -> BenchmarkResult:

        config = locals()
        model, device = send_model_to_device(model, device=device, num_gpu=num_gpu)
        model.eval()

        if not input_transform or target_transform or transforms:
            transforms = cls.transforms

        test_dataset = cls.dataset(root=data_root, split='val', transform=input_transform,
                                   target_transform=target_transform, transforms=transforms)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,
                                 collate_fn=collate_fn)
        test_loader.no_classes = 150  # Number of classes for ADE20K
        test_results = evaluate_segmentation(model=model, model_output_transform=model_output_transform, test_loader=test_loader, device=device)

        print(test_results)

        return BenchmarkResult(task="Semantic Segmentation", benchmark=cls, config=config, dataset=test_dataset,
                               results=test_results, pytorch_hub_url=pytorch_hub_url, paper_model_name=paper_model_name,
                               paper_arxiv_id=paper_arxiv_id, paper_pwc_id=paper_pwc_id)
