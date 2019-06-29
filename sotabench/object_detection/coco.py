import os
from torch.utils.data import DataLoader

from sotabench.core import BenchmarkResult
from sotabench.datasets import CocoDetection
from sotabench.utils import send_model_to_device

from .transforms import Compose, ConvertCocoPolysToMask, ToTensor
from .utils import collate_fn, evaluate_detection_coco


class COCO:

    dataset = CocoDetection
    transforms = Compose([ConvertCocoPolysToMask(), ToTensor()])

    @classmethod
    def benchmark(cls, model, dataset_year='2017', input_transform=None, target_transform=None, transforms=None, model_output_transform=None,
                  device: str = 'cuda', data_root: str = './.data/vision/coco', num_workers: int = 4, batch_size: int = 1,
                  num_gpu: int = 1, paper_model_name: str = None, paper_arxiv_id: str = None, paper_pwc_id: str = None,
                  pytorch_hub_url: str = None) -> BenchmarkResult:

        config = locals()
        model, device = send_model_to_device(model, device=device, num_gpu=num_gpu)
        model.eval()

        if not input_transform or target_transform or transforms:
            transforms = cls.transforms

        test_dataset = cls.dataset(root=os.path.join(data_root, 'val%s' % dataset_year),
                                   annFile=os.path.join(data_root, 'annotations/instances_val%s.json' % dataset_year),
                                   transform=input_transform, target_transform=target_transform, transforms=transforms)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,
                                 collate_fn=collate_fn)
        test_loader.no_classes = 91  # Number of classes for COCO Detection
        test_results = evaluate_detection_coco(model=model, model_output_transform=model_output_transform, test_loader=test_loader, device=device)

        print(test_results)

        return BenchmarkResult(task="Object Detection", benchmark=cls, config=config, dataset=test_dataset,
                               results=test_results, pytorch_hub_url=pytorch_hub_url, paper_model_name=paper_model_name,
                               paper_arxiv_id=paper_arxiv_id, paper_pwc_id=paper_pwc_id)
