import json
import os
from typing import Union

import torch.utils.data as data


class BenchmarkResult:

    def __init__(self, task: str,
                 dataset: Union[str, data.Dataset],
                 metrics: dict,
                 paper_model_name: str = None,
                 paper_arxiv_id: str = None,
                 paper_pwc_id: str = None,
                 pytorch_hub_url: str = None):
        """
        A class for holding benchmark results for a model

        :param task: string describing a task, e.g. "Image Classification"
        :param dataset: either a string for a name, e.g. "CIFAR-10", or a torch.data.Dataset object
        :param metrics: dict with keys as metric names, e.g. 'Top 1 Accuracy', and values as floats, e.g. 0.80
        :param: paper_model_name: (optional) Name of the model that comes from the paper, e.g. 'BERT small'
        :param: paper_arxiv_id: (optional) Representing the paper where the model comes from, e.g. '1901.07518'
        :param: paper_pwc_id: (optional) Representing the location of the PWC paper, e.g.: 'hybrid-task-cascade-for-instance-segmentation'
        :param: pytorch_hub_id: (optional) Representing the location of the PyTorch Hub model, e.g.: 'mateuszbuda_brain-segmentation-pytorch_unet'

        """
        self.task = task
        self.metrics = metrics
        self.paper_model_name = paper_model_name
        self.paper_arxiv_id = paper_arxiv_id
        self.paper_pwc_id = paper_pwc_id
        self.pytorch_hub_url = pytorch_hub_url

        if isinstance(dataset, str):
            self.dataset_name = dataset
            self.dataset_obj = None
        else:
            self.dataset_name = type(dataset).__name__
            self.dataset_obj = dataset

def evaluate(benchmark_function):
    """
    Performs evaluation using a benchmark function and saves results to a JSON

    :param benchmark_function: a benchmark function that returns a BenchmarkResult object
    :return: process_function: a function processing the benchmark function as an input
    """

    def process_function(*args, **kwargs):
        result = benchmark_function(*args, **kwargs)
        result_dict = {
            'metrics': result.metrics,
            'task': result.task,
            'dataset': result.dataset_name,
            'paper_model_name': result.paper_model_name,
            'paper_arxiv_id': result.paper_arxiv_id,
            'paper_pwc_id': result.paper_pwc_id,
            'pytorch_hub_url': result.pytorch_hub_url}

        if not os.path.isfile('evaluation.json'):
            models_dict = [result_dict]
        else:
            models_dict = json.load(open('evaluation.json'))
            models_dict.append(result_dict)

        with open('evaluation.json', 'w') as f:
            json.dump(models_dict, f, ensure_ascii=False)

    return process_function
