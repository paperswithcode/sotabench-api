import json
import os
from typing import Union

import torch.utils.data as data


class BenchmarkResult:

    def __init__(self, task: str,
                 benchmark,
                 config: dict,
                 dataset: Union[str, data.Dataset],
                 results: dict,
                 paper_model_name: str = None,
                 paper_arxiv_id: str = None,
                 paper_pwc_id: str = None,
                 pytorch_hub_url: str = None):
        """
        A class for holding benchmark results for a model

        :param task: string describing a task, e.g. "Image Classification"
        :param benchmark: Object containing the benchmark data and benchmark evaluation method
        :param config: dict containing inputs to the evaluation function
        :param dataset: either a string for a name, e.g. "CIFAR-10", or a torch.data.Dataset object
        :param results: dict with keys as metric names, e.g. 'Top 1 Accuracy', and values as floats, e.g. 0.80
        :param: paper_model_name: (optional) Name of the model that comes from the paper, e.g. 'BERT small'
        :param: paper_arxiv_id: (optional) Representing the paper where the model comes from, e.g. '1901.07518'
        :param: paper_pwc_id: (optional) Representing the location of the PWC paper, e.g.: 'hybrid-task-cascade-for-instance-segmentation'
        :param: pytorch_hub_id: (optional) Representing the location of the PyTorch Hub model, e.g.: 'mateuszbuda_brain-segmentation-pytorch_unet'

        """
        self.task = task
        self.benchmark = benchmark
        self.config = config
        self.results = results
        self.paper_model_name = paper_model_name
        self.paper_arxiv_id = paper_arxiv_id
        self.paper_pwc_id = paper_pwc_id
        self.pytorch_hub_url = pytorch_hub_url
        self.dataset = dataset

        if isinstance(self.dataset, str):
            self.dataset_name = self.dataset
            self.dataset_obj = None
        else:
            self.dataset_name = type(self.dataset).__name__
            self.dataset_obj = self.dataset

        self.create_json = True if os.environ.get('SOTABENCH_STORE_FILENAME') else False

        self.evaluate()

    def evaluate(self):
        """
        Performs evaluation using a benchmark function and returns a dictionary of the build results. If an environmental
        variable is set (SOTABENCH_STORE_RESULTS == 'True') then will also save a JSON called evaluation.json

        :return: build_dict: a dictionary containing results
        """

        build_dict = {
            'task': self.task,
            'results': self.results,
            'dataset': self.dataset_name,
            'paper_model_name': self.paper_model_name,
            'paper_arxiv_id': self.paper_arxiv_id,
            'paper_pwc_id': self.paper_pwc_id,
            'pytorch_hub_url': self.pytorch_hub_url}

        if self.create_json:
            file_name = os.environ.get('SOTABENCH_STORE_FILENAME')

            if not os.path.isfile(file_name):
                models_dict = [build_dict]
            else:
                models_dict = json.load(open(file_name))
                models_dict.append(build_dict)

            with open(file_name, 'w') as f:
                json.dump(models_dict, f, ensure_ascii=False)

        return build_dict
