import json
from typing import Union

import torch.utils.data as data


class BenchmarkResult:

    def __init__(self, task: str, dataset: Union[str, data.Dataset], metrics: dict):
        """
        A class for holding benchmark results for a model

        :param task: string describing a task, e.g. "Image Classification"
        :param dataset: either a string for a name, e.g. "CIFAR-10", or a torch.data.Dataset object
        :param metrics: dict with keys as metric names, e.g. 'top_1_accuracy', and values as floats, e.g. 0.80
        """
        self.task = task
        self.metrics = metrics

        if isinstance(dataset, str):
            self.dataset_name = dataset
            self.dataset_obj = None
        else:
            self.dataset_name = type(dataset).__name__
            self.dataset_obj = dataset

def evaluate(benchmark_function):
    """
    Performs evaluation using a benchmark function and saves results to a JSON

    TODO: work out functionality

    :param benchmark_function: a benchmark function that returns a BenchmarkResult object
    :return: process_function: a function processing the benchmark function as an input
    """

    def process_function(*args, **kwargs):
        result = benchmark_function(*args, **kwargs)
        result_dict = {'metrics': result.metrics, 'task': result.task, 'dataset': result.dataset_name}

        with open('evaluation.json', 'w') as f:
            json.dump(result_dict, f, ensure_ascii=False)

    return process_function
