import json
import os
from typing import Optional, Any


class BenchmarkResult:
    """BenchmarkResult represents the results of a benchmark.

    It takes in inputs from a benchmark evaluation and stores them to a JSON at
    ``evaluation.json``.

    This file is then processed to store and show results on the sotabench
    platform.

    Most of the inputs are optional - so when you create a benchmark, you can
    choose which subset of arguments you want to store (that are relevant for
    your benchmark).

    Arguments:
        model (str): Name of the model, e.g. ``EfficientNet-B0``.
        task (str): String describing a task, e.g. ``Image Classification``.
        dataset (str): String representing the name of a dataset, e.g.
            ``CIFAR-10``.
        results (dict): Dictionary with keys as metric names, e.g.
            ``Top 1 Accuracy``, and values as floats, e.g. ``0.80``.
        config (dict, optional): Dictionary storing user configuration
            arguments (inputs to the evaluation function), e.g. the transforms
            that were passed to the dataset object (resizing, cropping...)
        benchmark (object, optional): Object containing the benchmark data and
            benchmark evaluation method (see torchbench library for an
            example).
        arxiv_id (str), optional): String describing the paper where the model
            comes from, e.g. ``1901.07518``.
        pwc_id (str, optional): Describing the paperswithcode.com page - e.g.:
            ``hybrid-task-cascade-for-instance-segmentation``.
        pytorch_hub_id (str, optional): Describing the location of the PyTorch
            Hub model, e.g.: ``mateuszbuda_brain-segmentation-pytorch_unet``
        paper_results (dict, optional): Dictionary with original results from
            the PAPER, e.g.::

                {
                    'Top 1 Accuracy': 0.543,
                    'Top 5 Accuracy': 0.743
                }

            The metric names should match those used in the existing
            leaderboard.
        run_hash (str): The run_hash that uniquely identifies this run, based
            on results from the first batch. It is used to cache runs so we
            don't have to re-run benchmarks when nothing has changed.
    """

    def __init__(
        self,
        model: str,
        task: str,
        dataset: str,
        results: dict,
        config: Optional[dict] = None,
        benchmark: Optional[Any] = None,
        arxiv_id: Optional[str] = None,
        pwc_id: Optional[str] = None,
        pytorch_hub_id: Optional[str] = None,
        paper_results: Optional[dict] = None,
        run_hash: Optional[str] = None,
    ):

        self.model = model
        self.task = task
        self.dataset = dataset
        self.results = results
        self.config = config
        self.benchmark = benchmark
        self.arxiv_id = arxiv_id
        self.pwc_id = pwc_id
        self.pytorch_hub_id = pytorch_hub_id
        self.paper_results = paper_results
        self.run_hash = run_hash

        if isinstance(self.dataset, str):
            self.dataset_name = self.dataset
            self.dataset_obj = None
        else:
            self.dataset_name = type(self.dataset).__name__
            self.dataset_obj = self.dataset

        self.create_json = (
            True if os.environ.get("SOTABENCH_STORE_FILENAME") else False
        )

        self.to_dict()

    def to_dict(self) -> dict:
        """Performs evaluation and return build results.

        Performs evaluation using a benchmark function and returns a
        dictionary of the build results.

        If an environmental variable is set
        (``SOTABENCH_STORE_RESULTS == True``) then will also save a JSON called
        ``evaluation.json``

        Returns:
            dict: A dictionary containing results
        """

        build_dict = {
            "model": self.model,
            "task": self.task,
            "dataset_name": self.dataset_name,
            "results": self.results,
            "arxiv_id": self.arxiv_id,
            "pwc_id": self.pwc_id,
            "pytorch_hub_id": self.pytorch_hub_id,
            "paper_results": self.paper_results,
            "run_hash": self.run_hash
        }

        if self.create_json:
            file_name = os.environ.get("SOTABENCH_STORE_FILENAME")

            if not os.path.isfile(file_name):
                models_dict = [build_dict]
            else:
                models_dict = json.load(open(file_name))
                models_dict.append(build_dict)

            with open(file_name, "w") as f:
                json.dump(models_dict, f, ensure_ascii=False)

        return build_dict
