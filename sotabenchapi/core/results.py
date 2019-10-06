import io
import os
import json
from typing import Optional

import click

from sotabenchapi.client import Client
from sotabenchapi.check import in_check_mode, get_check_mode_type


class BenchmarkResult:
    """Class encapsulates data for the results of a model on a benchmark.

    It also provides methods for serialising that data and checking the
    parameters with the sotabench.com resource.

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
        speed_mem_metrics (dict, optional): Dictionary with information about speed,
            memory usage and other measures.
        model_description (str, optional): Optional description of the model.
        config (dict, optional): Dictionary storing user configuration
            arguments (inputs to the evaluation function), e.g. the transforms
            that were passed to the dataset object (resizing, cropping...)
        arxiv_id (str, optional): String describing the paper where the model
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
        speed_mem_metrics: Optional[dict] = None,
        model_description: Optional[str] = None,
        config: Optional[dict] = None,
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
        self.speed_mem_metrics = speed_mem_metrics
        self.model_description = model_description
        self.config = config
        self.arxiv_id = arxiv_id
        self.pwc_id = pwc_id
        self.pytorch_hub_id = pytorch_hub_id
        self.paper_results = paper_results
        self.run_hash = run_hash

        self.create_json = (
            True if os.environ.get("SOTABENCH_STORE_FILENAME") else False
        )

        self.in_check_mode = in_check_mode()
        self.check_mode_type = get_check_mode_type()

        self.to_dict()

    def to_dict(self) -> dict:
        """Serialises the benchmark result data.

        If an environmental variable is set, e.g.
        (``SOTABENCH_STORE_FILENAME == 'evaluation.json'``) then will also save
        a JSON called ``evaluation.json``

        The method also checks for errors with the sotabench.com server if in
        check mode.

        Returns:
            dict: A dictionary containing results
        """

        build_dict = {
            "model": self.model.encode("ascii", "ignore").decode("ascii"),
            "model_description": self.model_description,
            "task": self.task,
            "dataset_name": self.dataset,
            "results": self.results,
            "speed_mem_metrics": self.speed_mem_metrics,
            "arxiv_id": self.arxiv_id,
            "pwc_id": self.pwc_id,
            "pytorch_hub_id": self.pytorch_hub_id,
            "paper_results": self.paper_results,
            "run_hash": self.run_hash,
        }

        if self.in_check_mode:
            client = Client.public()
            r = client.check_results([build_dict])
            errors = r["response"]["errors"]
            click.secho("\n---\n", fg="white")
            print("Model: {name}\n".format(name=build_dict["model"]))
            if errors:
                click.secho("Error while checking:\n", fg="red")
                for error_dict in errors:
                    print(error_dict["error"])
            else:
                click.secho("No errors detected, looks good!", fg="green")
            click.secho("\n---\n", fg="white")
        elif self.create_json:
            file_name = os.environ.get("SOTABENCH_STORE_FILENAME")

            if not os.path.isfile(file_name):
                models_dict = [build_dict]
            else:
                with io.open(file_name) as f:
                    models_dict = json.load(f)
                models_dict.append(build_dict)

            with io.open(file_name, "w") as f:
                json.dump(models_dict, f, ensure_ascii=False)

        return build_dict
