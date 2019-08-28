import os

from sotabenchapi.core.results import BenchmarkResult


def check_inputs(func):
    """A decorator for checking inputs to a benchmark method.

    Args:
        func (callable): a benchmark method, e.g. ImageNet.benchmark(...)

    Returns:
        callable: If regular evaluation, then func; if parameter check only
            then skips evaluation and returns inputs in a BenchMarkResult
            object so they can be checked for correctness (e.g. if model name
            is correct).
    """
    check_mode = os.environ.get("SOTABENCH_CHECK")

    def param_check_only(*args, **kwargs):
        """Return a BenchmarkResult with only parameters.

        No evaluation - so we can check the inputs to see if sotabench.com will
        accept them.

        Args:
            args: args for the benchmark() method.
            kwargs: kwargs for the benchmark() method.

        Returns:
            BenchmarkResult: BenchmarkResult instance.
        """
        BenchmarkResult(
            task=args[0].task,
            config=None,
            dataset=args[0].dataset.__name__,
            results={},
            pytorch_hub_id=None,
            model=kwargs.get("paper_model_name", None),
            model_description=kwargs.get("model_description", ""),
            arxiv_id=kwargs.get("paper_arxiv_id", None),
            pwc_id=kwargs.get("paper_pwc_id", None),
            paper_results={},
            run_hash=None,
        )

    def regular_evaluation(*args, **kwargs):
        """A regular call to benchmark().

        If a SOTABENCH_SERVER environment variable is set then we enforce some
        parameters so it works on the server (e.g. number of gpus, device type,
        data location).

        Args:
            args: args for the benchmark() method.
            kwargs: kwargs for the benchmark() method.

        Returns:
            BenchmarkResult: BenchmarkResult instance.
        """

        check_server = os.environ.get("SOTABENCH_SERVER")

        if (
            check_server == "true"
        ):  # if being run on a server, we enforce some parameters
            kwargs.pop("data_root", None)

            if "num_gpu" in kwargs:
                if kwargs["num_gpu"] != "1":
                    kwargs["num_gpu"] = 1
                    print(
                        "Changing number of GPUs to 1 for sotabench.com "
                        "server \n"
                    )

            if "device" in kwargs:
                if kwargs["device"] != "cuda":
                    kwargs["device"] = "cuda"
                    print(
                        "Changing device to cuda for sotabench.com server \n"
                    )

        func(*args, **kwargs)

    if check_mode == "params":
        return param_check_only
    else:
        return regular_evaluation
