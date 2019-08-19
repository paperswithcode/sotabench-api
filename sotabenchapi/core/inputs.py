import os

from sotabenchapi.core.results import BenchmarkResult


def check_inputs(func):
    """
    A decorator for checking inputs to a benchmark method.
    :param func: a benchmark method, e.g. ImageNet.benchmark(...)
    :return: if regular evaluation, then func; if parameter check only then skips evaluation and returns inputs
    in a BenchMarkResult object so they can be checked for correctness (e.g. if model name is correct)
    """
    check_mode = os.environ.get("SOTABENCH_CHECK")

    def param_check_only(*args, **kwargs):
        BenchmarkResult(
            task=args[0].task,
            config=None,
            dataset=args[0].dataset.__name__,
            results={},
            pytorch_hub_id=None,
            model=None if 'paper_model_name' not in kwargs else kwargs['paper_model_name'],
            arxiv_id=None if 'paper_arxiv_id' not in kwargs else kwargs['paper_arxiv_id'],
            pwc_id=None if 'paper_pwc_id' not in kwargs else kwargs['paper_pwc_id'],
            paper_results={},
            run_hash=None)

    def regular_evaluation(*args, **kwargs):
        func(*args, **kwargs)

    if check_mode == 'params':
        return param_check_only
    else:
        return regular_evaluation