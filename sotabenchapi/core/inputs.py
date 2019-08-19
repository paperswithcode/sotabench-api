import os

from sotabenchapi.core.results import BenchmarkResult


def check_inputs(func):

    check_mode = os.environ.get("SOTABENCH_CHECK")

    def param_check_only(*args, **kwargs):
        BenchmarkResult(
            task=args.cls.task,
            benchmark=args.cls,
            config=None,
            dataset=args.cls.dataset.__name__,
            results=None,
            pytorch_hub_id=None,
            model=args.paper_model_name,
            arxiv_id=args.paper_arxiv_id,
            pwc_id=args.paper_pwc_id,
            paper_results={},
            run_hash=None)

    def regular_evaluation(*args, **kwargs):
        func(*args, **kwargs)

    if check_mode == 'params':
        return param_check_only
    else:
        return regular_evaluation