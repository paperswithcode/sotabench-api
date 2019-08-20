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
        """
        This function returns a BenchmarkResult with only parameters - no evaluation - so we can check the inputs
        to see if sotabench.com will accept them
        :param args: args for the benchmark() method
        :param kwargs: kwargs for the benchmark() method
        :return: BenchmarkResult instance
        """
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
        """
        A regular call to benchmark() - if a SOTABENCH_SERVER environment variable is set then we enforce some
        parameters so it works on the server (e.g. number of gpus, device type, data location)
        :param args: args for the benchmark() method
        :param kwargs: kwargs for the benchmark() method
        :return: BenchmarkResult instance
        """

        check_server = os.environ.get("SOTABENCH_SERVER")

        if check_server == 'true':  # if being run on a server, we enforce some parameters
            kwargs.pop('data_root', None)

            if 'num_gpu' in kwargs:
                if kwargs['num_gpu'] != '1':
                    kwargs['num_gpu'] = 1
                    print('Changing number of GPUs to 1 for sotabench.com server \n')

            if 'device' in kwargs:
                if kwargs['device'] != 'cuda':
                    kwargs['device'] = 'cuda'
                    print('Changing device to cuda for sotabench.com server \n')

        func(*args, **kwargs)

    if check_mode == 'params':
        return param_check_only
    else:
        return regular_evaluation