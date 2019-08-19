import os


def in_check_mode():
    """Return True/False if we are running the library in check mode.

    In check mode we perform a dry run to make sure the benchmarks are
    going to run and the parameters (such as paper IDs and model names)
    have been specified correctly.

    Returns:
        bool: True if we are in check mode.
    """
    check_mode = os.environ.get("SOTABENCH_CHECK")

    if check_mode == "full" or check_mode == "params":
        return True
    else:
        return False


def get_check_mode_type():
    """Get the type of checking we are doing.

    Returns:
         str: Either "full" for a full check including running the benchmark on
            the first batch, or "params" which only check input parameters of
            the benchmark function.
    """
    check_mode = os.environ.get("SOTABENCH_CHECK")

    if not in_check_mode():
        return None
    elif check_mode in ["full", "params"]:
        return check_mode
    else:
        return "n/a"
