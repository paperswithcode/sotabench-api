from sotabenchapi import consts
from sotabenchapi.client import Client
from sotabenchapi.config import Config
import os

def test_run_hash():
    config_path = os.path.expanduser(consts.DEFAULT_CONFIG_PATH)
    config = Config(config_path)
    client = Client(config)

    res = client.get_results_by_run_hash(run_hash="c474595718e06d524fa4eaeba35347181f1fa18b28f123e68eeaeca8c52336aa")

    assert isinstance(res, dict)
    assert res["Top 5 Accuracy"] == 0.9795
