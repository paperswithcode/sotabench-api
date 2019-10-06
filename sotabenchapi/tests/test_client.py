import os

import pytest

from sotabenchapi import consts
from sotabenchapi.client import Client
from sotabenchapi.config import Config


@pytest.fixture
def run_hash():
    return "c474595718e06d524fa4eaeba35347181f1fa18b28f123e68eeaeca8c52336aa"


def test_run_hash(run_hash):
    config_path = os.path.expanduser(consts.DEFAULT_CONFIG_PATH)
    config = Config(config_path)
    client = Client(config)

    res = client.get_results_by_run_hash(run_hash=run_hash)

    assert isinstance(res, dict)
    assert res["Top 5 Accuracy"] == 0.9795


def test_check_results(run_hash):
    client = Client.public()

    r = [
        {
            "model": "FixResNeXt-101 32x48d",
            "task": "Image Classification",
            "dataset_name": "ImageNet",
            "results": {
                "Top 1 Accuracy": 0.8636199999999999,
                "Top 5 Accuracy": 0.9795,
            },
            "arxiv_id": "1906.06423",
            "pwc_id": None,
            "pytorch_hub_id": None,
            "paper_results": None,
            "run_hash": run_hash,
        }
    ]

    res = client.check_results(r)
    assert len(res["response"]["errors"]) == 0

    r[0]["task"] = "Make a cup of tea"
    res = client.check_results(r)

    e = res["response"]["errors"][0]
    assert "error" in e
