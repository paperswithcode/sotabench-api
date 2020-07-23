import os
from typing import List, Optional

from sotabenchapi import uploader
from sotabenchapi.config import Config
from sotabenchapi.http import HttpClient


class Client(object):
    """NewReleases client.

    Args:
        config (sotabenchapi.config.Config): Instance of the sotabenchapi
            configuration.
    """

    def __init__(self, config: Config):
        self.config = config
        self.http = HttpClient(url=config.url, token=config.token)

    @classmethod
    def public(cls) -> "Client":
        """Get the public access sotabench client.

        Returns:
            Client: A client instance that can be used to make API
                requests to sotabench.com.
        """
        config = Config(None)
        return Client(config)

    def login(self, username: str, password: str) -> str:
        """Obtain authentication token.

        Args:
            username (str): SotaBench username.
            password (str): SotaBench password.

        Returns:
            str: Authentication token.
        """
        response = self.http.post(
            "auth/token/", data={"username": username, "password": password}
        )
        return response["token"]

    # Check
    def check_run_hashes(self, hashes: List[str]) -> dict:
        """Check if the hash exist in the database.

        Args:
            hashes (list of str): List of run hashes.

        Returns:
            dict: Dictionary of ``{hash: True/False}`` pairs. ``True``
                represents an existing hash, ``False`` a non existing.
        """
        response = self.http.post("check/run-hashes/", data={"hashes": hashes})
        return response

    def get_results_by_run_hash(self, run_hash: str) -> dict:
        """Get cached results by run_hash.

        Args:
            run_hash (str): SHA256 run_hash that identifies the run

        Returns:
            dict: A dictionary of results, e.g::

                {
                    "Top 1 Accuracy": 0.85,
                    "Top 5 Accuracy": 0.90
                }
        """

        response = self.http.get(
            "check/get_results_by_hash/", params={"run_hash": run_hash}
        )
        return response

    def check_results(self, results: List[dict]) -> List[dict]:
        """Check if the results would be accepted by sotabench.com.

        Args:
            results: A list of results dictionaries (ie same format as
                sotabench-results.json.

        Returns:
            List[dict]: A list of dictionaries highlighting any errors with the
                submitted results.
        """

        return self.http.post("check/results/", data={"results": results})

    # Repository
    def repository_list(self, username: Optional[str] = None):
        """List repositories.

        Optionally filter by repository owner.
        """
        if username is None:
            return self.http.get("repositories/")
        else:
            return self.http.get(f"repositories/{username}/")

    def repository_get(self, repository: str):
        """Get repository.

        Args:
            repository (str): Repository in ``owner/project`` format.
        """
        return self.http.get(f"repositories/{repository}/")

    def repository_update(self, repository: str, build_enabled: bool):
        """Update build_enabled flag.

        Args:
            repository (str): Repository in ``owner/project`` format.
            build_enabled (bool): Should the build be enabled or not.
        """
        return self.http.patch(
            f"repositories/{repository}/",
            data={"build_enabled": build_enabled},
        )

    # Dataset
    def dataset_list(self, repository: str):
        """List all datasets for the given repository.

        Args:
            repository (str): Repository in ``owner/project`` format.
        """
        return self.http.get(f"repositories/{repository}/datasets/")

    def dataset_upload(
        self,
        dataset: str,
        repository: str,
        path: Optional[str] = None,
        part_size: Optional[int] = None,
    ):
        """Upload dataset for a repository.

        Args:
            dataset (str): Path to a dataset file.
            repository (str): repository slug.
            path (str): Path under the .data folder where the dataset should be
                downloaded. Default: `basename(dataset).
            part_size (int, optional): Optional user defined part size.
        """

        uploader.multipart_upload(
            http=self.http,
            filename=dataset,
            repository=repository,
            path=path or os.path.basename(dataset),
            part_size=part_size,
        )

    def dataset_get(self, repository: str, dataset: str):
        """Get dataset.

        Args:
            repository (str): Repository in ``owner/project`` format.
            dataset (str): Path to the dataset.
        """
        return self.http.get(
            f"repositories/{repository}/datasets/?dataset={dataset}"
        )

    def dataset_delete(self, repository: str, dataset: str):
        """Update build_enabled flag.

        Args:
            repository (str): Repository in ``owner/project`` format.
            dataset (str): Path to the dataset.
        """
        return self.http.delete(
            f"repositories/{repository}/datasets/", data={"dataset": dataset},
        )

    # Build
    def build_start(self, repository: str):
        """Initiate repository build.

        Args:
            repository (str): Repository in ``owner/project`` format.
        """
        return self.http.post(f"builds/{repository}/")

    def build_list(self, repository: str):
        """List builds for a given repository.

        Args:
            repository (str): Repository in ``owner/project`` format.
        """
        return self.http.get(f"builds/{repository}/")

    def build_get(self, repository: str, run_number: int):
        """Get build.

        Args:
            repository (str): Repository in ``owner/project`` format.
            run_number (int): Run number of the build.
        """
        return self.http.get(f"builds/{repository}/{run_number}/")

    # Benchmark
    def benchmark_list(self):
        """List users benchmarks."""
        return self.http.get("benchmarks/")

    def benchmark_get(self, benchmark: str):
        """Get benchmark.

        Args:
            benchmark (str): Benchmark slug.
        """
        return self.http.get(f"benchmarks/{benchmark}/")
