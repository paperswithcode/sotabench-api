from typing import List, Optional

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

    def check_run_hashes(self, hashes: List[str]) -> dict:
        """Check if the hash exist in the database.

        Args:
            hashes (list of str): List of run hashes.

        Returns:
            dict: Dictionary of ``{hash: True/False}`` pairs. ``True``
                represents an existing hash, ``False`` a non existing.
        """
        return self.http.post("check/run-hashes/", data={"hashes": hashes})

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
