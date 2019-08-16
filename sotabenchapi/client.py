from typing import List

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
        response = self.http.post("check/run-hashes/", data={"hashes": hashes})
        return response
