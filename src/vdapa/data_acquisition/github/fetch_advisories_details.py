import requests
from vdapa.config import config
from vdapa.utils import setup_logging

logger = setup_logging("data_acquisition", "advisory_details")


class AdvisoryDetailsFetcher:
    def __init__(self):
        self.api_base_url = f"{config['github']['api_url']}/repos/github/advisory-database/advisories"
        token = config['github'].get('token')
        self.headers = {}
        if token:
            self.headers['Authorization'] = f"token {token}"

    def fetch_advisory_details(self, advisory_id):
        """
        Fetch detailed advisory information from GitHub API.

        Args:
            advisory_id (str): Unique advisory ID (e.g., GHSA-xxxx-xxxx).

        Returns:
            dict: Advisory details JSON.

        Raises:
            requests.RequestException: On request failure.
            ValueError: If advisory_id is invalid.
        """
        if not isinstance(advisory_id, str) or not advisory_id.startswith("GHSA-"):
            raise ValueError("Invalid advisory_id format")

        url = f"{self.api_base_url}/{advisory_id}"
        try:
            logger.info(f"Fetching advisory details for {advisory_id}")
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Failed to fetch advisory {advisory_id}: {e}")
            raise


AdvisoryDetailsFetcher().fetch_advisory_details("GHSA-j6g5-p62x-58hw")  # Example usage, replace with actual ID