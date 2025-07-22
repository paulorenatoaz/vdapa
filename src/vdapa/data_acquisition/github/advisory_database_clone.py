import subprocess
from pathlib import Path
from vdapa.config import config, BASE_DIR
from vdapa.utils import setup_logging

"""
Module to clone or update the GitHub Advisory Database repository locally.

This module manages a local copy of the official GitHub advisory-database repository,
which contains JSON files with detailed security advisories.

Functions:
- clone_or_update_repo: Clone the repo if not present, or pull latest changes.
- run: Entry point for command-line or script execution.

The repository is stored under a path defined in the project configuration.
"""

logger = setup_logging("data_acquisition", "advisory_database_sparse")

REPO_URL = "https://github.com/github/advisory-database.git"
LOCAL_PATH = BASE_DIR / Path(config['paths']['raw_data']) / "advisory-database"


def clone_or_update_repo(local_path: Path = LOCAL_PATH):
    """
    Clone the entire advisory-database repository if it does not exist locally,
    or update it by pulling the latest changes if it does.

    Args:
        local_path (Path): The local directory path where the repository will be cloned or updated.

    Raises:
        ValueError: If `local_path` is not a Path instance.
        subprocess.CalledProcessError: If git commands fail during clone or pull.
    """
    if not isinstance(local_path, Path):
        raise ValueError(f"local_path must be a pathlib.Path instance, got {type(local_path)}")

    if not local_path.exists():
        logger.info(f"Cloning advisory-database repository into {local_path}")
        subprocess.run(["git", "clone", REPO_URL, str(local_path), "--verbose" ], check=True)
    else:
        logger.info(f"Updating advisory-database repository at {local_path}")
        subprocess.run(["git", "-C", str(local_path), "pull", "origin", "main", "--verbose"], check=True)


def run():
    """
    Entry point to clone or update the advisory-database repository.
    Logs the process and handles exceptions by logging and re-raising them.
    """
    try:
        clone_or_update_repo()
    except Exception as e:
        logger.error(f"Failed to clone or update advisory-database repository: {e}")
        raise


if __name__ == "__main__":
    run()
