"""Utility helpers for downloads and checksum validation."""

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter, Retry
from tqdm import tqdm

# Configure logger

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@dataclass(frozen=True)
class RetryConfig:
    """Retry configuration for HTTP downloads."""

    timeout: int | None = 30
    retries: int = 3
    backoff_factor: float = 0.3


def _build_retry_session(config: RetryConfig) -> requests.Session:
    """Create a requests session configured with retry behavior."""
    session = requests.Session()
    retry_strategy = Retry(
        total=config.retries,
        backoff_factor=config.backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _download_to_path(
    response: requests.Response,
    destination: Path,
    block_size: int = 8192,
) -> tuple[int, int]:
    """Write a streaming HTTP response to disk and return expected/actual bytes."""
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    progress_bar = tqdm(
        total=total_size_in_bytes,
        unit="iB",
        unit_scale=True,
        desc=destination.name,
    )
    with destination.open("wb") as output_file:
        for chunk in response.iter_content(chunk_size=block_size):
            if chunk:
                output_file.write(chunk)
                progress_bar.update(len(chunk))
    progress_bar.close()
    return total_size_in_bytes, progress_bar.n


def download_file(
    url: str,
    path: str,
    makedirs: bool = False,
    retry_config: RetryConfig | None = None,
) -> None:
    """
    Download a file from a URL to a local path with retry and progress indication.

    Args:
        url (str): The URL to download the file from.
        path (str): The destination file path where the downloaded file will be saved.
        makedirs (bool, optional): If True, creates parent directories when
            needed. Defaults to False.
        retry_config (RetryConfig | None, optional): HTTP retry behavior.

    Raises:
        requests.HTTPError: If the HTTP request returned an unsuccessful status code.
        IOError: If the downloaded file size does not match the expected size.
        requests.RequestException: For network and request errors.
        OSError: For file system and downloaded-size mismatch errors.
    """
    config = retry_config or RetryConfig()
    destination = Path(path)

    if makedirs:
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            logger.debug("Created directories up to: %s", destination.parent)
        except OSError as err:
            logger.error("Failed to create directories for %s: %s", destination, err)
            raise
    session = _build_retry_session(config)

    try:
        logger.info("Starting download from %s to %s", url, destination)
        with session.get(url, stream=True, timeout=config.timeout) as response:
            response.raise_for_status()
            expected_size, written_size = _download_to_path(response, destination)
        if expected_size not in (0, written_size):
            error_msg = f"Downloaded size mismatch: {written_size} != {expected_size}"
            logger.error(error_msg)
            raise OSError(error_msg)
        logger.info("Successfully downloaded %s to %s", url, destination)
    except requests.RequestException as err:
        logger.error("Request error occurred while downloading %s: %s", url, err)
        raise
    except OSError as err:
        logger.error("An error occurred while writing %s: %s", destination, err)
        raise


def check_file_md5(path: str, target_md5: str, chunk_size: int = 8192) -> bool:
    """
    Check if the MD5 checksum of a file matches the target checksum.

    Args:
        path (str): Path to the file to be checked.
        target_md5 (str): The target MD5 checksum to compare against.
        chunk_size (int, optional): Size of each chunk to read from the file. Defaults to 8192.

    Returns:
        bool: True if the file's MD5 matches the target, False otherwise.

    Raises:
        FileNotFoundError: If the file does not exist.
        OSError: For checksum calculation I/O issues.
    """
    file_path = Path(path)
    if not file_path.is_file():
        logger.error("File not found: %s", file_path)
        raise FileNotFoundError(f"No such file: '{file_path}'")
    hash_md5 = hashlib.md5()
    try:
        with file_path.open("rb") as file_obj:
            for chunk in iter(lambda: file_obj.read(chunk_size), b""):
                hash_md5.update(chunk)
        computed_md5 = hash_md5.hexdigest()
        if computed_md5.lower() == target_md5.lower():
            logger.debug("MD5 checksum matched for %s: %s", file_path, computed_md5)
            return True
        logger.warning(
            "MD5 checksum mismatch for %s: %s != %s",
            file_path,
            computed_md5,
            target_md5,
        )
        return False
    except OSError as err:
        logger.error("Error computing MD5 for %s: %s", file_path, err)
        raise
