import hashlib
import logging
from pathlib import Path
from typing import Optional

import requests
from requests.adapters import HTTPAdapter, Retry
from tqdm import tqdm

# Configure logger

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def download_file(
    url: str,
    path: str,
    makedirs: bool = False,
    timeout: Optional[int] = 30,
    retries: int = 3,
    backoff_factor: float = 0.3,
) -> None:
    """
    Download a file from a given URL to a specified path with retry and progress indication.

    Args:
        url (str): The URL to download the file from.
        path (str): The destination file path where the downloaded file will be saved.
        makedirs (bool, optional): If True, creates the parent directories if they do not exist. Defaults to False.
        timeout (Optional[int], optional): Timeout for the HTTP request in seconds. Defaults to 30.
        retries (int, optional): Number of retries for failed downloads. Defaults to 3.
        backoff_factor (float, optional): Backoff factor for retries. Defaults to 0.3.

    Raises:
        requests.HTTPError: If the HTTP request returned an unsuccessful status code.
        IOError: If the downloaded file size does not match the expected size.
        Exception: For other exceptions during the download process.
    """
    destination = Path(path)

    if makedirs:
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directories up to: {destination.parent}")
        except Exception as e:
            logger.error(f"Failed to create directories for {destination}: {e}")
            raise
    session = requests.Session()
    retry_strategy = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    try:
        logger.info(f"Starting download from {url} to {destination}")
        with session.get(url, stream=True, timeout=timeout) as response:
            response.raise_for_status()
            total_size_in_bytes = int(response.headers.get("content-length", 0))
            block_size = 8192  # 8 KB
            progress_bar = tqdm(
                total=total_size_in_bytes,
                unit="iB",
                unit_scale=True,
                desc=destination.name,
            )

            with destination.open("wb") as f:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:  # Filter out keep-alive chunks
                        f.write(chunk)
                        progress_bar.update(len(chunk))
            progress_bar.close()
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            error_msg = (
                f"Downloaded size mismatch: {progress_bar.n} != {total_size_in_bytes}"
            )
            logger.error(error_msg)
            raise IOError(error_msg)
        logger.info(f"Successfully downloaded {url} to {destination}")
    except requests.HTTPError as http_err:
        logger.error(f"HTTP error occurred while downloading {url}: {http_err}")
        raise
    except Exception as err:
        logger.error(f"An error occurred while downloading {url}: {err}")
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
        Exception: For other exceptions during the checksum calculation.
    """
    file_path = Path(path)
    if not file_path.is_file():
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"No such file: '{file_path}'")
    hash_md5 = hashlib.md5()
    try:
        with file_path.open("rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                hash_md5.update(chunk)
        computed_md5 = hash_md5.hexdigest()
        if computed_md5.lower() == target_md5.lower():
            logger.debug(f"MD5 checksum matched for {file_path}: {computed_md5}")
            return True
        else:
            logger.warning(
                f"MD5 checksum mismatch for {file_path}: {computed_md5} != {target_md5}"
            )
            return False
    except Exception as e:
        logger.error(f"Error computing MD5 for {file_path}: {e}")
        raise
