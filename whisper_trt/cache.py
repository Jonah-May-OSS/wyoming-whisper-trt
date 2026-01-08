# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA
# CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# [License text continues...]


import logging
from pathlib import Path
from typing import Optional

# Configure logger

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Initialize cache directory

_CACHE_DIR = Path.home() / ".cache" / "whisper_trt"


def get_cache_dir() -> Path:
    """
    Retrieve the current cache directory path.

    Returns:
        Path: The current cache directory.
    """
    logger.debug(f"Retrieving cache directory: {_CACHE_DIR}")
    return _CACHE_DIR


def make_cache_dir() -> None:
    """
    Create the cache directory if it does not exist.

    Raises:
        RuntimeError: If the cache directory cannot be created.
    """
    try:
        if not _CACHE_DIR.exists():
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created cache directory at: {_CACHE_DIR}")
        else:
            logger.debug(f"Cache directory already exists at: {_CACHE_DIR}")
    except Exception as e:
        logger.error(f"Failed to create cache directory at {_CACHE_DIR}: {e}")
        raise RuntimeError(f"Could not create cache directory at {_CACHE_DIR}") from e


def set_cache_dir(path: str) -> None:
    """
    Set a new cache directory path.

    Args:
        path (str): The new cache directory path.

    Raises:
        RuntimeError: If the new cache directory cannot be created.
        TypeError: If the provided path is not a string.
    """
    global _CACHE_DIR

    if not isinstance(path, str):
        error_msg = "The 'path' parameter must be a string."
        logger.error(error_msg)
        raise TypeError(error_msg)
    new_cache_dir = Path(path).expanduser().resolve()
    logger.debug(f"Attempting to set new cache directory to: {new_cache_dir}")

    try:
        if not new_cache_dir.exists():
            new_cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created new cache directory at: {new_cache_dir}")
        else:
            logger.debug(f"New cache directory already exists at: {new_cache_dir}")
    except Exception as e:
        logger.error(f"Failed to create new cache directory at {new_cache_dir}: {e}")
        raise RuntimeError(
            f"Could not create new cache directory at {new_cache_dir}"
        ) from e
    _CACHE_DIR = new_cache_dir
    logger.info(f"Cache directory successfully set to: {_CACHE_DIR}")
