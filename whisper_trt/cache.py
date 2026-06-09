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

"""Cache directory utilities for Whisper TRT artifacts."""

import logging
from pathlib import Path

# Configure logger

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Initialize cache directory

_CACHE_STATE = {"dir": Path.home() / ".cache" / "whisper_trt"}


def get_cache_dir() -> Path:
    """
    Retrieve the current cache directory path.

    Returns:
        Path: The current cache directory.
    """
    cache_dir = _CACHE_STATE["dir"]
    logger.debug("Retrieving cache directory: %s", cache_dir)
    return cache_dir


def make_cache_dir() -> None:
    """
    Create the cache directory if it does not exist.

    Raises:
        RuntimeError: If the cache directory cannot be created.
    """
    cache_dir = get_cache_dir()
    try:
        if not cache_dir.exists():
            cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Created cache directory at: %s", cache_dir)
        else:
            logger.debug("Cache directory already exists at: %s", cache_dir)
    except OSError as err:
        logger.error("Failed to create cache directory at %s: %s", cache_dir, err)
        raise RuntimeError(f"Could not create cache directory at {cache_dir}") from err


def set_cache_dir(path: str) -> None:
    """
    Set a new cache directory path.

    Args:
        path (str): The new cache directory path.

    Raises:
        RuntimeError: If the new cache directory cannot be created.
        TypeError: If the provided path is not a string.
    """
    new_cache_dir = Path(path).expanduser().resolve()
    logger.debug("Attempting to set new cache directory to: %s", new_cache_dir)

    try:
        if not new_cache_dir.exists():
            new_cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Created new cache directory at: %s", new_cache_dir)
        else:
            logger.debug("New cache directory already exists at: %s", new_cache_dir)
    except OSError as err:
        logger.error(
            "Failed to create new cache directory at %s: %s", new_cache_dir, err
        )
        raise RuntimeError(
            f"Could not create new cache directory at {new_cache_dir}"
        ) from err
    _CACHE_STATE["dir"] = new_cache_dir
    logger.info("Cache directory successfully set to: %s", _CACHE_STATE["dir"])
