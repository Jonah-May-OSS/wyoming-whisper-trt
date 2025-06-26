#!/usr/bin/env python3


"""
Main entry point for the Whisper TRT server.

This script initializes the Whisper TRT model, sets up the server, and handles client events.
"""

# SDPA fix for Whisper 20240930 and newer per https://github.com/openai/whisper/discussions/2423
from whisper.model import disable_sdpa

import argparse
import asyncio
import logging
import sys
import time
from functools import partial
from pathlib import Path
from typing import Optional, List

from wyoming.info import AsrModel, AsrProgram, Attribution, Info
from wyoming.server import AsyncServer

from . import __version__
from .handler import WhisperTrtEventHandler
from whisper_trt.cache import get_cache_dir, make_cache_dir
from whisper_trt.utils import check_file_md5, download_file
from whisper_trt import MODEL_FILENAMES
from huggingface_hub import hf_hub_download

from whisper_trt import load_trt_model, WhisperTRT

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class NanosecondFormatter(logging.Formatter):
    """Custom formatter to include nanoseconds in log timestamps."""

    def formatTime(
        self, record: logging.LogRecord, datefmt: Optional[str] = None
    ) -> str:
        """Formats the time with nanosecond precision."""
        ct = record.created
        t = time.localtime(ct)
        s = time.strftime("%Y-%m-%d %H:%M:%S", t)
        return f"{s}.{int(ct * 1e9) % 1_000_000_000:09d}"


def setup_logging(debug: bool, log_format: str) -> None:
    """
    Sets up logging with the specified level and format.

    Args:
        debug (bool): Whether to enable DEBUG level logging.
        log_format (str): Format string for log messages.
    """
    formatter = NanosecondFormatter(log_format)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if debug else logging.INFO)
    root_logger.handlers = [handler]

    logger.debug("Logging has been configured.")


def normalize_model_name(model_name: str) -> str:
    """
    Normalizes the model name to ensure it matches the expected format.
    Retains language-specific suffixes to allow selection between multilingual and language-specific models.

    Args:
        model_name (str): The input model name.

    Returns:
        str: The normalized model name.
    """
    normalized_name = model_name.lower()
    logger.debug(f"Normalized model name: '{model_name}' to '{normalized_name}'.")
    return normalized_name


def is_language_specific(model_name: str) -> bool:
    """
    Determines if the model is language-specific based on its name.

    Args:
        model_name (str): The name of the model.

    Returns:
        bool: True if the model is language-specific, False otherwise.
    """
    return "." in model_name


def extract_languages(model: WhisperTRT, model_name: str) -> List[str]:
    """
    Extracts supported languages from the Whisper TRT model.

    Args:
        model (WhisperTRT): The Whisper TRT model instance.
        model_name (str): The name of the model.

    Returns:
        List[str]: A list of supported language codes.
    """
    if is_language_specific(model_name):
        # For example, "small.en" -> "en"

        language_code = model_name.split(".")[-1]
        languages = [language_code]
        logger.debug(
            f"Model '{model_name}' is language-specific. Supported language: {languages}"
        )
    else:
        try:
            # If your WhisperTRT class has a get_supported_languages() method,
            # it would return a list of all valid language codes. If not, fallback to ["en"].

            languages = model.get_supported_languages()
            logger.debug(
                f"Supported languages retrieved for model '{model_name}': {languages}"
            )
        except AttributeError:
            logger.warning(
                "Model does not have 'get_supported_languages' method. Defaulting to ['en']."
            )
            languages = ["en"]
        except Exception as e:
            logger.error(
                f"Error retrieving languages from model: {e}. Defaulting to ['en']."
            )
            languages = ["en"]
    return languages


def build_wyoming_info(model_name: str, languages: List[str]) -> Info:
    """
    Builds the Wyoming Info object with ASR model details.

    Args:
        model_name (str): The name of the ASR model.
        languages (List[str]): Supported languages for the model.

    Returns:
        Info: The constructed Info object.
    """
    wyoming_info = Info(
        asr=[
            AsrProgram(
                name="whisper-trt",
                description="OpenAI Whisper with TensorRT",
                attribution=Attribution(
                    name="NVIDIA-AI-IOT",
                    url="https://github.com/NVIDIA-AI-IOT/whisper_trt",
                ),
                installed=True,
                version=__version__,
                models=[
                    AsrModel(
                        name=model_name,
                        description=model_name,
                        attribution=Attribution(
                            name="OpenAI",
                            url="https://huggingface.co/OpenAI",
                        ),
                        installed=True,
                        languages=languages,
                        version=__version__,
                    )
                ],
            )
        ],
    )
    logger.debug(
        f"Wyoming Info built with model '{model_name}' and languages {languages}."
    )
    return wyoming_info


async def run_server(uri: str, handler_factory_func, *args, **kwargs) -> None:
    """
    Initializes and runs the asynchronous server.

    Args:
        uri (str): The URI to bind the server to.
        handler_factory_func: The handler factory function.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
    """
    try:
        server = AsyncServer.from_uri(uri)
        logger.info(f"Server initialized and listening on {uri}.")
    except Exception as e:
        logger.error(f"Failed to initialize server with URI '{uri}': {e}")
        raise
    try:
        await server.run(handler_factory_func, *args, **kwargs)
    except Exception as e:
        logger.error(f"Server encountered an error: {e}")
        raise
    finally:
        await server.close()
        logger.info("Server has been shut down.")


def fetch_model_source(model_name: str, download_dir: Path) -> Path:
    download_dir.mkdir(exist_ok=True, parents=True)
    # 1) If it’s one of the built-ins, keep your existing PTH filename:

    if model_name in MODEL_FILENAMES:
        return download_dir / MODEL_FILENAMES[model_name]
    # 2) Otherwise treat it as a HF repo ID → download ONNX

    onnx_name = f"{model_name.replace('/', '_')}.onnx"
    local_onnx = download_dir / onnx_name
    if not local_onnx.exists():
        print(f"⤵  Downloading ONNX model for '{model_name}'")
        try:
            # Note: This assumes the ONNX file is named "model.onnx" in the HF repo
            downloaded_path = hf_hub_download(
                repo_id=model_name,
                filename="model.onnx",
                local_dir=str(download_dir),
                local_dir_use_symlinks=False,
            )
            # Rename to expected filename if needed
            if Path(downloaded_path).name != onnx_name:
                Path(downloaded_path).rename(local_onnx)
        except Exception as e:
            logger.error(f"Failed to download ONNX model from '{model_name}': {e}")
            raise RuntimeError(f"Could not fetch model '{model_name}': {e}")
    return local_onnx


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Whisper TRT ASR Server")
    parser.add_argument(
        "--model",
        required=True,
        help="Name of Whisper model to use (e.g., 'tiny', 'small', 'small.en')",
    )
    parser.add_argument(
        "--uri",
        required=True,
        help="Server URI to bind to (e.g., 'unix://', 'tcp://')",
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        action="append",
        help="Data directory to check for downloaded models",
    )
    parser.add_argument(
        "--download-dir",
        help="Directory to download models into (default: first data dir)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for inference (default: cuda)",
    )
    parser.add_argument(
        "--compute-type",
        default="default",
        choices=["default", "float16", "int8"],
        help="Compute type (float16, int8, etc.)",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Beam size for decoding (default: 5)",
    )
    parser.add_argument(
        "--initial-prompt",
        help="Optional text to provide as a prompt for the first window",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG level logging",
    )
    parser.add_argument(
        "--log-format",
        default="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        help="Format for log messages",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="Print version and exit",
    )

    # Restored/added argument for default language selection

    parser.add_argument(
        "--language",
        type=str,
        default="auto",
        help="Default language to use for transcription. Use 'auto' for detection.",
    )

    args = parser.parse_args()

    # Setup logging

    setup_logging(args.debug, args.log_format)

    # Normalize model name

    model_name = normalize_model_name(args.model)

    # Determine if the model is language-specific

    model_is_lang_specific = is_language_specific(model_name)
    logger.debug(f"Model '{model_name}' is language-specific: {model_is_lang_specific}")

    # Set download directory to first data directory if not specified

    if not args.download_dir:
        args.download_dir = args.data_dir[0]
        logger.debug(
            f"No download directory specified. Using first data directory: {args.download_dir}"
        )
    download_path = Path(args.download_dir)
    try:
        download_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured download directory exists at '{download_path}'.")
    except OSError as e:
        logger.error(f"Failed to create download directory at '{download_path}': {e}")
        sys.exit(1)
    # Load Whisper TRT model

    try:
        source_path = fetch_model_source(model_name, download_path)
        # if it’s an ONNX file, pass build=True so load_trt_model will convert it:

        logger.info(f"Loading Whisper TRT model '{model_name}'...")
        trt_model = load_trt_model(model_name, path=str(source_path), build=True)
        logger.info(f"Whisper TRT model '{model_name}' loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load Whisper TRT model '{model_name}': {e}")
        sys.exit(1)
    # Extract supported languages

    languages = extract_languages(trt_model, model_name)

    # Build Wyoming Info

    wyoming_info = build_wyoming_info(model_name, languages)

    # Initialize asyncio lock for model access

    model_lock = asyncio.Lock()
    logger.debug("Initialized asyncio lock for model access.")

    # Create the event handler factory, passing the user-defined language

    handler_factory_func = partial(
        WhisperTrtEventHandler,
        wyoming_info=wyoming_info,
        cli_args=args,
        model=trt_model,
        model_lock=model_lock,
        initial_prompt=args.initial_prompt,
        model_is_lang_specific=model_is_lang_specific,
        default_language=args.language,  # Pass the user-selected language
    )

    # Run the server

    try:
        logger.info("Starting the Whisper TRT ASR server...")
        await run_server(args.uri, handler_factory_func)
    except Exception as e:
        logger.error(f"Server encountered an unexpected error: {e}")
        sys.exit(1)


def run() -> None:
    """Runs the main asynchronous entry point."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down due to keyboard interrupt.")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # SDPA fix for Whisper 20240930 and newer per https://github.com/openai/whisper/discussions/2423
    with disable_sdpa():
        run()
