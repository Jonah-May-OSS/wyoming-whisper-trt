#!/usr/bin/env python3

"""
Main entry point for the Whisper TRT server.

This script initializes the Whisper TRT model, sets up the server, and handles client events.
"""

# SDPA fix for Whisper 20240930 and newer per https://github.com/openai/whisper/discussions/2423
import argparse
import asyncio
import logging
import os
import sys
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

from whisper.model import disable_sdpa
from wyoming.info import AsrModel, AsrProgram, Attribution, Info
from wyoming.server import AsyncServer

from whisper_trt import MODEL_FILENAMES, WhisperTRT, WhisperTRTBuilder, load_trt_model

from . import __version__
from .handler import HandlerContext, HandlerSettings, WhisperTrtEventHandler

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class NanosecondFormatter(logging.Formatter):
    """Custom formatter to include nanoseconds in log timestamps."""

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
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
    Normalizes the model name to ensure it matches the expected format while
    retaining language-specific suffixes.

    Args:
        model_name (str): The input model name.

    Returns:
        str: The normalized model name.
    """
    normalized_name = model_name.lower()
    logger.debug("Normalized model name: '%s' to '%s'.", model_name, normalized_name)
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


def extract_languages(model: WhisperTRT, model_name: str) -> list[str]:
    """
    Extracts supported languages from the Whisper TRT model.

    Args:
        model (WhisperTRT): The Whisper TRT model instance.
        model_name (str): The name of the model.

    Returns:
        list[str]: A list of supported language codes.
    """
    if is_language_specific(model_name):
        # For example, "small.en" -> "en"
        language_code = model_name.split(".")[-1]
        languages = [language_code]
        logger.debug(
            "Model '%s' is language-specific. Supported language: %s",
            model_name,
            languages,
        )
    else:
        try:
            # If your WhisperTRT class has a get_supported_languages() method,
            # it would return a list of all valid language codes. If not, fallback to ["en"].
            languages = model.get_supported_languages()
            logger.debug(
                "Supported languages retrieved for model '%s': %s",
                model_name,
                languages,
            )
        except AttributeError:
            logger.warning(
                "Model does not have 'get_supported_languages' method. Defaulting to ['en']."
            )
            languages = ["en"]
        except (RuntimeError, ValueError, TypeError) as err:
            logger.error(
                "Error retrieving languages from model: %s. Defaulting to ['en'].",
                err,
            )
            languages = ["en"]
    return languages


def build_wyoming_info(
    model_name: str, languages: list[str], streaming: bool = False
) -> Info:
    """
    Builds the Wyoming Info object with ASR model details.

    Args:
        model_name (str): The name of the ASR model.
        languages (list[str]): Supported languages for the model.
        streaming (bool): Whether transcript streaming is enabled.

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
                supports_transcript_streaming=streaming,
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
        "Wyoming Info built with model '%s' and languages %s.",
        model_name,
        languages,
    )
    return wyoming_info


async def run_server(
    uri: str,
    handler_factory_func: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> None:
    """
    Initializes and runs the asynchronous server.

    Args:
        uri (str): The URI to bind the server to.
        handler_factory_func: The handler factory function.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
    """
    # Create the wyoming server (e.g. AsyncTcpServer)
    server = AsyncServer.from_uri(uri)
    logger.info("Server initialized and listening on %s.", uri)

    try:
        await server.run(handler_factory_func, *args, **kwargs)
    except (RuntimeError, OSError, ValueError) as err:
        logger.error("Server encountered an error: %s", err)
        raise
    finally:
        # Use the stop method to handle event handler shutdown and server closure
        await server.stop()
        logger.info("Server and event handlers stopped gracefully.")


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
        default="float16",
        choices=["float32", "float16"],
        help="Compute type (float32, float16)",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Beam size for decoding (default: 5)",
    )
    parser.add_argument(
        "--initial-prompt",
        default=None,
        help="Optional text to provide as a prompt for the first window",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Enable partial transcription streaming",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG level logging",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="Print version and exit",
    )
    parser.add_argument(
        "--log-format",
        default="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        help="Format for log messages",
    )
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

    # Set compute-type
    WhisperTRTBuilder.fp16_mode = args.compute_type == "float16"

    # Set download directory to first data directory if not specified
    if not args.download_dir:
        args.download_dir = args.data_dir[0]
        logger.debug(
            "No download directory specified. Using first data directory: %s",
            args.download_dir,
        )

    download_path = Path(args.download_dir)
    try:
        download_path.mkdir(parents=True, exist_ok=True)
        logger.debug("Ensured download directory exists at '%s'.", download_path)
    except OSError as e:
        logger.error(
            "Failed to create download directory at '%s': %s", download_path, e
        )
        sys.exit(1)

    # Load Whisper TRT model
    try:
        logger.info("Loading Whisper TRT model '%s'...", model_name)
        model_path = os.path.join(args.download_dir, MODEL_FILENAMES[model_name])
        trt_model = load_trt_model(
            args.model,
            path=model_path,
            build=True,
            verbose=args.debug,
            language=args.language,
        )
        logger.info("Whisper TRT model '%s' loaded successfully.", model_name)
    except (KeyError, RuntimeError, OSError, ValueError) as err:
        logger.error("Failed to load Whisper TRT model '%s': %s", model_name, err)
        sys.exit(1)

    # Extract supported languages
    languages = extract_languages(trt_model, model_name)

    # Build Wyoming Info
    wyoming_info = build_wyoming_info(model_name, languages, streaming=args.streaming)

    # Initialize asyncio lock for model access
    model_lock = asyncio.Lock()
    logger.debug("Initialized asyncio lock for model access.")

    # Create the event handler factory, passing the user-defined language
    handler_factory_func = partial(
        WhisperTrtEventHandler,
        context=HandlerContext(
            wyoming_info=wyoming_info,
            model=trt_model,
            model_lock=model_lock,
        ),
        settings=HandlerSettings(
            initial_prompt=args.initial_prompt,
            streaming=args.streaming,
            default_language=args.language,
        ),
    )

    # Run the server
    try:
        logger.info("Starting the Whisper TRT ASR server...")
        await run_server(args.uri, handler_factory_func)
    except (RuntimeError, OSError, ValueError) as err:
        logger.error("Server encountered an unexpected error: %s", err)
        sys.exit(1)


def run() -> None:
    """Runs the main asynchronous entry point."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down due to keyboard interrupt.")
    except (RuntimeError, OSError, ValueError) as err:
        logger.error("Unhandled exception: %s", err)
        sys.exit(1)


if __name__ == "__main__":
    # SDPA fix for Whisper 20240930 and newer per https://github.com/openai/whisper/discussions/2423
    with disable_sdpa():
        run()
