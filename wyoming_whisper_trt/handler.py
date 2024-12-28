"""Event handler for clients of the server."""

import argparse
import asyncio
import logging
import time
from pathlib import Path
from typing import Optional, Union
import functools

import torch
import wave
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStop
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler, WyomingServer

import whisper_trt

# Configure module-specific logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class NanosecondFormatter(logging.Formatter):
    """Custom formatter to include nanoseconds in log timestamps."""

    def formatTime(self, record: logging.LogRecord, datefmt: Optional[str] = None) -> str:
        """Formats the time with nanosecond precision."""
        ct = time.time()
        t = time.localtime(ct)
        s = time.strftime("%Y-%m-%d %H:%M:%S", t)
        return f"{s}.{int(ct * 1e9) % 1_000_000_000:09d}"


# Set up logging with the custom formatter
formatter = NanosecondFormatter('%(asctime)s [%(levelname)s] %(message)s')
handler = logging.StreamHandler()
handler.setFormatter(formatter)

# Apply the formatter to the root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)
root_logger.handlers = [handler]

_LOGGER = logging.getLogger(__name__)


class WhisperTrtEventHandler(AsyncEventHandler):
    """Event handler for clients utilizing the Whisper TRT model."""

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        *,
        wyoming_info: Info,
        cli_args: argparse.Namespace,
        model: whisper_trt.WhisperTRT,
        model_lock: asyncio.Lock,
        initial_prompt: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Initializes the WhisperTrtEventHandler.

        Args:
            reader (asyncio.StreamReader): The stream reader for client communication.
            writer (asyncio.StreamWriter): The stream writer for client communication.
            wyoming_info (Info): Information about the Wyoming server.
            cli_args (argparse.Namespace): Command-line arguments.
            model (whisper_trt.WhisperTRT): The Whisper TRT model instance.
            model_lock (asyncio.Lock): Asynchronous lock for model access.
            initial_prompt (Optional[str], optional): Initial prompt for transcription. Defaults to None.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(reader, writer, **kwargs)

        self.cli_args = cli_args
        self.wyoming_info_event = wyoming_info.event()
        self.model = model
        self.model_lock = model_lock
        self.initial_prompt = initial_prompt
        self._language = self.cli_args.language
        self._wav_dir = tempfile.TemporaryDirectory()
        self._wav_path = Path(self._wav_dir.name) / "speech.wav"
        self._wav_file: Optional[wave.Wave_write] = None

    async def handle_event(self, event: Event) -> bool:
        """
        Handles incoming events from clients.

        Args:
            event (Event): The event to handle.

        Returns:
            bool: Whether to continue handling events.
        """
        try:
            if AudioChunk.is_type(event.type):
                await self._handle_audio_chunk(event)
                return True

            if AudioStop.is_type(event.type):
                await self._handle_audio_stop()
                return False

            if Transcribe.is_type(event.type):
                await self._handle_transcribe(event)
                return True

            if Describe.is_type(event.type):
                await self._handle_describe()
                return True

            return True
        except Exception as e:
            _LOGGER.error(f"Error handling event {event.type}: {e}")
            return True  # Continue handling other events

    async def _handle_audio_chunk(self, event: Event) -> None:
        """
        Handles an AudioChunk event by writing audio data to a WAV file.

        Args:
            event (Event): The AudioChunk event.
        """
        chunk = AudioChunk.from_event(event)

        if self._wav_file is None:
            try:
                self._wav_file = wave.open(self._wav_path, "wb")
                self._wav_file.setframerate(chunk.rate)
                self._wav_file.setsampwidth(chunk.width)
                self._wav_file.setnchannels(chunk.channels)
                _LOGGER.debug(f"Initialized WAV file at '{self._wav_path}'.")
            except wave.Error as e:
                _LOGGER.error(f"Failed to open WAV file: {e}")
                raise

        self._wav_file.writeframes(chunk.audio)
        _LOGGER.debug(f"Wrote {len(chunk.audio)} frames to WAV file.")

    async def _handle_audio_stop(self) -> None:
        """
        Handles an AudioStop event by transcribing the recorded audio.
        """
        if self._wav_file is None:
            _LOGGER.warning("AudioStop received but no audio was recorded.")
            return

        try:
            self._wav_file.close()
            _LOGGER.debug(f"Closed WAV file at '{self._wav_path}'.")
        except wave.Error as e:
            _LOGGER.error(f"Failed to close WAV file: {e}")
            raise
        finally:
            self._wav_file = None

        async with self.model_lock:
            try:
                result = await asyncio.get_event_loop().run_in_executor(
                    None, self.model.transcribe, str(self._wav_path)
                )
                text = result.get("text", "")
                _LOGGER.info(f"Transcription result: {text}")
                await self.write_event(Transcript(text=text).event())
                _LOGGER.debug("Completed transcription request.")
            except Exception as e:
                _LOGGER.error(f"Transcription failed: {e}")
                await self.write_event(Transcript(text="Transcription failed.").event())

        # Reset language to CLI argument
        self._language = self.cli_args.language
        _LOGGER.debug(f"Reset language to '{self._language}'.")

    async def _handle_transcribe(self, event: Event) -> None:
        """
        Handles a Transcribe event by setting the language.

        Args:
            event (Event): The Transcribe event.
        """
        transcribe = Transcribe.from_event(event)
        if transcribe.language:
            self._language = transcribe.language
            _LOGGER.debug(f"Language set to '{transcribe.language}'.")

    async def _handle_describe(self) -> None:
        """
        Handles a Describe event by sending server information.
        """
        try:
            await self.write_event(self.wyoming_info_event)
            _LOGGER.debug("Sent server information.")
        except Exception as e:
            _LOGGER.error(f"Failed to send server information: {e}")

    def cleanup(self) -> None:
        """
        Cleans up resources such as temporary directories.
        """
        if self._wav_dir:
            self._wav_dir.cleanup()
            _LOGGER.debug(f"Cleaned up temporary directory '{self._wav_dir.name}'.")

def create_handler_factory(
    wyoming_info: Info,
    cli_args: argparse.Namespace,
    model: whisper_trt.WhisperTRT,
    model_lock: asyncio.Lock,
    initial_prompt: Optional[str] = None,
):
    """
    Creates a handler factory function with pre-specified arguments.

    Args:
        wyoming_info (Info): Information about the Wyoming server.
        cli_args (argparse.Namespace): Command-line arguments.
        model (whisper_trt.WhisperTRT): The Whisper TRT model instance.
        model_lock (asyncio.Lock): Asynchronous lock for model access.
        initial_prompt (Optional[str], optional): Initial prompt for transcription. Defaults to None.

    Returns:
        Callable: A factory function that accepts reader and writer.
    """
    def handler_factory(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> WhisperTrtEventHandler:
        return WhisperTrtEventHandler(
            reader,
            writer,
            wyoming_info=wyoming_info,
            cli_args=cli_args,
            model=model,
            model_lock=model_lock,
            initial_prompt=initial_prompt,
        )
    
    return handler_factory


def main():
    """Main entry point for the event handler."""
    parser = argparse.ArgumentParser(description="Whisper TRT Event Handler")
    parser.add_argument("--language", type=str, default="en", help="Language for transcription")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    args = parser.parse_args()

    # Initialize Whisper TRT model and asyncio lock
    model = whisper_trt.load_trt_model("tiny.en", build=True)
    model_lock = asyncio.Lock()

    # Initialize Wyoming Info
    wyoming_info = Info()

    # Create the handler factory with the necessary arguments
    handler_factory = create_handler_factory(
        wyoming_info=wyoming_info,
        cli_args=args,
        model=model,
        model_lock=model_lock,
        initial_prompt=None,  # Modify as needed
    )

    # Initialize the Wyoming server
    server = WyomingServer(
        host=args.host,
        port=args.port,
        handler_factory=handler_factory,
    )

    # Run the server within an asyncio event loop
    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        _LOGGER.info("Shutting down server.")
    finally:
        # Perform any necessary cleanup here
        pass

if __name__ == "__main__":
    main()
