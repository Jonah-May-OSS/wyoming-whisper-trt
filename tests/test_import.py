"""Basic smoke tests to verify package imports."""

import pytest


def test_import_main_module():
    """Test that the main module can be imported."""
    import wyoming_whisper_trt

    assert wyoming_whisper_trt is not None


def test_import_handler():
    """Test that the handler module can be imported."""
    from wyoming_whisper_trt import handler

    assert handler is not None


def test_import_main():
    """Test that the __main__ module can be imported."""
    from wyoming_whisper_trt import __main__

    assert __main__ is not None
