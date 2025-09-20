"""Test for large model memory optimizations."""

import unittest
import sys
from unittest.mock import MagicMock, patch

# Mock all torch-related modules at import time
torch_mock = MagicMock()
torch_mock.nn = MagicMock()
torch_mock.nn.functional = MagicMock()
torch_mock.cuda = MagicMock()
torch_mock.no_grad = MagicMock()

mock_modules = {
    'torch': torch_mock,
    'torch.nn': torch_mock.nn,
    'torch.nn.functional': torch_mock.nn.functional,
    'torch2trt': MagicMock(),
    'tensorrt': MagicMock(),
    'whisper': MagicMock(),
    'whisper.model': MagicMock(),
    'whisper.tokenizer': MagicMock(),
    'whisper.audio': MagicMock(),
    'numpy': MagicMock(),
}

# Add the whisper_trt directory to sys.path
sys.path.insert(0, '/home/runner/work/wyoming-whisper-trt/wyoming-whisper-trt')

with patch.dict('sys.modules', mock_modules):
    from whisper_trt.model import WhisperTRTBuilder, LargeV3TurboBuilder, TinyBuilder


class TestLargeModelOptimizations(unittest.TestCase):
    """Test that large models get appropriate memory allocations."""

    def test_workspace_size_for_large_models(self):
        """Test that large models get more workspace memory."""
        # Large models should get 4GB workspace
        large_workspace = LargeV3TurboBuilder.get_workspace_size()
        expected_large = 1 << 32  # 4GB
        self.assertEqual(large_workspace, expected_large,
                        f"Large models should get 4GB workspace, got {large_workspace / (1024**3):.1f}GB")

    def test_workspace_size_for_small_models(self):
        """Test that small models get standard workspace memory."""
        # Small models should get 1GB workspace
        tiny_workspace = TinyBuilder.get_workspace_size()
        expected_tiny = 1 << 30  # 1GB
        self.assertEqual(tiny_workspace, expected_tiny,
                        f"Small models should get 1GB workspace, got {tiny_workspace / (1024**3):.1f}GB")

    def test_large_model_identification(self):
        """Test that all large model variants are properly identified."""
        large_models = {"large", "large-v2", "large-v3", "large-v3-turbo"}
        
        class TestBuilder(WhisperTRTBuilder):
            pass
        
        for model in large_models:
            TestBuilder.model = model
            workspace = TestBuilder.get_workspace_size()
            expected = 1 << 32  # 4GB
            self.assertEqual(workspace, expected,
                           f"Model {model} should get 4GB workspace")

    def test_small_model_identification(self):
        """Test that small models get standard workspace allocation."""
        small_models = {"tiny", "base", "small", "medium", "tiny.en", "base.en", "small.en"}
        
        class TestBuilder(WhisperTRTBuilder):
            pass
        
        for model in small_models:
            TestBuilder.model = model
            workspace = TestBuilder.get_workspace_size()
            expected = 1 << 30  # 1GB
            self.assertEqual(workspace, expected,
                           f"Model {model} should get 1GB workspace")


if __name__ == '__main__':
    unittest.main()