"""Whisper's special markers must not reach the transcript.

No GPU and no model: this is a filter over token ids, and the real tokenizers
load from the vocabulary files bundled with openai-whisper.
"""

import pytest
import torch
import whisper.tokenizer

from whisper_trt.model import WhisperTRT

_MULTILINGUAL = whisper.tokenizer.get_tokenizer(
    multilingual=True, language="en", task="transcribe"
)
_ENGLISH_ONLY = whisper.tokenizer.get_tokenizer(multilingual=False)

_TOKENIZERS = [
    pytest.param(_MULTILINGUAL, id="multilingual"),
    pytest.param(_ENGLISH_ONLY, id="english-only"),
]


def _decode(tokenizer, ids: list[int]) -> str:
    return WhisperTRT._decode_tokens(tokenizer, torch.tensor([ids]))


@pytest.mark.parametrize("tokenizer", _TOKENIZERS)
def test_plain_text_survives(tokenizer) -> None:
    ids = tokenizer.encoding.encode(" Turn on the living room lamp.")
    assert _decode(tokenizer, ids) == "Turn on the living room lamp."


@pytest.mark.parametrize("tokenizer", _TOKENIZERS)
def test_the_prompt_sequence_is_dropped(tokenizer) -> None:
    """The regression: sot and the language tag decoded as literal text.

    whisper's Tokenizer.decode keeps every id below ``timestamp_begin``, and
    that threshold is *above* the markers, so it drops none of these. `medium`
    returned "<|startoftranscript|><|en|> Turn on the living room lamp."
    """
    ids = [
        *tokenizer.sot_sequence,
        *tokenizer.encoding.encode(" Turn on the living room lamp."),
        tokenizer.eot,
    ]
    assert _decode(tokenizer, ids) == "Turn on the living room lamp."


@pytest.mark.parametrize("tokenizer", _TOKENIZERS)
def test_no_marker_spelling_reaches_the_transcript(tokenizer) -> None:
    """Every id at or above eot, not the three the denylist happened to name."""
    specials = sorted(tokenizer.special_tokens.values())
    assert len(specials) > 3, "expected the full marker set, not a handful"
    text = _decode(tokenizer, [*specials, *tokenizer.encoding.encode(" hello")])
    assert text == "hello"
    assert "<|" not in text


@pytest.mark.parametrize("tokenizer", _TOKENIZERS)
def test_timestamp_tokens_are_dropped(tokenizer) -> None:
    """Above eot as well, so the id filter covers them without a second rule."""
    ids = [
        tokenizer.timestamp_begin,
        *tokenizer.encoding.encode(" hello"),
        tokenizer.timestamp_begin + 20,
    ]
    assert _decode(tokenizer, ids) == "hello"


@pytest.mark.parametrize("tokenizer", _TOKENIZERS)
def test_no_speech_marker_is_dropped(tokenizer) -> None:
    """Sits between eot and timestamp_begin, so decode() alone keeps it."""
    assert _decode(tokenizer, [tokenizer.no_speech]) == ""


@pytest.mark.parametrize("tokenizer", _TOKENIZERS)
def test_an_empty_sequence_decodes_to_empty(tokenizer) -> None:
    assert _decode(tokenizer, [tokenizer.eot]) == ""
