"""Unit tests for the Qwen3-ASR echo guard, against a scripted fake decoder.

The guard has two jobs -- tell an echo of the biasing prompt (the names read
back in order, however the model cases and spaces them) from a command that
mentions a name or two, and stop decoding early without disturbing the cached KV
of the prompt -- and neither needs real weights. The fake decoder emits a scripted
token sequence, so each test states exactly what the model "says". See
test_qwen3_asr_noise for the same behaviour on the real model.
"""

import re
import wave
from types import SimpleNamespace
from typing import Dict, List

import numpy as np
import pytest

qwen3_asr_handler = pytest.importorskip(
    "wyoming_faster_whisper.qwen3_asr_handler",
    reason="qwen3-asr extra not installed",
)

from .test_qwen3_asr import (  # noqa: E402
    _drain,
    _fake_model_dir,
    _FakeSession,
    _FakeStreamingModel,
    _session,
    _silence,
    _SpanClipper,
)

_COMMA = 11
_ASR_TEXT = qwen3_asr_handler._ASR_TEXT
_EOS = 151645
_N = qwen3_asr_handler._ECHO_MIN_NAMES
_normalize = qwen3_asr_handler._normalize
_echo_patterns = qwen3_asr_handler._echo_patterns
_VOCAB = 151710
_N_AUDIO = 4

_PROMPT = "Office, Study, Upstairs, Top Floor, Office Lamp, Desk Lamp, Kitchen."


# --- fakes -----------------------------------------------------------------


class _FakeTokenizer:
    """Words and punctuation, one id each."""

    def __init__(self) -> None:
        self._ids: Dict[str, int] = {",": _COMMA}
        self._text: Dict[int, str] = {_COMMA: ","}

    def _id(self, piece: str) -> int:
        if piece not in self._ids:
            new_id = 1000 + len(self._ids)
            self._ids[piece] = new_id
            self._text[new_id] = piece
        return self._ids[piece]

    def encode(self, text: str, add_special_tokens: bool = False):
        pieces = re.findall(r"[,.?\uff0c]|\s?[^\s,.?\uff0c]+", text)
        return SimpleNamespace(ids=[self._id(piece) for piece in pieces])

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        return "".join(self._text.get(i, "") for i in ids if i < 151643)


class _ScriptedDecoder:
    """Says ``tokens`` after the prompt, then EOS, whatever it is fed.

    Stands in for whichever decoder layout the model has: _run_merged, or the
    split layout's decoder_init and decoder_step sessions. Row j of a call's
    logits picks the token after position j of that call, so a prefill or step
    that also checks draft tokens gets the right answer for each of them.
    """

    def __init__(self, model, tokens: List[int]) -> None:
        self.tokens = list(tokens)
        self.calls: List[SimpleNamespace] = []
        self.prompt_len = 0

        build_prompt_ids = model._build_prompt_ids

        def recording(*args):
            prompt_ids = build_prompt_ids(*args)
            self.prompt_len = len(prompt_ids)
            return prompt_ids

        model._build_prompt_ids = recording
        if model._merged is not None:
            model._run_merged = self._merged
        else:
            model._decoder_init = SimpleNamespace(run=self._init)
            model._decoder_step = SimpleNamespace(run=self._step)

    def _run(self, positions, past: int):
        positions = np.asarray(positions).reshape(-1)
        self.calls.append(
            SimpleNamespace(start=int(positions[0]), q_len=len(positions), past=past)
        )

        logits = np.zeros((1, len(positions), _VOCAB), dtype=np.float32)
        for row, position in enumerate(positions):
            index = int(position) + 1 - self.prompt_len
            token = self.tokens[index] if 0 <= index < len(self.tokens) else _EOS
            logits[0, row, token] = 1.0

        kv = np.zeros((4, 1, 2, past + len(positions), 8), dtype=np.float32)
        return logits, kv, kv

    def _merged(
        self, input_embeds, positions, past_keys, past_values, run_options=None
    ):
        return self._run(positions, past_keys.shape[3])

    def _init(self, outputs, feeds, run_options=None):
        return self._run(feeds["position_ids"], 0)

    def _step(self, outputs, feeds, run_options=None):
        return self._run(feeds["position_ids"], feeds["past_keys"].shape[3])

    @property
    def steps(self) -> int:
        """Decoder passes after the prefill, each feeding a generated token."""
        return sum(1 for call in self.calls if call.start >= self.prompt_len)

    @property
    def prefills_from_scratch(self) -> int:
        return sum(1 for call in self.calls if call.past == 0)


def _make_model(tmp_path, monkeypatch, merged: bool):
    monkeypatch.setattr(qwen3_asr_handler.ort, "InferenceSession", _FakeSession)
    monkeypatch.setattr(
        qwen3_asr_handler,
        "Tokenizer",
        SimpleNamespace(from_file=lambda path: _FakeTokenizer()),
    )
    monkeypatch.setattr(
        qwen3_asr_handler.np,
        "fromfile",
        lambda *args, **kwargs: np.zeros(16, dtype=np.float16),
    )
    model = qwen3_asr_handler.Qwen3AsrTranscriber(
        str(_fake_model_dir(tmp_path, merged=merged)), str(tmp_path)
    )
    model._embed_tokens = np.zeros((_VOCAB, 8), dtype=np.float16)
    model._encoder = SimpleNamespace(
        run=lambda outputs, inputs, run_options=None: [
            np.zeros((1, _N_AUDIO, 8), dtype=np.float32)
        ]
    )
    return model


@pytest.fixture(name="model", params=["merged", "split"])
def model_fixture(request, tmp_path, monkeypatch):
    """The guard is in the decode loop both layouts share; test it on each."""
    return _make_model(tmp_path, monkeypatch, merged=request.param == "merged")


@pytest.fixture(name="merged_model")
def merged_model_fixture(tmp_path, monkeypatch):
    """Only the merged layout caches the prompt's KV, and only it takes drafts."""
    return _make_model(tmp_path, monkeypatch, merged=True)


@pytest.fixture(name="wav_path")
def wav_path_fixture(tmp_path):
    path = tmp_path / "audio.wav"
    wav_file: wave.Wave_write = wave.open(str(path), "wb")
    with wav_file:
        wav_file.setframerate(16000)
        wav_file.setsampwidth(2)
        wav_file.setnchannels(1)
        wav_file.writeframes(bytes(2 * 16000))
    return path


def _ids(model, text: str) -> List[int]:
    return model._tokenizer.encode(text).ids


def _say(model, text: str) -> _ScriptedDecoder:
    """Make the fake model say ``text`` for the next utterance."""
    return _ScriptedDecoder(model, _ids(model, text) + [_EOS])


# --- _normalize / _echo_patterns -----------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Office, Study, Upstairs.", "office,study,upstairs"),
        # How the 1.7B model echoes: its own casing and word splits.
        (
            "The office, study, up stairs, top floor",
            "theoffice,study,upstairs,topfloor",
        ),
        ("Living Room, Light", "livingroom,light"),
        ("办公室，书房、厨房", "办公室,书房,厨房"),  # CJK separators
        ("What's the 2nd one?", "whatsthe2ndone"),
    ],
)
def test_normalize(text, expected) -> None:
    assert _normalize(text) == expected


def test_patterns_are_runs_of_consecutive_names() -> None:
    patterns = _echo_patterns("Office, Study, Upstairs, Top Floor.")
    assert patterns == {"office,study,upstairs", "study,upstairs,topfloor"}


def test_a_prompt_prefix_stays_part_of_the_first_name() -> None:
    patterns = _echo_patterns("Vocabulary: Office, Study, Upstairs, Top Floor.")
    assert "vocabularyoffice,study,upstairs" in patterns
    assert "study,upstairs,topfloor" in patterns  # so an echo is still caught


def test_a_short_prompt_is_matched_as_a_whole() -> None:
    assert _echo_patterns("Kitchen, Office.") == {"kitchen,office"}


@pytest.mark.parametrize("prompt", [None, "", "Ecobee.", " , ."])
def test_a_prompt_that_is_not_a_list_has_no_patterns(prompt) -> None:
    assert _echo_patterns(prompt) == frozenset()


# --- transcribe: not an echo -----------------------------------------------


@pytest.mark.parametrize(
    "said",
    [
        "Turn on the Office Lamp.",
        "Office Lamp.",  # just a name from the list
        "Office, Study.",  # two names in prompt order: too short to be an echo
        "Turn on Upstairs and Top Floor.",
        "Kitchen, Study, Office.",  # three names, but not in the prompt's order
    ],
)
def test_speech_is_returned_unchanged(model, wav_path, said) -> None:
    _say(model, said)
    assert model.transcribe(wav_path, "en", initial_prompt=_PROMPT) == said


def test_without_a_prompt_nothing_is_an_echo(model, wav_path) -> None:
    _say(model, _PROMPT)
    assert model.transcribe(wav_path, "en") == _PROMPT


def test_a_different_prompt_is_not_an_echo(model, wav_path) -> None:
    # The same words, but not what this utterance was biased with.
    _say(model, _PROMPT)
    assert model.transcribe(wav_path, "en", initial_prompt="Ecobee.") == _PROMPT


# --- transcribe: echo ------------------------------------------------------


def test_an_echo_of_the_prompt_gives_an_empty_transcript(model, wav_path) -> None:
    _say(model, _PROMPT)
    assert model.transcribe(wav_path, "en", initial_prompt=_PROMPT) == ""


def test_an_echo_of_the_middle_of_the_prompt_is_caught(model, wav_path) -> None:
    _say(model, "Upstairs, Top Floor, Office Lamp, Desk Lamp.")
    assert model.transcribe(wav_path, "en", initial_prompt=_PROMPT) == ""


def test_an_echo_after_real_words_is_caught(model, wav_path) -> None:
    # Speech that trails off into the list is still not a usable command.
    _say(model, "Turn on the" + _PROMPT)
    assert model.transcribe(wav_path, "en", initial_prompt=_PROMPT) == ""


@pytest.mark.parametrize(
    "echo",
    [
        # How the 1.7B model echoes the list.
        "Office, study, upstairs, top floor, office lamp, desk lamp, kitchen.",
        "The office, study, up stairs, top floor, office lamp.",
        "OFFICE, STUDY, UPSTAIRS.",
        "Office\uff0cStudy\uff0cUpstairs.",  # fullwidth commas
    ],
)
def test_an_echo_in_the_models_own_spelling_is_caught(model, wav_path, echo) -> None:
    _say(model, echo)
    assert model.transcribe(wav_path, "en", initial_prompt=_PROMPT) == ""


def test_an_echo_of_a_short_prompt_is_caught(model, wav_path) -> None:
    _say(model, "Kitchen, office.")
    assert model.transcribe(wav_path, "en", initial_prompt="Kitchen, Office.") == ""


def test_a_name_from_a_short_prompt_is_not_an_echo(model, wav_path) -> None:
    _say(model, "Turn on the kitchen.")
    assert (
        model.transcribe(wav_path, "en", initial_prompt="Kitchen, Office.")
        == "Turn on the kitchen."
    )


def test_an_echo_after_the_language_preamble_is_caught(model, wav_path) -> None:
    # Language not forced: the prompt opens the answer with "language" and the
    # model finishes the preamble (" English<asr_text>") before the transcript.
    _ScriptedDecoder(
        model, _ids(model, " English") + [_ASR_TEXT] + _ids(model, _PROMPT) + [_EOS]
    )
    assert model.transcribe(wav_path, None, initial_prompt=_PROMPT) == ""


def test_speech_after_the_language_preamble_is_kept(model, wav_path) -> None:
    said = "Turn on the Office Lamp."
    _ScriptedDecoder(
        model, _ids(model, " English") + [_ASR_TEXT] + _ids(model, said) + [_EOS]
    )
    assert model.transcribe(wav_path, None, initial_prompt=_PROMPT) == said


def test_an_echo_is_stopped_once_the_names_line_up(model, wav_path) -> None:
    echo = _ids(model, _PROMPT)
    # The first prefix of the output that reads back _N names in a row.
    patterns = _echo_patterns(_PROMPT)
    caught_at = next(
        k
        for k in range(1, len(echo) + 1)
        if any(p in _normalize(model._tokenizer.decode(echo[:k])) for p in patterns)
    )
    assert caught_at < len(echo) // 2

    decoder = _say(model, _PROMPT)
    model.transcribe(wav_path, "en", initial_prompt=_PROMPT)
    # That many tokens are generated (the first by the prefill, the rest by
    # steps); the rest of the list is never read out.
    assert decoder.steps == caught_at - 1


# --- drafts (merged layout) ------------------------------------------------


def _audio() -> np.ndarray:
    return np.zeros(16000, dtype=np.float32)


def test_an_echo_accepted_from_a_draft_is_caught(merged_model) -> None:
    # The whole draft is confirmed by the prefill, so the echo arrives in one
    # go rather than a token at a time; it must still be caught, at once.
    echo = _ids(merged_model, _PROMPT)
    decoder = _ScriptedDecoder(merged_model, echo + [_EOS])
    tokens = merged_model.decode_tokens(_audio(), "en", _PROMPT, draft=echo)
    assert tokens is None
    assert decoder.steps == 0


def test_a_draft_does_not_change_what_is_kept(merged_model) -> None:
    said = _ids(merged_model, "Turn on Upstairs and Top Floor.") + [_EOS]
    _ScriptedDecoder(merged_model, said)
    tokens = merged_model.decode_tokens(_audio(), "en", _PROMPT, draft=said[:3])
    assert tokens == said


# --- streaming -------------------------------------------------------------


class _EchoingStreamingModel(_FakeStreamingModel):
    """The streaming test model, with the passes in ``echo_passes`` echoing."""

    def __init__(self, echo_passes) -> None:
        super().__init__()
        self.echo_passes = set(echo_passes)

    def decode_tokens(
        self, audio, language, initial_prompt, draft=(), run_options=None
    ):
        tokens = super().decode_tokens(
            audio, language, initial_prompt, draft, run_options
        )
        return None if (len(self.calls) - 1) in self.echo_passes else tokens

    def tokens_to_text(self, tokens):
        return "" if tokens is None else super().tokens_to_text(tokens)


def test_an_echoing_pass_is_not_used_as_a_draft() -> None:
    model = _EchoingStreamingModel(echo_passes={0})
    session = _session(model)
    session.accept_chunk(_silence(0.1))
    _drain(session)
    session.accept_chunk(_silence(0.1))
    _drain(session)

    assert model.calls[1] == (3200, [])


def test_speech_after_an_echoing_pass_is_transcribed() -> None:
    # The TV was on before the command; only the final pass heard it out.
    model = _EchoingStreamingModel(echo_passes={0})
    session = _session(model)
    session.accept_chunk(_silence(0.1))
    _drain(session)
    session.accept_chunk(_silence(0.05))

    assert session.finish() == "24 tokens"
    assert model.calls[-1] == (2400, [])


def test_an_echoing_final_pass_gives_an_empty_transcript() -> None:
    model = _EchoingStreamingModel(echo_passes={0})
    session = _session(model, chunk_seconds=10)
    session.accept_chunk(_silence(0.1))
    assert session.finish() == ""


def test_an_echoing_speech_end_pass_is_reused_as_empty() -> None:
    model = _EchoingStreamingModel(echo_passes={0})
    clipper = _SpanClipper()
    session = _session(model, chunk_seconds=10, clipper=clipper)
    clipper.settled = (0, 3200)
    session.accept_chunk(_silence(0.1))
    _drain(session)

    # The pass that heard exactly the final audio was an echo: nothing to redo.
    assert session.finish() == ""
    assert len(model.calls) == 1


def test_a_streaming_echo_gives_an_empty_transcript(merged_model) -> None:
    # The real decode path, through background passes and the final one.
    _say(merged_model, _PROMPT)
    session = qwen3_asr_handler.Qwen3AsrStreamingSession(
        merged_model, "en", _PROMPT, chunk_seconds=0.1
    )
    for _ in range(3):
        session.accept_chunk(_silence(0.1))
        _drain(session)

    assert session.finish() == ""


def test_streaming_speech_is_returned_unchanged(merged_model) -> None:
    said = "Turn on the Office Lamp."
    _say(merged_model, said)
    session = qwen3_asr_handler.Qwen3AsrStreamingSession(
        merged_model, "en", _PROMPT, chunk_seconds=0.1
    )
    for _ in range(3):
        session.accept_chunk(_silence(0.1))
        _drain(session)
    session.accept_chunk(_silence(0.05))

    assert session.finish() == said


# --- the prefix cache ------------------------------------------------------


def test_an_echo_leaves_the_prompts_cached_kv_in_place(merged_model, wav_path) -> None:
    context, _ = merged_model._context_ids(_PROMPT, "en")
    system_key = tuple(qwen3_asr_handler._system_ids(context))

    _say(merged_model, _PROMPT)
    merged_model.transcribe(wav_path, "en", initial_prompt=_PROMPT)

    assert merged_model._prefix_key == system_key
    assert merged_model._prefix_kv is not None


def test_the_utterance_after_an_echo_reuses_the_cached_prompt(
    merged_model, wav_path
) -> None:
    first = _say(merged_model, _PROMPT)
    merged_model.transcribe(wav_path, "en", initial_prompt=_PROMPT)
    assert first.prefills_from_scratch == 1  # the system turn, once
    cached = merged_model._prefix_kv

    second = _say(merged_model, "Turn on the Kitchen.")
    assert (
        merged_model.transcribe(wav_path, "en", initial_prompt=_PROMPT)
        == "Turn on the Kitchen."
    )
    assert second.prefills_from_scratch == 0  # nothing re-read
    assert merged_model._prefix_kv is cached
