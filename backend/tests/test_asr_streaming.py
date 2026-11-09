from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from app.config import get_settings
from app.services.asr import (
    FasterWhisperASRService,
    _TimedWord,
    _WhisperState,
    _append_transcript,
    _common_prefix_count,
    _looks_like_promotional_hallucination,
)


def test_local_agreement_ignores_case_and_punctuation() -> None:
    previous = (
        _TimedWord(0.0, 0.3, " Hello"),
        _TimedWord(0.3, 0.7, " world,"),
    )
    current = (
        _TimedWord(0.0, 0.3, " hello"),
        _TimedWord(0.3, 0.7, " world"),
        _TimedWord(0.7, 1.0, " again"),
    )

    assert _common_prefix_count(previous, current) == 2


def test_stable_prefix_is_committed_and_audio_is_trimmed() -> None:
    settings = replace(
        get_settings(),
        asr_stability_margin_sec=0.8,
        asr_max_window_sec=1,
    )
    service = FasterWhisperASRService(settings)
    previous = (
        _TimedWord(0.0, 0.4, " Hello"),
        _TimedWord(0.4, 0.9, " world"),
    )
    state = _WhisperState(
        forced_language=None,
        samples=np.zeros(32_000, dtype=np.float32),
        full_samples=np.zeros(32_000, dtype=np.float32),
        previous_words=previous,
    )
    current = previous + (_TimedWord(0.9, 1.5, " again"),)

    service._confirm_stable_prefix(state, current)

    assert state.confirmed_text == "Hello world"
    assert state.samples.size == 32_000 - round(0.9 * 16_000)
    assert state.full_samples.size == 32_000
    assert state.previous_words == ()


def test_short_window_keeps_all_audio_context() -> None:
    settings = replace(
        get_settings(),
        asr_stability_margin_sec=0.8,
        asr_max_window_sec=12,
    )
    service = FasterWhisperASRService(settings)
    previous = (_TimedWord(0.0, 0.4, " speaking"),)
    state = _WhisperState(
        forced_language=None,
        samples=np.zeros(32_000, dtype=np.float32),
        full_samples=np.zeros(32_000, dtype=np.float32),
        previous_words=previous,
    )

    service._confirm_stable_prefix(state, previous)

    assert state.samples.size == 32_000
    assert state.confirmed_text == ""
    assert state.previous_words == previous


def test_transcript_join_preserves_japanese_without_extra_space() -> None:
    assert _append_transcript("今日は", "いい天気です") == "今日はいい天気です"
    assert _append_transcript("Hello", " world") == "Hello world"


def test_low_confidence_or_no_speech_segments_are_rejected() -> None:
    service = FasterWhisperASRService(get_settings())

    assert service._segment_is_reliable(
        SimpleNamespace(no_speech_prob=0.1, avg_logprob=-0.2, compression_ratio=1.1)
    )
    assert not service._segment_is_reliable(
        SimpleNamespace(no_speech_prob=0.9, avg_logprob=-0.2, compression_ratio=1.1)
    )
    assert not service._segment_is_reliable(
        SimpleNamespace(no_speech_prob=0.1, avg_logprob=-1.5, compression_ratio=1.1)
    )


def test_detected_language_is_reused_for_the_rest_of_utterance() -> None:
    settings = replace(get_settings(), asr_max_window_sec=12)
    service = FasterWhisperASRService(settings)

    class Model:
        def __init__(self) -> None:
            self.languages: list[str | None] = []

        def transcribe(self, _audio, **options):
            self.languages.append(options["language"])
            word = SimpleNamespace(start=0.0, end=0.4, word=" Hello")
            segment = SimpleNamespace(
                words=[word],
                text="Hello",
                no_speech_prob=0.1,
                avg_logprob=-0.2,
                compression_ratio=1.0,
            )
            return iter([segment]), SimpleNamespace(language="en")

    model = Model()
    service._models["small"] = model
    state = _WhisperState(
        forced_language=None,
        samples=np.zeros(16_000, dtype=np.float32),
        full_samples=np.zeros(16_000, dtype=np.float32),
    )

    service._decode_sync(state, final=False)
    service._decode_sync(state, final=False)

    assert model.languages == [None, "en"]
    assert state.detected_language == "en"


def _segment(
    text: str,
    start: float,
    end: float,
    *,
    no_speech_prob: float = 0.1,
    avg_logprob: float = -0.2,
):
    word = SimpleNamespace(start=start, end=end, word=f" {text}")
    return SimpleNamespace(
        start=start,
        end=end,
        words=[word],
        text=text,
        no_speech_prob=no_speech_prob,
        avg_logprob=avg_logprob,
        compression_ratio=1.0,
    )


def _decode_final(
    segments,
    *,
    duration_after_vad: float = 2.0,
    speech_intervals: tuple[tuple[float, float], ...] | None = None,
):
    settings = get_settings()
    service = FasterWhisperASRService(settings)

    class Model:
        def __init__(self) -> None:
            self.options = None

        def transcribe(self, _audio, **options):
            self.options = options
            return iter(segments), SimpleNamespace(
                language="en",
                duration_after_vad=duration_after_vad,
            )

    model = Model()
    service._models["small"] = model
    if speech_intervals is None:
        speech_intervals = (
            ((0.0, duration_after_vad),) if duration_after_vad > 0 else ()
        )
    service._speech_intervals = lambda _audio: speech_intervals
    state = _WhisperState(
        forced_language=None,
        samples=np.zeros(32_000, dtype=np.float32),
        full_samples=np.zeros(32_000, dtype=np.float32),
    )
    service._decode_sync(state, final=True)
    return state, model


def test_final_decode_drops_weak_hallucination_after_real_speech() -> None:
    state, _model = _decode_final(
        [
            _segment("Hello", 0.0, 0.5),
            _segment(
                "Please subscribe to the channel",
                1.2,
                2.0,
                no_speech_prob=0.9,
            ),
        ]
    )

    assert state.text == "Hello"


def test_final_decode_drops_confident_segment_outside_vad_speech() -> None:
    state, _model = _decode_final(
        [
            _segment("Hello", 0.0, 0.5),
            _segment("Unrelated generated sentence", 1.2, 2.0),
        ],
        speech_intervals=((0.0, 0.6),),
    )

    assert state.text == "Hello"


def test_final_decode_keeps_a_weak_fast_segment_between_reliable_speech() -> None:
    state, _model = _decode_final(
        [
            _segment("I", 0.0, 0.3),
            _segment(
                "am speaking quickly",
                0.3,
                0.8,
                no_speech_prob=0.75,
                avg_logprob=-1.2,
            ),
            _segment("today", 0.8, 1.2),
        ]
    )

    assert state.text == "I am speaking quickly today"


def test_final_decode_rejects_confident_text_when_vad_found_no_speech() -> None:
    state, _model = _decode_final(
        [_segment("Thank you", 0.0, 0.5)],
        duration_after_vad=0.0,
    )

    assert state.text == ""
    assert state.detected_language is None


def test_promotional_hallucination_is_rejected_even_when_confident() -> None:
    text = (
        "Hãy subscribe cho kênh La La School "
        "để không bỏ lỡ những video hấp dẫn"
    )
    state, _model = _decode_final([_segment(text, 0.0, 2.0)])

    assert _looks_like_promotional_hallucination(text)
    assert state.text == ""
    assert state.detected_language is None


def test_normal_sentence_about_school_is_not_a_promotional_hallucination() -> None:
    assert not _looks_like_promotional_hallucination(
        "Hôm nay trường La La School tổ chức một buổi học mới"
    )
    assert not _looks_like_promotional_hallucination(
        "Tôi đã subscribe cho một channel về lập trình"
    )


def test_decode_uses_hallucination_resistant_whisper_options() -> None:
    _state, model = _decode_final([_segment("Hello", 0.0, 0.5)])

    assert model.options["condition_on_previous_text"] is False
    assert model.options["hallucination_silence_threshold"] == 0.5
    assert model.options["vad_parameters"]["threshold"] == 0.6
    assert model.options["vad_parameters"]["min_speech_duration_ms"] == 250
