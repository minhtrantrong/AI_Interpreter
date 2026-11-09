from __future__ import annotations

import asyncio
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from app.config import Settings


LANGUAGE_NAMES = {
    "auto": None,
    "vi": "vi",
    "en": "en",
    "ja": "ja",
    "ko": "ko",
}

logger = logging.getLogger("translate-realtime.asr")


@dataclass(slots=True)
class ASRSnapshot:
    language: str | None
    text: str


class ASRService(Protocol):
    loaded: bool
    available_models: dict[str, str]

    async def load(self, model_key: str | None = None) -> None: ...
    async def new_state(
        self,
        language: str | None = None,
        model_key: str | None = None,
    ) -> Any: ...
    async def transcribe(self, samples: np.ndarray, state: Any) -> ASRSnapshot: ...
    async def finish(self, state: Any) -> ASRSnapshot: ...
    def is_model_loaded(self, model_key: str) -> bool: ...


@dataclass(frozen=True, slots=True)
class _TimedWord:
    start: float
    end: float
    text: str


@dataclass(slots=True)
class _WhisperState:
    forced_language: str | None
    model_key: str = "small"
    samples: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float32)
    )
    full_samples: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float32)
    )
    confirmed_text: str = ""
    previous_words: tuple[_TimedWord, ...] = ()
    detected_language: str | None = None
    text: str = ""
    dirty: bool = False


class FasterWhisperASRService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.available_models = dict(settings.asr_models)
        self.loaded = False
        self._models: dict[str, Any] = {}
        self._load_lock = asyncio.Lock()
        self._inference_lock = asyncio.Lock()

    async def load(self, model_key: str | None = None) -> None:
        key = self._validate_model(model_key or self.settings.asr_default_model)
        if key in self._models:
            return
        async with self._load_lock:
            if key in self._models:
                return
            # Load synchronously while holding lock to prevent race conditions
            self._models[key] = await asyncio.to_thread(self._load_sync, key)
            # Set loaded flag AFTER model is fully loaded
            self.loaded = True

    def _load_sync(self, model_key: str) -> Any:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "Thiếu faster-whisper. Hãy cài backend bằng: pip install -e '.[models]'"
            ) from exc

        return WhisperModel(
            self.available_models[model_key],
            device=self.settings.asr_device,
            compute_type=self.settings.asr_compute_type,
            cpu_threads=self.settings.asr_cpu_threads,
            num_workers=self.settings.asr_num_workers,
        )

    async def new_state(
        self,
        language: str | None = None,
        model_key: str | None = None,
    ) -> _WhisperState:
        key = self._validate_model(model_key or self.settings.asr_default_model)
        await self.load(key)
        return _WhisperState(forced_language=language, model_key=key)

    def is_model_loaded(self, model_key: str) -> bool:
        key = self._validate_model(model_key)
        return key in self._models

    def _validate_model(self, model_key: str) -> str:
        if model_key not in self.available_models:
            supported = ", ".join(self.available_models)
            raise ValueError(
                f"Model STT không hợp lệ: {model_key}. Hỗ trợ: {supported}"
            )
        return model_key

    async def transcribe(
        self,
        samples: np.ndarray,
        state: _WhisperState,
    ) -> ASRSnapshot:
        if samples.size:
            incoming = samples.astype(np.float32, copy=False)
            state.samples = np.concatenate((state.samples, incoming))
            state.full_samples = np.concatenate((state.full_samples, incoming))
            state.dirty = True
        minimum = round(
            self.settings.audio_sample_rate * self.settings.asr_min_audio_sec
        )
        if state.samples.size < minimum:
            return self._snapshot(state)
        await self._decode(state, final=False)
        return self._snapshot(state)

    async def finish(self, state: _WhisperState) -> ASRSnapshot:
        if state.samples.size:
            await self._decode(state, final=True)
        elif state.confirmed_text:
            state.text = state.confirmed_text
        return self._snapshot(state)

    async def _decode(self, state: _WhisperState, *, final: bool) -> None:
        async with self._inference_lock:
            await asyncio.to_thread(self._decode_sync, state, final)

    def _decode_sync(self, state: _WhisperState, final: bool) -> None:
        audio = state.full_samples if final else state.samples
        language_hint = state.forced_language or state.detected_language
        model = self._models[state.model_key]
        segments, info = model.transcribe(
            audio,
            language=language_hint,
            task="transcribe",
            beam_size=(
                self.settings.asr_final_beam_size
                if final
                else self.settings.asr_beam_size
            ),
            temperature=0.0,
            # Carrying decoder text across windows is a common source of
            # repetition loops when a window contains mostly silence.
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters={
                "threshold": self.settings.asr_vad_threshold,
                "min_speech_duration_ms": self.settings.asr_vad_min_speech_ms,
                "min_silence_duration_ms": 350,
                "speech_pad_ms": 150,
            },
            compression_ratio_threshold=2.4,
            log_prob_threshold=self.settings.asr_min_avg_logprob,
            no_speech_threshold=self.settings.asr_max_no_speech_prob,
            word_timestamps=True,
            hallucination_silence_threshold=0.5,
        )
        generated = list(segments)
        speech_intervals = (
            self._speech_intervals(audio) if final and generated else None
        )
        materialized = (
            self._select_final_segments(generated, speech_intervals)
            if final
            else [
                segment
                for segment in generated
                if self._segment_is_reliable(segment)
            ]
        )
        if not self._has_vad_speech(info):
            materialized = []
        candidate_text = " ".join(
            str(getattr(segment, "text", "")).strip()
            for segment in materialized
            if str(getattr(segment, "text", "")).strip()
        )
        if (
            self.settings.asr_suppress_promotional_hallucinations
            and _looks_like_promotional_hallucination(candidate_text)
        ):
            logger.info("Rejected promotional ASR hallucination")
            materialized = []
        if final and generated and not materialized:
            logger.info(
                "Rejected final ASR hypothesis without reliable speech "
                "(segments=%d, vad_speech=%.3fs)",
                len(generated),
                self._vad_speech_seconds(info),
            )
        words = tuple(
            _TimedWord(word.start, word.end, word.word)
            for segment in materialized
            for word in (segment.words or ())
            if word.word.strip()
        )
        window_text = (
            "".join(word.text for word in words)
            if words
            else " ".join(
                segment.text.strip()
                for segment in materialized
                if segment.text.strip()
            ).strip()
        )
        state.text = (
            window_text.strip()
            if final
            else _append_transcript(state.confirmed_text, window_text)
        )
        if state.forced_language:
            state.detected_language = state.forced_language
        elif materialized and state.detected_language is None:
            state.detected_language = getattr(info, "language", None)
        state.dirty = False

        if final:
            state.confirmed_text = state.text
            state.previous_words = ()
            return

        self._confirm_stable_prefix(state, words)

    def _segment_is_reliable(self, segment: Any) -> bool:
        no_speech_prob = float(getattr(segment, "no_speech_prob", 0.0))
        avg_logprob = float(getattr(segment, "avg_logprob", 0.0))
        compression_ratio = float(getattr(segment, "compression_ratio", 0.0))
        return (
            no_speech_prob <= self.settings.asr_max_no_speech_prob
            and avg_logprob >= self.settings.asr_min_avg_logprob
            and compression_ratio <= 2.4
        )

    def _select_final_segments(
        self,
        segments: list[Any],
        speech_intervals: tuple[tuple[float, float], ...] | None = None,
    ) -> list[Any]:
        """Keep strong segments and only rescue weak speech inside a strong span.

        Whisper can append plausible phrases to leading or trailing silence.  The
        previous implementation accepted every final segment after seeing a single
        speech-like one.  A weak segment is now retained only when reliable speech
        bounds it on both sides, which preserves fast words in the middle without
        accepting a hallucinated ending.
        """
        reliable = [
            self._segment_is_reliable(segment)
            and self._segment_overlaps_speech(segment, speech_intervals)
            for segment in segments
        ]
        selected = reliable.copy()
        index = 0
        while index < len(segments):
            if reliable[index]:
                index += 1
                continue
            end = index
            while end < len(segments) and not reliable[end]:
                end += 1
            bounded = index > 0 and end < len(segments)
            weak_group = segments[index:end]
            if (
                bounded
                and all(
                    self._segment_is_bridge_candidate(item)
                    and self._segment_overlaps_speech(item, speech_intervals)
                    for item in weak_group
                )
                and self._segments_are_contiguous(segments[index - 1 : end + 1])
            ):
                for accepted in range(index, end):
                    selected[accepted] = True
            index = end
        return [segment for segment, keep in zip(segments, selected) if keep]

    def _speech_intervals(
        self,
        audio: np.ndarray,
    ) -> tuple[tuple[float, float], ...] | None:
        """Run an unpadded final VAD pass for segment/audio alignment."""
        try:
            from faster_whisper.vad import VadOptions, get_speech_timestamps

            chunks = get_speech_timestamps(
                audio,
                VadOptions(
                    threshold=self.settings.asr_vad_threshold,
                    min_speech_duration_ms=self.settings.asr_vad_min_speech_ms,
                    min_silence_duration_ms=350,
                    speech_pad_ms=0,
                ),
                sampling_rate=self.settings.audio_sample_rate,
            )
        except Exception:
            # Transcription already ran its own VAD. Fail open if the extra
            # alignment pass is unavailable rather than dropping real speech.
            logger.exception("Could not calculate final VAD alignment")
            return None

        sample_rate = self.settings.audio_sample_rate
        return tuple(
            (chunk["start"] / sample_rate, chunk["end"] / sample_rate)
            for chunk in chunks
        )

    @staticmethod
    def _segment_overlaps_speech(
        segment: Any,
        speech_intervals: tuple[tuple[float, float], ...] | None,
    ) -> bool:
        if speech_intervals is None:
            return True
        start = float(getattr(segment, "start", -1.0))
        end = float(getattr(segment, "end", -1.0))
        duration = end - start
        if duration <= 0:
            return False
        overlap = sum(
            max(0.0, min(end, speech_end) - max(start, speech_start))
            for speech_start, speech_end in speech_intervals
        )
        return overlap >= max(0.06, duration * 0.35)

    def _segment_is_bridge_candidate(self, segment: Any) -> bool:
        no_speech_prob = float(getattr(segment, "no_speech_prob", 1.0))
        avg_logprob = float(getattr(segment, "avg_logprob", -10.0))
        compression_ratio = float(getattr(segment, "compression_ratio", 10.0))
        return (
            no_speech_prob
            <= min(0.85, self.settings.asr_max_no_speech_prob + 0.15)
            and avg_logprob >= self.settings.asr_min_avg_logprob - 0.35
            and compression_ratio <= 2.4
        )

    @staticmethod
    def _segments_are_contiguous(segments: list[Any]) -> bool:
        for previous, current in zip(segments, segments[1:]):
            previous_end = getattr(previous, "end", None)
            current_start = getattr(current, "start", None)
            if previous_end is None or current_start is None:
                return False
            if float(current_start) - float(previous_end) > 0.6:
                return False
        return True

    def _has_vad_speech(self, info: Any) -> bool:
        duration = getattr(info, "duration_after_vad", None)
        if duration is None:
            # Compatibility with test doubles and older faster-whisper releases.
            return True
        return self._vad_speech_seconds(info) >= (
            self.settings.asr_vad_min_speech_ms / 1000
        )

    @staticmethod
    def _vad_speech_seconds(info: Any) -> float:
        return max(0.0, float(getattr(info, "duration_after_vad", 0.0) or 0.0))

    def _confirm_stable_prefix(
        self,
        state: _WhisperState,
        words: tuple[_TimedWord, ...],
    ) -> None:
        if not words:
            state.previous_words = ()
            return

        duration = state.samples.size / self.settings.audio_sample_rate
        if duration <= self.settings.asr_max_window_sec:
            state.previous_words = words
            return

        stable_before = max(0.0, duration - self.settings.asr_stability_margin_sec)
        common_count = _common_prefix_count(state.previous_words, words)
        stable_count = sum(
            1 for word in words[:common_count] if word.end <= stable_before
        )

        if not stable_count and duration > self.settings.asr_max_window_sec * 1.5:
            stable_count = sum(1 for word in words if word.end <= stable_before)

        if not stable_count:
            state.previous_words = words
            return

        stable_words = words[:stable_count]
        state.confirmed_text = _append_transcript(
            state.confirmed_text,
            "".join(word.text for word in stable_words),
        )
        cut_samples = min(
            state.samples.size,
            max(1, round(stable_words[-1].end * self.settings.audio_sample_rate)),
        )
        state.samples = state.samples[cut_samples:].copy()
        state.previous_words = ()

    @staticmethod
    def _snapshot(state: _WhisperState) -> ASRSnapshot:
        return ASRSnapshot(
            language=state.detected_language,
            text=state.text,
        )


@dataclass(slots=True)
class _FakeState:
    calls: int = 0
    text: str = ""
    language: str = "vi"
    model_key: str = "small"


class FakeASRService:
    """Small deterministic adapter for testing the WebSocket without a GPU."""

    loaded = True
    available_models = {
        "small": "Systran/faster-whisper-small",
        "turbo": "large-v3-turbo",
    }
    _steps = (
        "Xin chào",
        "Xin chào, rất vui",
        "Xin chào, rất vui được gặp bạn.",
    )

    async def load(self, model_key: str | None = None) -> None:
        if model_key is not None and model_key not in self.available_models:
            raise ValueError(f"Model STT không hợp lệ: {model_key}")
        return None

    async def new_state(
        self,
        language: str | None = None,
        model_key: str | None = None,
    ) -> _FakeState:
        key = model_key or "small"
        await self.load(key)
        return _FakeState(language=language or "vi", model_key=key)

    def is_model_loaded(self, model_key: str) -> bool:
        return model_key in self.available_models

    async def transcribe(self, samples: np.ndarray, state: _FakeState) -> ASRSnapshot:
        if samples.size:
            state.calls += 1
            state.text = self._steps[min(state.calls - 1, len(self._steps) - 1)]
        return ASRSnapshot(language=state.language, text=state.text)

    async def finish(self, state: _FakeState) -> ASRSnapshot:
        if state.text:
            state.text = self._steps[-1]
        return ASRSnapshot(language=state.language, text=state.text)


def create_asr_service(settings: Settings) -> ASRService:
    if settings.use_fake_models:
        return FakeASRService()
    return FasterWhisperASRService(settings)


_WORD_KEY = re.compile(r"\w+", re.UNICODE)


def _word_key(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return "".join(_WORD_KEY.findall(normalized)) or normalized.strip()


def _looks_like_promotional_hallucination(text: str) -> bool:
    if not text:
        return False
    normalized = unicodedata.normalize("NFKD", text).casefold()
    normalized = "".join(
        character for character in normalized
        if not unicodedata.combining(character)
    ).replace("đ", "d")
    words = set(_WORD_KEY.findall(normalized))
    asks_to_subscribe = "subscribe" in words or {"dang", "ky"} <= words
    mentions_media = bool(words & {"kenh", "channel", "video"})
    call_to_action = (
        bool(words & {"hay", "please"})
        or {"khong", "bo", "lo"} <= words
        or {"dont", "miss"} <= words
    )
    return asks_to_subscribe and mentions_media and call_to_action


def _common_prefix_count(
    previous: tuple[_TimedWord, ...],
    current: tuple[_TimedWord, ...],
) -> int:
    count = 0
    for old, new in zip(previous, current):
        if _word_key(old.text) != _word_key(new.text):
            break
        count += 1
    return count


def _append_transcript(committed: str, fragment: str) -> str:
    if not committed:
        return fragment.strip()
    if not fragment:
        return committed.strip()
    separator = ""
    if (
        not fragment[0].isspace()
        and _word_key(fragment[0])
        and not _is_japanese_script(committed[-1])
        and not _is_japanese_script(fragment[0])
    ):
        separator = " "
    return f"{committed.rstrip()}{separator}{fragment}".strip()


def _is_japanese_script(char: str) -> bool:
    return (
        "\u3040" <= char <= "\u30ff"
        or "\u31f0" <= char <= "\u31ff"
        or "\u4e00" <= char <= "\u9fff"
    )
