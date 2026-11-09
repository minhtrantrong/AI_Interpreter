from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from collections.abc import Awaitable, Callable
from contextlib import suppress
from typing import Any

import numpy as np

from app.audio import PCM16Decoder, root_mean_square
from app.config import Settings
from app.language_detection import StableLanguageDetector
from app.services.asr import ASRService, ASRSnapshot, LANGUAGE_NAMES
from app.services.translator import Translator, translation_is_ready


SUPPORTED_LANGUAGES = frozenset(LANGUAGE_NAMES)
EventSink = Callable[[dict[str, Any]], Awaitable[None]]
logger = logging.getLogger("translate-realtime.session")


class TextInputSession:
    """Handle text input-only session (no audio)."""

    def __init__(
        self,
        settings: Settings,
        translator: Translator,
        source_language: str,
        target_language: str,
        event_sink: EventSink | None = None,
    ) -> None:
        self.settings = settings
        self.translator = translator
        self.source_language = self._validate_source(source_language)
        self.target_language = self._validate_target(target_language)
        self._event_sink = event_sink
        self._sequence = 0
        self._context: deque[str] = deque(maxlen=settings.translation_context_turns)

    async def translate_text(
        self,
        text: str,
        source_language: str | None = None,
    ) -> list[dict[str, Any]]:
        """Translate a text input directly."""
        events: list[dict[str, Any]] = []
        if not text.strip():
            return events

        source = source_language or self.source_language
        detected = self._resolve_language_from_text(text, source)

        # Emit transcript event
        events.append(self._transcript_event(text, detected, final=True))

        # Translate immediately
        translation = await self._translate(text, detected)
        if translation:
            events.append(
                self._translation_event(
                    translation,
                    text,
                    final=True,
                    target_language=self.target_language,
                )
            )
            self._append_context(text, translation, detected)
            events.append(self._event("utterance.end", reason="text_input"))

        return events

    async def _translate(self, text: str, detected_language: str | None = None) -> str:
        return await self.translator.translate(
            text=text,
            source_language=self._translation_source(detected_language),
            target_language=self.target_language,
            context=list(self._context),
        )

    def _translation_source(self, detected_language: str | None) -> str:
        if self.source_language == "auto":
            return detected_language or "auto"
        return self.source_language

    def _resolve_language_from_text(self, text: str, source: str) -> str | None:
        if source != "auto":
            return source
        # Use script-based detection for text input
        from app.language_detection import language_from_script
        return language_from_script(text) or None

    def _append_context(
        self,
        text: str,
        translation: str,
        detected_language: str | None,
    ) -> None:
        self._context.append(
            f"{detected_language or self.source_language}: {text}\n"
            f"{self.target_language}: {translation}"
        )

    def _transcript_event(
        self,
        text: str,
        detected_language: str | None,
        final: bool,
    ) -> dict[str, Any]:
        return self._event(
            "transcript.final" if final else "transcript.partial",
            text=text,
            language=detected_language,
            is_final=final,
        )

    def _translation_event(
        self,
        translation: str,
        source_text: str,
        final: bool,
        target_language: str,
    ) -> dict[str, Any]:
        return self._event(
            "translation.final" if final else "translation.partial",
            text=translation,
            source_text=source_text,
            language=target_language,
            is_final=final,
        )

    def _event(self, event_type: str, **payload: Any) -> dict[str, Any]:
        self._sequence += 1
        return {
            "type": event_type,
            "sequence": self._sequence,
            "timestamp": time.time(),
            **payload,
        }

    @staticmethod
    def _validate_source(language: str) -> str:
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Ngôn ngữ nguồn không hợp lệ: {language}")
        return language

    @staticmethod
    def _validate_target(language: str) -> str:
        if language not in SUPPORTED_LANGUAGES - {"auto"}:
            raise ValueError(f"Ngôn ngữ đích không hợp lệ: {language}")
        return language


class RealtimeSession:
    def __init__(
        self,
        settings: Settings,
        asr: ASRService,
        translator: Translator,
        source_language: str,
        target_language: str,
        asr_model: str | None = None,
        event_sink: EventSink | None = None,
    ) -> None:
        self.settings = settings
        self.asr = asr
        self.translator = translator
        self.source_language = self._validate_source(source_language)
        self.target_language = self._validate_target(target_language)
        self.asr_model = self._validate_asr_model(
            asr_model or settings.asr_default_model
        )
        self._event_sink = event_sink

        self._decoder = PCM16Decoder()
        self._state: Any = None
        self._pending_audio = np.empty(0, dtype=np.float32)
        self._pre_roll_audio = np.empty(0, dtype=np.float32)
        self._has_speech = False
        self._speech_candidate_samples = 0
        self._speech_samples = 0
        self._silence_samples = 0
        self._noise_rms = 0.0
        self._last_transcript = ""
        self._last_translated_text = ""
        self._last_translation_at = 0.0
        self._sequence = 0
        self._context: deque[str] = deque(maxlen=settings.translation_context_turns)
        self._language_detector = StableLanguageDetector()
        self._translation_lock = asyncio.Lock()
        self._partial_translation_task: asyncio.Task[None] | None = None
        self._final_translation_tasks: set[asyncio.Task[None]] = set()
        self._closed = False

    async def start(self) -> None:
        self._state = await self.asr.new_state(
            self._forced_asr_language(),
            self.asr_model,
        )

    async def feed_audio(self, payload: bytes) -> list[dict[str, Any]]:
        samples = self._decoder.decode(payload)
        if samples.size == 0:
            return []

        rms = root_mean_square(samples)
        threshold = max(
            self.settings.speech_rms_threshold,
            self._noise_rms * self.settings.noise_floor_multiplier,
        )
        activated_now = False
        if not self._has_speech:
            self._append_pre_roll(samples)
            if rms >= threshold:
                self._speech_candidate_samples += int(samples.size)
                speech_start_limit = round(
                    self.settings.audio_sample_rate
                    * self.settings.speech_start_ms
                    / 1000
                )
                if self._speech_candidate_samples >= speech_start_limit:
                    self._has_speech = True
                    activated_now = True
                    self._speech_samples = self._speech_candidate_samples
                    self._pending_audio = np.concatenate(
                        (self._pending_audio, self._pre_roll_audio)
                    )
                    self._pre_roll_audio = np.empty(0, dtype=np.float32)
            else:
                self._speech_candidate_samples = 0
                self._update_noise_floor(rms)

            if not self._has_speech:
                return []

        if not activated_now and rms >= threshold:
            self._speech_samples += int(samples.size)
            self._silence_samples = 0
        elif not activated_now:
            self._silence_samples += int(samples.size)

        if not activated_now:
            self._pending_audio = np.concatenate((self._pending_audio, samples))
        events: list[dict[str, Any]] = []
        step = self.settings.asr_step_samples

        while self._pending_audio.size >= step:
            chunk = self._pending_audio[:step]
            self._pending_audio = self._pending_audio[step:]
            snapshot = await self.asr.transcribe(chunk, self._state)
            events.extend(await self._partial_events(snapshot, rms))

        silence_limit = round(
            self.settings.audio_sample_rate * self.settings.silence_finalize_ms / 1000
        )
        utterance_limit = self.settings.audio_sample_rate * self.settings.max_utterance_sec
        if self._has_speech and self._silence_samples >= silence_limit:
            events.extend(await self.commit("silence"))
        elif self._speech_samples >= utterance_limit:
            events.extend(await self.commit("max_duration"))

        return events

    def _append_pre_roll(self, samples: np.ndarray) -> None:
        self._pre_roll_audio = np.concatenate((self._pre_roll_audio, samples))
        max_samples = round(
            self.settings.audio_sample_rate * self.settings.speech_preroll_ms / 1000
        )
        if self._pre_roll_audio.size > max_samples:
            self._pre_roll_audio = self._pre_roll_audio[-max_samples:].copy()

    def _update_noise_floor(self, rms: float) -> None:
        if rms <= 0:
            return
        if self._noise_rms == 0:
            self._noise_rms = rms
            return
        # Adaptive alpha: faster update when RMS increases (new noise source)
        # slower decay when RMS decreases (noise fading away)
        alpha = 0.08 if rms > self._noise_rms else 0.02
        self._noise_rms = self._noise_rms * (1 - alpha) + rms * alpha

    async def commit(
        self,
        reason: str = "client",
        background_translation: bool = True,
    ) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        if self._state is None:
            await self.start()

        if self._pending_audio.size:
            snapshot = await self.asr.transcribe(self._pending_audio, self._state)
            self._pending_audio = np.empty(0, dtype=np.float32)
            if snapshot.text and snapshot.text != self._last_transcript:
                detected = self._resolve_language(snapshot, final=False)
                events.append(
                    self._transcript_event(snapshot, detected, final=False)
                )

        snapshot = await self.asr.finish(self._state)
        if snapshot.text:
            detected = self._resolve_language(snapshot, final=True)
            await self._cancel_partial_translation()
            events.append(self._transcript_event(snapshot, detected, final=True))
            if self._event_sink is not None and background_translation:
                self._schedule_final_translation(snapshot.text, detected, reason)
            else:
                translation = await self._translate(snapshot.text, detected)
                if translation:
                    events.append(
                        self._translation_event(
                            translation,
                            snapshot.text,
                            final=True,
                            target_language=self.target_language,
                        )
                    )
                    self._append_context(snapshot.text, translation, detected)
                events.append(self._event("utterance.end", reason=reason))
        else:
            await self._cancel_partial_translation()
            events.extend(self._clear_partial_events(snapshot))
            events.append(self._event("utterance.end", reason=reason))
        await self._reset_utterance()
        return events

    async def reset(self, clear_context: bool = False) -> list[dict[str, Any]]:
        await self._cancel_translation_tasks(cancel_final=True)
        if clear_context:
            self._context.clear()
            self._language_detector.clear()
        await self._reset_utterance()
        return [self._event("session.reset", context_cleared=clear_context)]

    async def set_languages(self, source: str, target: str) -> list[dict[str, Any]]:
        source = self._validate_source(source)
        target = self._validate_target(target)
        events: list[dict[str, Any]] = []
        if self._last_transcript or self._has_speech:
            events.extend(
                await self.commit(
                    "language_change",
                    background_translation=False,
                )
            )
        self.source_language = source
        self.target_language = target
        self._language_detector.clear()
        await self._reset_utterance()
        self._last_translated_text = ""
        events.append(
            self._event(
                "session.languages",
                source_language=source,
                target_language=target,
            )
        )
        return events

    async def set_asr_model(self, model_key: str) -> list[dict[str, Any]]:
        model_key = self._validate_asr_model(model_key)
        if model_key == self.asr_model:
            return [
                self._event(
                    "session.model",
                    asr_model=model_key,
                    model_loaded=self.asr.is_model_loaded(model_key),
                )
            ]

        events: list[dict[str, Any]] = []
        if self._last_transcript or self._has_speech:
            events.extend(
                await self.commit(
                    "model_change",
                    background_translation=False,
                )
            )

        loading_event = self._event(
            "session.model_loading",
            asr_model=model_key,
            cached=self.asr.is_model_loaded(model_key),
        )
        if self._event_sink is not None:
            for event in events:
                await self._event_sink(event)
            events.clear()
            await self._event_sink(loading_event)
        else:
            events.append(loading_event)

        await self.asr.load(model_key)
        self.asr_model = model_key
        self._language_detector.clear()
        await self._reset_utterance()
        events.append(
            self._event(
                "session.model",
                asr_model=model_key,
                model_loaded=True,
            )
        )
        return events

    async def translate_text(self, text: str) -> list[dict[str, Any]]:
        """Translate a text input directly (no audio processing)."""
        events: list[dict[str, Any]] = []
        if not text.strip():
            return events

        # Detect language from text
        from app.language_detection import language_from_script, language_from_text
        detected = language_from_script(text)
        if detected is None:
            lang, _ = language_from_text(text)
            detected = lang
        if detected is None and self.source_language != "auto":
            detected = self.source_language

        source = self._translation_source(detected)

        # Emit transcript event
        events.append(self._transcript_event_text(text, detected))

        # Translate immediately (synchronous)
        translation = await self._translate(text, detected)
        if translation:
            events.append(
                self._translation_event(
                    translation,
                    text,
                    final=True,
                    target_language=self.target_language,
                )
            )
            self._append_context(text, translation, detected)
            events.append(self._event("utterance.end", reason="text_input"))

        return events

    def _transcript_event_text(
        self,
        text: str,
        detected_language: str | None,
    ) -> dict[str, Any]:
        return self._event(
            "transcript.final",
            text=text,
            language=detected_language,
            is_final=True,
            source="text",
        )

    async def _partial_events(
        self, snapshot: ASRSnapshot, rms: float
    ) -> list[dict[str, Any]]:
        if snapshot.text == self._last_transcript:
            return []
        if not snapshot.text:
            await self._cancel_partial_translation()
            return self._clear_partial_events(snapshot)

        self._last_transcript = snapshot.text
        detected = self._resolve_language(snapshot, final=False)
        events = [
            self._transcript_event(snapshot, detected, final=False, rms=rms)
        ]
        now = time.monotonic()
        interval = self.settings.translation_interval_ms / 1000
        if (
            snapshot.text != self._last_translated_text
            and now - self._last_translation_at >= interval
            and translation_is_ready(snapshot.text)
            and (detected is not None or self.source_language != "auto")
        ):
            self._last_translated_text = snapshot.text
            self._last_translation_at = now
            if self._event_sink is not None:
                self._schedule_partial_translation(snapshot.text, detected)
            else:
                translation = await self._translate(snapshot.text, detected)
                if translation:
                    events.append(
                        self._translation_event(
                            translation,
                            snapshot.text,
                            final=False,
                            target_language=self.target_language,
                        )
                    )
        return events

    def _clear_partial_events(self, snapshot: ASRSnapshot) -> list[dict[str, Any]]:
        if not self._last_transcript and not self._last_translated_text:
            return []
        events: list[dict[str, Any]] = []
        if self._last_transcript:
            events.append(
                self._transcript_event(snapshot, None, final=False)
            )
        if self._last_translated_text:
            events.append(
                self._translation_event(
                    "",
                    "",
                    final=False,
                    target_language=self.target_language,
                )
            )
        self._last_transcript = ""
        self._last_translated_text = ""
        return events

    def _schedule_partial_translation(
        self,
        text: str,
        detected_language: str | None,
    ) -> None:
        if self._partial_translation_task is not None:
            self._partial_translation_task.cancel()
        task = asyncio.create_task(
            self._run_background_translation(
                text=text,
                source_language=self._translation_source(detected_language),
                detected_language=detected_language,
                target_language=self.target_language,
                context=list(self._context),
                final=False,
            )
        )
        self._partial_translation_task = task
        task.add_done_callback(self._partial_translation_done)

    def _schedule_final_translation(
        self,
        text: str,
        detected_language: str | None,
        reason: str,
    ) -> None:
        task = asyncio.create_task(
            self._run_background_translation(
                text=text,
                source_language=self._translation_source(detected_language),
                detected_language=detected_language,
                target_language=self.target_language,
                context=list(self._context),
                final=True,
                end_reason=reason,
            )
        )
        self._final_translation_tasks.add(task)
        task.add_done_callback(self._final_translation_done)

    async def _run_background_translation(
        self,
        *,
        text: str,
        source_language: str,
        detected_language: str | None,
        target_language: str,
        context: list[str],
        final: bool,
        end_reason: str | None = None,
    ) -> None:
        try:
            if not final:
                await asyncio.sleep(self.settings.translation_debounce_ms / 1000)
            async with self._translation_lock:
                translation = await self.translator.translate(
                    text=text,
                    source_language=source_language,
                    target_language=target_language,
                    context=context,
                )
            if self._closed or not translation or self._event_sink is None:
                return
            await self._event_sink(
                self._translation_event(
                    translation,
                    text,
                    final=final,
                    target_language=target_language,
                )
            )
            if final:
                self._append_context(
                    text,
                    translation,
                    detected_language,
                    target_language,
                )
                await self._event_sink(
                    self._event("utterance.end", reason=end_reason or "silence")
                )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Background translation failed")
            if final and not self._closed and self._event_sink is not None:
                await self._event_sink(
                    self._event(
                        "translation.error",
                        message="Không thể dịch câu vừa nhận.",
                    )
                )
                await self._event_sink(
                    self._event("utterance.end", reason=end_reason or "silence")
                )

    def _partial_translation_done(self, task: asyncio.Task[None]) -> None:
        if self._partial_translation_task is task:
            self._partial_translation_task = None

    def _final_translation_done(self, task: asyncio.Task[None]) -> None:
        self._final_translation_tasks.discard(task)

    async def _cancel_partial_translation(self) -> None:
        task = self._partial_translation_task
        if task is None:
            return
        self._partial_translation_task = None
        task.cancel()
        try:
            # Shield against outer cancellation, with timeout to avoid indefinite blocking
            await asyncio.wait_for(
                asyncio.shield(task),
                timeout=1.0,
            )
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass

    async def _cancel_translation_tasks(self, cancel_final: bool) -> None:
        await self._cancel_partial_translation()
        if not cancel_final:
            return
        tasks = list(self._final_translation_tasks)
        for task in tasks:
            task.cancel()
        for task in tasks:
            try:
                await asyncio.wait_for(
                    asyncio.shield(task),
                    timeout=2.0,
                )
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
        self._final_translation_tasks.clear()

    def _append_context(
        self,
        text: str,
        translation: str,
        detected_language: str | None,
        target_language: str | None = None,
    ) -> None:
        self._context.append(
            f"{detected_language or self.source_language}: {text}\n"
            f"{target_language or self.target_language}: {translation}"
        )

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._cancel_translation_tasks(cancel_final=True)

    async def _translate(self, text: str, detected_language: str | None = None) -> str:
        return await self.translator.translate(
            text=text,
            source_language=self._translation_source(detected_language),
            target_language=self.target_language,
            context=list(self._context),
        )

    def _translation_source(self, detected_language: str | None) -> str:
        if self.source_language == "auto":
            return detected_language or "auto"
        return self.source_language

    async def _reset_utterance(self) -> None:
        self._state = await self.asr.new_state(
            self._forced_asr_language(),
            self.asr_model,
        )
        self._decoder.reset()
        self._pending_audio = np.empty(0, dtype=np.float32)
        self._pre_roll_audio = np.empty(0, dtype=np.float32)
        self._has_speech = False
        self._speech_candidate_samples = 0
        self._speech_samples = 0
        self._silence_samples = 0
        self._last_transcript = ""
        self._last_translated_text = ""
        self._last_translation_at = 0.0
        self._language_detector.reset_utterance()

    def _forced_asr_language(self) -> str | None:
        return LANGUAGE_NAMES[self.source_language]

    def _resolve_language(self, snapshot: ASRSnapshot, final: bool) -> str | None:
        if self.source_language != "auto":
            return self.source_language
        return self._language_detector.observe(
            snapshot.text,
            snapshot.language,
            final=final,
        )

    def _transcript_event(
        self,
        snapshot: ASRSnapshot,
        detected_language: str | None,
        final: bool,
        rms: float | None = None,
    ) -> dict[str, Any]:
        return self._event(
            "transcript.final" if final else "transcript.partial",
            text=snapshot.text,
            language=detected_language,
            is_final=final,
            rms=round(rms, 5) if rms is not None else None,
        )

    def _translation_event(
        self,
        translation: str,
        source_text: str,
        final: bool,
        target_language: str,
    ) -> dict[str, Any]:
        return self._event(
            "translation.final" if final else "translation.partial",
            text=translation,
            source_text=source_text,
            language=target_language,
            is_final=final,
        )

    def _event(self, event_type: str, **payload: Any) -> dict[str, Any]:
        self._sequence += 1
        return {
            "type": event_type,
            "sequence": self._sequence,
            "timestamp": time.time(),
            **payload,
        }

    @staticmethod
    def _validate_source(language: str) -> str:
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Ngôn ngữ nguồn không hợp lệ: {language}")
        return language

    @staticmethod
    def _validate_target(language: str) -> str:
        if language not in SUPPORTED_LANGUAGES - {"auto"}:
            raise ValueError(f"Ngôn ngữ đích không hợp lệ: {language}")
        return language

    def _validate_asr_model(self, model_key: str) -> str:
        if model_key not in self.asr.available_models:
            supported = ", ".join(self.asr.available_models)
            raise ValueError(
                f"Model STT không hợp lệ: {model_key}. Hỗ trợ: {supported}"
            )
        return model_key
