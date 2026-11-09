import asyncio
from dataclasses import replace

import numpy as np

from app.config import get_settings
from app.language_detection import StableLanguageDetector
from app.main import _handle_command
from app.services.asr import ASRSnapshot, FakeASRService
from app.services.translator import FakeTranslator
from app.session import RealtimeSession


def _session() -> RealtimeSession:
    settings = replace(
        get_settings(),
        use_fake_models=True,
        asr_step_ms=100,
        speech_start_ms=100,
        silence_finalize_ms=100,
        translation_interval_ms=0,
    )
    return RealtimeSession(
        settings,
        FakeASRService(),
        FakeTranslator(),
        source_language="auto",
        target_language="en",
    )


def test_session_streams_partial_and_final_results() -> None:
    async def scenario() -> None:
        session = _session()
        await session.start()
        speech = np.full(1600, 8000, dtype="<i2").tobytes()
        silence = np.zeros(1600, dtype="<i2").tobytes()

        early_events = await session.feed_audio(speech)
        assert [event["type"] for event in early_events] == ["transcript.partial"]

        partial_events = await session.feed_audio(speech)
        assert [event["type"] for event in partial_events] == [
            "transcript.partial",
            "translation.partial",
        ]

        final_events = await session.feed_audio(silence)
        event_types = {event["type"] for event in final_events}
        assert "transcript.final" in event_types
        assert "translation.final" in event_types
        assert "utterance.end" in event_types

    asyncio.run(scenario())


def test_language_socket_command() -> None:
    async def scenario() -> None:
        session = _session()
        await session.start()
        events, should_close = await _handle_command(
            session,
            '{"type":"set_languages","source":"vi","target":"ja"}',
        )
        assert should_close is False
        assert events[-1]["type"] == "session.languages"
        assert events[-1]["source_language"] == "vi"
        assert events[-1]["target_language"] == "ja"
        assert session._state.language == "vi"

    asyncio.run(scenario())


def test_asr_model_socket_command() -> None:
    async def scenario() -> None:
        session = _session()
        await session.start()

        events, should_close = await _handle_command(
            session,
            '{"type":"set_asr_model","model":"turbo"}',
        )

        assert should_close is False
        assert [event["type"] for event in events] == [
            "session.model_loading",
            "session.model",
        ]
        assert events[-1]["asr_model"] == "turbo"
        assert session.asr_model == "turbo"
        assert session._state.model_key == "turbo"

    asyncio.run(scenario())


def test_invalid_asr_model_socket_command() -> None:
    async def scenario() -> None:
        session = _session()
        await session.start()

        try:
            await _handle_command(
                session,
                '{"type":"set_asr_model","model":"unknown"}',
            )
        except ValueError as exc:
            assert "Model STT không hợp lệ" in str(exc)
        else:
            raise AssertionError("Model không hỗ trợ phải bị từ chối")

    asyncio.run(scenario())


def test_silence_never_reaches_asr_or_translation() -> None:
    async def scenario() -> None:
        session = _session()
        await session.start()
        silence = np.zeros(16_000, dtype="<i2").tobytes()

        assert await session.feed_audio(silence) == []
        assert session._state.calls == 0
        assert session._has_speech is False

        committed = await session.commit()
        assert [event["type"] for event in committed] == ["utterance.end"]

    asyncio.run(scenario())


def test_short_noise_spike_does_not_start_transcription() -> None:
    async def scenario() -> None:
        settings = replace(
            get_settings(),
            use_fake_models=True,
            asr_step_ms=100,
            speech_start_ms=200,
            silence_finalize_ms=100,
        )
        session = RealtimeSession(
            settings,
            FakeASRService(),
            FakeTranslator(),
            source_language="auto",
            target_language="en",
        )
        await session.start()
        spike = np.full(1600, 8000, dtype="<i2").tobytes()
        silence = np.zeros(1600, dtype="<i2").tobytes()

        assert await session.feed_audio(spike) == []
        assert await session.feed_audio(silence) == []
        assert session._state.calls == 0
        assert session._has_speech is False

    asyncio.run(scenario())


def test_rejected_snapshot_clears_uncommitted_partial_text() -> None:
    async def scenario() -> None:
        session = _session()
        await session.start()
        session._last_transcript = "Hãy subscribe cho kênh"
        session._last_translated_text = "Please subscribe to the channel"

        events = await session._partial_events(
            ASRSnapshot(language="vi", text=""),
            rms=0.0,
        )

        assert [event["type"] for event in events] == [
            "transcript.partial",
            "translation.partial",
        ]
        assert [event["text"] for event in events] == ["", ""]
        assert session._last_transcript == ""
        assert session._last_translated_text == ""

    asyncio.run(scenario())


def test_auto_language_requires_stable_model_evidence() -> None:
    detector = StableLanguageDetector()

    assert detector.observe("hello there", "English") == "en"
    assert detector.observe("hello there friend", "Japanese") == "en"
    assert detector.observe("hello there friend", "Korean") == "en"
    assert detector.observe("hello there my friend", "English") == "en"


def test_text_detector_corrects_wrong_asr_label_after_confirmation() -> None:
    detector = StableLanguageDetector()

    assert detector.observe("xin chao ban", "English") is None
    assert detector.observe("xin chao ban, hom nay ban khoe khong", "English") == "vi"


def test_distinctive_script_overrides_noisy_language_label() -> None:
    detector = StableLanguageDetector()

    assert detector.observe("hello there", "English") == "en"
    assert detector.observe("hello there again", "English") == "en"
    assert detector.observe("안녕하세요", "English") == "ko"
    assert detector.observe("こんにちは", "Korean") == "ja"
    assert detector.observe("Xin chào bạn", "English") == "vi"


def test_background_translation_does_not_block_audio_events() -> None:
    async def scenario() -> None:
        started = asyncio.Event()
        release = asyncio.Event()
        emitted: list[dict] = []

        class SlowTranslator(FakeTranslator):
            async def translate(self, *args, **kwargs) -> str:
                started.set()
                await release.wait()
                return "Hello"

        async def sink(event: dict) -> None:
            emitted.append(event)

        settings = replace(
            get_settings(),
            use_fake_models=True,
            asr_step_ms=100,
            speech_start_ms=100,
            translation_interval_ms=0,
            translation_debounce_ms=0,
        )
        session = RealtimeSession(
            settings,
            FakeASRService(),
            SlowTranslator(),
            source_language="auto",
            target_language="en",
            event_sink=sink,
        )
        await session.start()
        speech = np.full(1600, 8000, dtype="<i2").tobytes()

        early = await asyncio.wait_for(session.feed_audio(speech), timeout=0.1)
        assert [event["type"] for event in early] == ["transcript.partial"]

        immediate = await asyncio.wait_for(session.feed_audio(speech), timeout=0.1)
        assert [event["type"] for event in immediate] == ["transcript.partial"]
        await asyncio.wait_for(started.wait(), timeout=0.1)

        release.set()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert [event["type"] for event in emitted] == ["translation.partial"]
        await session.close()

    asyncio.run(scenario())
