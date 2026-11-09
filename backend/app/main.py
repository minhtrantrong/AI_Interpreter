from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.config import Settings, get_settings
from app.services.asr import ASRService, create_asr_service
from app.services.translator import Translator, create_translator
from app.session import RealtimeSession, TextInputSession

logger = logging.getLogger("translate-realtime")
MAX_SOCKET_FRAME_BYTES = 1_048_576


def create_app(
    settings: Settings | None = None,
    asr: ASRService | None = None,
    translator: Translator | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    asr = asr or create_asr_service(settings)
    translator = translator or create_translator(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = settings
        app.state.asr = asr
        app.state.translator = translator
        if settings.preload_models:
            await asr.load(settings.asr_default_model)
            await translator.load()
        yield
        await translator.close()

    app = FastAPI(
        title="Translate Realtime Local API",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.allowed_origins),
        allow_credentials=True,
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    @app.get("/")
    async def root() -> dict[str, Any]:
        return {
            "name": "translate-realtime-backend",
            "transport": "websocket",
            "websocket": "/ws/realtime?source=auto&target=en",
        }

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "fake_models": settings.use_fake_models,
            "asr_model": settings.asr_default_model,
            "asr_models": {
                key: {
                    "model": model,
                    "loaded": asr.is_model_loaded(key),
                }
                for key, model in asr.available_models.items()
            },
            "asr_loaded": asr.loaded,
            "translation_model": settings.translation_model_path,
            "translation_loaded": translator.loaded,
            "audio": {
                "encoding": "pcm_s16le",
                "sample_rate": settings.audio_sample_rate,
                "channels": 1,
            },
        }

    @app.websocket("/ws/realtime")
    async def realtime_socket(websocket: WebSocket) -> None:
        origin = websocket.headers.get("origin")
        if origin and settings.allowed_origins and origin not in settings.allowed_origins:
            await websocket.close(code=1008, reason="Origin không được phép")
            return

        source = websocket.query_params.get("source", "auto")
        target = websocket.query_params.get("target", "en")
        asr_model = websocket.query_params.get(
            "asr_model", settings.asr_default_model
        )
        await websocket.accept()
        send_lock = asyncio.Lock()
        session: RealtimeSession | None = None

        async def emit(event: dict[str, Any]) -> None:
            async with send_lock:
                await websocket.send_json(event)

        try:
            session = RealtimeSession(
                settings,
                asr,
                translator,
                source,
                target,
                asr_model=asr_model,
                event_sink=emit,
            )
            await emit(
                {
                    "type": "session.loading",
                    "asr_model": session.asr_model,
                    "asr_model_id": asr.available_models[session.asr_model],
                    "translation_model": settings.translation_model_path,
                }
            )
            await session.start()
            await emit(
                {
                    "type": "session.ready",
                    "source_language": session.source_language,
                    "target_language": session.target_language,
                    "asr_model": session.asr_model,
                    "audio": {
                        "encoding": "pcm_s16le",
                        "sample_rate": settings.audio_sample_rate,
                        "channels": 1,
                    },
                }
            )

            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    break
                # Only process one message type per frame to prevent mixed frames
                if message.get("bytes") is not None:
                    payload = message["bytes"]
                    if len(payload) > MAX_SOCKET_FRAME_BYTES:
                        await emit(
                            {"type": "error", "code": "frame_too_large"}
                        )
                        continue
                    await _send_events(emit, await session.feed_audio(payload))
                    continue
                if message.get("text") is not None:
                    events, should_close = await _handle_command(session, message["text"])
                    await _send_events(emit, events)
                    if should_close:
                        await websocket.close(code=1000)
                        break
        except WebSocketDisconnect:
            return
        except ValueError as exc:
            if session is not None:
                await session.close()
            await _safe_error(websocket, "invalid_request", str(exc), close_code=1008)
        except Exception as exc:  # keep model/runtime details out of client responses
            logger.exception("Realtime socket failed")
            if session is not None:
                await session.close()
            await _safe_error(
                websocket,
                "runtime_error",
                "Không thể xử lý realtime. Kiểm tra model local và log backend.",
                close_code=1011,
            )
        finally:
            if session is not None:
                await session.close()

    return app


async def _handle_command(
    session: RealtimeSession, raw_message: str
) -> tuple[list[dict[str, Any]], bool]:
    try:
        command = json.loads(raw_message)
    except json.JSONDecodeError as exc:
        raise ValueError("Lệnh socket phải là JSON hợp lệ") from exc

    command_type = command.get("type")
    if command_type == "commit":
        return await session.commit("client"), False
    if command_type == "reset":
        return await session.reset(bool(command.get("clear_context"))), False
    if command_type == "set_languages":
        return await session.set_languages(
            command.get("source", session.source_language),
            command.get("target", session.target_language),
        ), False
    if command_type == "set_asr_model":
        return await session.set_asr_model(
            command.get("model", session.asr_model)
        ), False
    if command_type == "ping":
        return [{"type": "pong"}], False
    if command_type == "end":
        return await session.commit("client_end", background_translation=False), True
    if command_type == "translate_text":
        # Text-only translation (no audio)
        text = command.get("text", "").strip()
        if not text:
            return [{"type": "error", "code": "empty_text", "message": "Text is empty"}], False
        return await session.translate_text(text), False
    raise ValueError(f"Lệnh socket không được hỗ trợ: {command_type}")


async def _send_events(
    event_sink: Any,
    events: list[dict[str, Any]],
) -> None:
    for event in events:
        await event_sink(event)


async def _safe_error(
    websocket: WebSocket,
    code: str,
    message: str,
    close_code: int,
) -> None:
    try:
        await websocket.send_json({"type": "error", "code": code, "message": message})
        await websocket.close(code=close_code)
    except RuntimeError:
        pass


app = create_app()
