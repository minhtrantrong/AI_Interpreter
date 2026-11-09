from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _as_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _as_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


@dataclass(frozen=True, slots=True)
class Settings:
    host: str
    port: int
    allowed_origins: tuple[str, ...]
    use_fake_models: bool
    preload_models: bool

    asr_default_model: str
    asr_model: str
    asr_turbo_model: str
    asr_device: str
    asr_compute_type: str
    asr_cpu_threads: int
    asr_num_workers: int
    asr_beam_size: int
    asr_final_beam_size: int
    asr_min_audio_sec: float
    asr_step_ms: int
    asr_max_window_sec: float
    asr_stability_margin_sec: float
    asr_max_no_speech_prob: float
    asr_min_avg_logprob: float
    asr_vad_threshold: float
    asr_vad_min_speech_ms: int
    asr_suppress_promotional_hallucinations: bool

    audio_sample_rate: int
    silence_finalize_ms: int
    speech_rms_threshold: float
    speech_start_ms: int
    speech_preroll_ms: int
    noise_floor_multiplier: float
    max_utterance_sec: int

    translation_model_path: str
    translation_device: str
    translation_compute_type: str
    translation_intra_threads: int
    translation_beam_size: int
    translation_interval_ms: int
    translation_debounce_ms: int
    translation_max_tokens: int
    translation_context_turns: int

    @property
    def asr_step_samples(self) -> int:
        return max(1, round(self.audio_sample_rate * self.asr_step_ms / 1000))

    @property
    def asr_models(self) -> dict[str, str]:
        return {
            "small": self.asr_model,
            "turbo": self.asr_turbo_model,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    origins = tuple(
        origin.strip()
        for origin in os.getenv(
            "ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
        ).split(",")
        if origin.strip()
    )
    return Settings(
        host=os.getenv("HOST", "0.0.0.0"),
        port=_as_int("PORT", 8000),
        allowed_origins=origins,
        use_fake_models=_as_bool(os.getenv("USE_FAKE_MODELS")),
        preload_models=_as_bool(os.getenv("PRELOAD_MODELS")),
        asr_default_model=os.getenv("ASR_DEFAULT_MODEL", "small"),
        asr_model=os.getenv("ASR_MODEL", "Systran/faster-whisper-small"),
        asr_turbo_model=os.getenv("ASR_TURBO_MODEL", "large-v3-turbo"),
        asr_device=os.getenv("ASR_DEVICE", "cuda"),
        asr_compute_type=os.getenv("ASR_COMPUTE_TYPE", "int8_float16"),
        asr_cpu_threads=_as_int("ASR_CPU_THREADS", 4),
        asr_num_workers=_as_int("ASR_NUM_WORKERS", 1),
        asr_beam_size=_as_int("ASR_BEAM_SIZE", 1),
        asr_final_beam_size=_as_int("ASR_FINAL_BEAM_SIZE", 5),
        asr_min_audio_sec=_as_float("ASR_MIN_AUDIO_SEC", 0.6),
        asr_step_ms=_as_int("ASR_STEP_MS", 600),
        asr_max_window_sec=_as_float("ASR_MAX_WINDOW_SEC", 12.0),
        asr_stability_margin_sec=_as_float("ASR_STABILITY_MARGIN_SEC", 2.5),
        asr_max_no_speech_prob=_as_float("ASR_MAX_NO_SPEECH_PROB", 0.65),
        asr_min_avg_logprob=_as_float("ASR_MIN_AVG_LOGPROB", -1.0),
        asr_vad_threshold=_as_float("ASR_VAD_THRESHOLD", 0.6),
        asr_vad_min_speech_ms=_as_int("ASR_VAD_MIN_SPEECH_MS", 250),
        asr_suppress_promotional_hallucinations=_as_bool(
            os.getenv("ASR_SUPPRESS_PROMOTIONAL_HALLUCINATIONS"),
            default=True,
        ),
        audio_sample_rate=_as_int("AUDIO_SAMPLE_RATE", 16000),
        silence_finalize_ms=_as_int("SILENCE_FINALIZE_MS", 900),
        speech_rms_threshold=_as_float("SPEECH_RMS_THRESHOLD", 0.01),
        speech_start_ms=_as_int("SPEECH_START_MS", 250),
        speech_preroll_ms=_as_int("SPEECH_PREROLL_MS", 350),
        noise_floor_multiplier=_as_float("NOISE_FLOOR_MULTIPLIER", 2.5),
        max_utterance_sec=_as_int("MAX_UTTERANCE_SEC", 45),
        translation_model_path=os.getenv(
            "TRANSLATION_MODEL_PATH", "models/nllb-200-distilled-600m-ct2"
        ),
        translation_device=os.getenv("TRANSLATION_DEVICE", "cuda"),
        translation_compute_type=os.getenv(
            "TRANSLATION_COMPUTE_TYPE", "int8_float16"
        ),
        translation_intra_threads=_as_int("TRANSLATION_INTRA_THREADS", 4),
        translation_beam_size=_as_int("TRANSLATION_BEAM_SIZE", 3),
        translation_interval_ms=_as_int("TRANSLATION_INTERVAL_MS", 900),
        translation_debounce_ms=_as_int("TRANSLATION_DEBOUNCE_MS", 120),
        translation_max_tokens=_as_int("TRANSLATION_MAX_TOKENS", 256),
        translation_context_turns=_as_int("TRANSLATION_CONTEXT_TURNS", 3),
    )
