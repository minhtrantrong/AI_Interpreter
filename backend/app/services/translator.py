from __future__ import annotations

import asyncio
import gc
import re
import unicodedata
from pathlib import Path
from typing import Any, Protocol

from app.config import Settings
from app.language_detection import language_from_script, language_from_text


SUPPORTED_LANGUAGES = frozenset({"vi", "en", "ja", "ko"})
NLLB_LANGUAGE_CODES = {
    "vi": "vie_Latn",
    "en": "eng_Latn",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
}


class Translator(Protocol):
    loaded: bool

    async def load(self) -> None: ...
    async def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
        context: list[str],
    ) -> str: ...

    async def unload(self) -> None: ...
    async def close(self) -> None: ...


class NLLBTranslator:
    # Maximum context turns to include in translation prefix
    MAX_CONTEXT_TURNS = 3

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.loaded = False
        self._translator: Any = None
        self._tokenizer: Any = None
        self._load_lock = asyncio.Lock()
        self._inference_lock = asyncio.Lock()

    async def load(self) -> None:
        if self.loaded:
            return
        async with self._load_lock:
            if self.loaded:
                return
            await asyncio.to_thread(self._load_sync)
            self.loaded = True

    def _load_sync(self) -> None:
        try:
            import ctranslate2
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Thiếu CTranslate2/Transformers. Hãy cài backend bằng: "
                "pip install -e '.[models]'"
            ) from exc

        model_path = Path(self.settings.translation_model_path).expanduser().resolve()
        if not (model_path / "model.bin").exists():
            raise RuntimeError(
                f"Chưa có model dịch tại {model_path}. "
                "Hãy chạy: python scripts/prepare_models.py"
            )
        self._translator = ctranslate2.Translator(
            str(model_path),
            device=self.settings.translation_device,
            compute_type=self.settings.translation_compute_type,
            inter_threads=1,
            intra_threads=self.settings.translation_intra_threads,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            src_lang=NLLB_LANGUAGE_CODES["en"],
            use_fast=False,
            local_files_only=True,
        )

    async def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
        context: list[str],
    ) -> str:
        await self.load()
        source = self._resolve_source(text, source_language)
        if source == target_language:
            return text.strip()

        # Build context from recent conversation turns
        # Context is prepended as a soft prefix to improve coherence
        context_text = self._build_context_prefix(context)

        async with self._inference_lock:
            return await asyncio.to_thread(
                self._translate_sync,
                text,
                source,
                target_language,
                context_text,
            )

    def _translate_sync(
        self,
        text: str,
        source: str,
        target: str,
        context_text: str = "",
    ) -> str:
        source_code = NLLB_LANGUAGE_CODES[source]
        target_code = NLLB_LANGUAGE_CODES[target]
        self._tokenizer.src_lang = source_code

        # Combine context with current text for better coherence
        full_text = f"{context_text} {text}".strip() if context_text else text
        chunks = _translation_chunks(full_text, max_chars=220)

        source_tokens = [
            self._tokenizer.convert_ids_to_tokens(self._tokenizer.encode(chunk))
            for chunk in chunks
        ]
        target_prefix = [target_code]
        results = self._translator.translate_batch(
            source_tokens,
            target_prefix=[target_prefix for _ in source_tokens],
            beam_size=self.settings.translation_beam_size,
            max_decoding_length=self.settings.translation_max_tokens,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
        )
        translations: list[str] = []
        for result in results:
            output_tokens = result.hypotheses[0]
            if output_tokens and output_tokens[0] == target_prefix[0]:
                output_tokens = output_tokens[1:]
            output_ids = self._tokenizer.convert_tokens_to_ids(output_tokens)
            translated = self._tokenizer.decode(
                output_ids,
                skip_special_tokens=True,
            ).strip()
            if translated:
                translations.append(translated)
        separator = "" if target == "ja" else " "
        return separator.join(translations).strip()

    def _build_context_prefix(self, context: list[str]) -> str:
        """Build a context prefix from recent conversation turns for better coherence."""
        if not context:
            return ""
        # Take the most recent turns, limited to MAX_CONTEXT_TURNS
        # Filter out None values for safety
        valid_context = [c for c in context if c]
        if not valid_context:
            return ""
        recent_context = valid_context[-self.MAX_CONTEXT_TURNS :]
        return " ".join(recent_context)

    @staticmethod
    def _resolve_source(text: str, source_language: str) -> str:
        if source_language in SUPPORTED_LANGUAGES:
            return source_language
        detected = language_from_script(text) or language_from_text(text)[0]
        return detected or "en"

    async def unload(self) -> None:
        self._translator = None
        self._tokenizer = None
        self.loaded = False
        gc.collect()

    async def close(self) -> None:
        await self.unload()


class FakeTranslator:
    loaded = True
    translations = {
        "vi": "Xin chào, rất vui được gặp bạn.",
        "en": "Hello, it is very nice to meet you.",
        "ja": "こんにちは、お会いできてとてもうれしいです。",
        "ko": "안녕하세요, 만나서 정말 반갑습니다.",
    }

    async def load(self) -> None:
        return None

    async def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
        context: list[str],
    ) -> str:
        return self.translations.get(target_language, f"[{target_language}] {text}")

    async def close(self) -> None:
        return None

    async def unload(self) -> None:
        return None


def create_translator(settings: Settings) -> Translator:
    if settings.use_fake_models:
        return FakeTranslator()
    return NLLBTranslator(settings)


_SENTENCE_PARTS = re.compile(r"[^.!?。！？]+[.!?。！？]*", re.UNICODE)
_NON_WORD = re.compile(r"[^\w]+", re.UNICODE)
_ACOUSTIC_FILLERS = frozenset({"mhm", "mmhmm", "hmm", "uh", "um"})


def _translation_chunks(text: str, max_chars: int = 220) -> list[str]:
    chunks: list[str] = []
    for match in _SENTENCE_PARTS.finditer(text.strip()):
        sentence = match.group(0).strip()
        if not sentence or _is_acoustic_filler(sentence):
            continue
        if len(sentence) <= max_chars:
            chunks.append(sentence)
            continue

        words = sentence.split()
        if len(words) == 1:
            chunks.extend(
                sentence[index : index + max_chars]
                for index in range(0, len(sentence), max_chars)
            )
            continue

        current: list[str] = []
        current_length = 0
        for word in words:
            added_length = len(word) + (1 if current else 0)
            if current and current_length + added_length > max_chars:
                chunks.append(" ".join(current))
                current = []
                current_length = 0
            current.append(word)
            current_length += len(word) + (1 if len(current) > 1 else 0)
        if current:
            chunks.append(" ".join(current))
    return chunks or [text.strip()]


def _is_acoustic_filler(text: str) -> bool:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return _NON_WORD.sub("", normalized) in _ACOUSTIC_FILLERS


def translation_is_ready(text: str) -> bool:
    """Avoid sending fragments that are too short for useful machine translation."""
    meaningful = " ".join(_translation_chunks(text))
    letter_count = sum(char.isalpha() for char in meaningful)
    has_compact_script = any(
        "\u3040" <= char <= "\u30ff"
        or "\u31f0" <= char <= "\u31ff"
        or "\u4e00" <= char <= "\u9fff"
        or "\uac00" <= char <= "\ud7af"
        for char in meaningful
    )
    if has_compact_script:
        return letter_count >= 4
    return letter_count >= 12 and len(meaningful.split()) >= 3
