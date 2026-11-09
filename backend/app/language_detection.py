from __future__ import annotations

import re
from functools import lru_cache

from lingua import Language, LanguageDetectorBuilder


LANGUAGE_CODE_BY_NAME = {
    "vi": "vi",
    "en": "en",
    "ja": "ja",
    "ko": "ko",
    "vietnamese": "vi",
    "english": "en",
    "japanese": "ja",
    "korean": "ko",
}

_LANGUAGE_SEPARATOR = re.compile(r"[,/+]")
_LINGUA_CODE = {
    Language.VIETNAMESE: "vi",
    Language.ENGLISH: "en",
    Language.JAPANESE: "ja",
    Language.KOREAN: "ko",
}
_VIETNAMESE_MARKS = frozenset(
    "ăâđêôơưĂÂĐÊÔƠƯ"
    "áàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệ"
    "íìỉĩịóòỏõọốồổỗộớờởỡợ"
    "úùủũụứừửữựýỳỷỹỵ"
    "ÁÀẢÃẠẤẦẨẪẬẮẰẲẴẶÉÈẺẼẸẾỀỂỄỆ"
    "ÍÌỈĨỊÓÒỎÕỌỐỒỔỖỘỚỜỞỠỢ"
    "ÚÙỦŨỤỨỪỬỮỰÝỲỶỸỴ"
)


def normalize_model_language(language: str | None) -> str | None:
    """Return one supported language code, ignoring ambiguous/unsupported labels."""
    if not language:
        return None
    codes = {
        LANGUAGE_CODE_BY_NAME[token.strip().lower()]
        for token in _LANGUAGE_SEPARATOR.split(language)
        if token.strip().lower() in LANGUAGE_CODE_BY_NAME
    }
    return next(iter(codes)) if len(codes) == 1 else None


def language_from_script(text: str) -> str | None:
    """Detect scripts that uniquely identify one of the four UI languages."""
    if any("\uac00" <= char <= "\ud7af" for char in text):
        return "ko"
    if any(
        "\u3040" <= char <= "\u30ff" or "\u31f0" <= char <= "\u31ff"
        for char in text
    ):
        return "ja"
    if any(char in _VIETNAMESE_MARKS for char in text):
        return "vi"
    return None


@lru_cache(maxsize=1)
def _text_detector():
    return LanguageDetectorBuilder.from_languages(*_LINGUA_CODE).build()


def language_from_text(text: str) -> tuple[str | None, float]:
    """Classify transcript text only when Lingua finds a clear winner."""
    if sum(char.isalpha() for char in text) < 6:
        return None, 0.0
    confidence_values = _text_detector().compute_language_confidence_values(text)
    if not confidence_values:
        return None, 0.0

    top = confidence_values[0]
    runner_up = confidence_values[1].value if len(confidence_values) > 1 else 0.0
    margin = top.value - runner_up
    if top.value < 0.65 or margin < 0.20:
        return None, top.value
    return _LINGUA_CODE[top.language], top.value


class StableLanguageDetector:
    """Add hysteresis to the per-chunk language label returned by streaming ASR."""

    def __init__(self, min_letters: int = 6, confirmations: int = 2) -> None:
        self.min_letters = min_letters
        self.confirmations = confirmations
        self.language: str | None = None
        self._candidate: str | None = None
        self._candidate_count = 0

    def observe(
        self,
        text: str,
        model_language: str | None,
        *,
        final: bool = False,
    ) -> str | None:
        script_language = language_from_script(text)
        if script_language:
            self.language = script_language
            self._clear_candidate()
            return self.language

        model_candidate = normalize_model_language(model_language)
        text_candidate, _ = language_from_text(text)
        corroborated = bool(
            model_candidate and text_candidate and model_candidate == text_candidate
        )
        candidate = text_candidate or model_candidate
        letter_count = sum(char.isalpha() for char in text)
        if candidate is None or (letter_count < self.min_letters and not final):
            return self.language

        if candidate == self.language:
            self._clear_candidate()
            return self.language

        if candidate == self._candidate:
            self._candidate_count += 1
        else:
            self._candidate = candidate
            self._candidate_count = 1

        required = self.confirmations
        if self.language is None and (corroborated or final):
            required = 1
        if self._candidate_count >= required:
            self.language = candidate
            self._clear_candidate()
        return self.language

    def reset_utterance(self) -> None:
        """Keep the session language, but discard weak evidence from the old sentence."""
        self._clear_candidate()

    def clear(self) -> None:
        self.language = None
        self._clear_candidate()

    def _clear_candidate(self) -> None:
        self._candidate = None
        self._candidate_count = 0
