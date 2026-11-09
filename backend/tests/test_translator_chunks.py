from types import SimpleNamespace

from app.config import get_settings
from app.services.translator import (
    NLLB_LANGUAGE_CODES,
    NLLBTranslator,
    _translation_chunks,
    translation_is_ready,
)


def test_translation_chunks_drop_non_semantic_filler_and_split_sentences() -> None:
    assert _translation_chunks("Mm-hmm. Hello there. How are you?") == [
        "Hello there.",
        "How are you?",
    ]


def test_translation_chunks_bound_long_text() -> None:
    chunks = _translation_chunks("word " * 100, max_chars=40)

    assert len(chunks) > 1
    assert all(len(chunk) <= 40 for chunk in chunks)


def test_partial_translation_waits_for_meaningful_content() -> None:
    assert translation_is_ready("Mm-hmm. Oh yeah") is False
    assert translation_is_ready("He was not very famous") is True
    assert translation_is_ready("今日は晴れ") is True


def test_nllb_uses_expected_language_codes_and_target_prefix() -> None:
    class Tokenizer:
        src_lang = ""

        @staticmethod
        def encode(text: str) -> list[int]:
            return [len(text)]

        @staticmethod
        def convert_ids_to_tokens(token_ids: list[int]) -> list[str]:
            return [f"token-{token_ids[0]}"]

        @staticmethod
        def convert_tokens_to_ids(tokens: list[str]) -> list[int]:
            return [len(tokens)]

        @staticmethod
        def decode(token_ids: list[int], skip_special_tokens: bool) -> str:
            assert token_ids == [1]
            assert skip_special_tokens is True
            return "Xin chào"

    class Runtime:
        target_prefix: list[list[str]] | None = None
        received_source: list = []

        def translate_batch(self, source_tokens, *, target_prefix, **options):
            self.received_source = source_tokens
            assert options["beam_size"] == get_settings().translation_beam_size
            self.target_prefix = target_prefix
            return [SimpleNamespace(hypotheses=[["vie_Latn", "translated"]])]

    translator = NLLBTranslator(get_settings())
    translator._tokenizer = Tokenizer()
    translator._translator = Runtime()

    # Test without context
    result = translator._translate_sync("Hello there!", "en", "vi")
    assert result == "Xin chào"
    assert translator._tokenizer.src_lang == NLLB_LANGUAGE_CODES["en"]
    assert translator._translator.target_prefix == [[NLLB_LANGUAGE_CODES["vi"]]]


def test_nllb_uses_context_for_better_coherence() -> None:
    """Context should be prepended to improve translation coherence."""
    encoded_texts: list[str] = []

    class Tokenizer:
        src_lang = ""

        @staticmethod
        def encode(text: str) -> list[int]:
            # Record the actual text being encoded
            encoded_texts.append(text)
            # Return tokens based on text length
            return [i for i in range(len(text))]

        @staticmethod
        def convert_ids_to_tokens(token_ids: list[int]) -> list[str]:
            return [f"t{i}" for i in token_ids]

        @staticmethod
        def convert_tokens_to_ids(tokens: list[str]) -> list[int]:
            # Handle both "t{N}" format from our mock and regular tokens
            result = []
            for t in tokens:
                if t.startswith("t") and t[1:].isdigit():
                    result.append(int(t[1:]))
                else:
                    result.append(hash(t) % 1000)
            return result

        @staticmethod
        def decode(token_ids: list[int], skip_special_tokens: bool) -> str:
            return "Translated with context"

    class Runtime:
        target_prefix: list[list[str]] | None = None

        def translate_batch(self, source_tokens, *, target_prefix, **options):
            self.target_prefix = target_prefix
            return [SimpleNamespace(hypotheses=[["vie_Latn", "translated"]])]

    translator = NLLBTranslator(get_settings())
    translator._tokenizer = Tokenizer()
    translator._translator = Runtime()

    # Test with context - full text should be longer than just the new text
    context_text = "Hello there my friend How are you today"
    result = translator._translate_sync(
        "I'm fine thank you",
        "en",
        "vi",
        context_text=context_text,
    )

    # Verify the full text (context + original) was encoded
    assert len(encoded_texts) > 0
    # The first encoded text should contain the context
    assert context_text in encoded_texts[0], f"Context should be in: {encoded_texts[0]}"
    assert "I'm fine thank you" in encoded_texts[0]


def test_context_prefix_builder_respects_max_turns() -> None:
    translator = NLLBTranslator(get_settings())

    # More than MAX_CONTEXT_TURNS should be truncated
    long_context = [f"turn {i}" for i in range(10)]
    prefix = translator._build_context_prefix(long_context)

    # Should only include last 3 turns
    assert prefix.count("turn") == 3
    assert "turn 9" in prefix
    assert "turn 7" in prefix


def test_context_prefix_builder_handles_empty_context() -> None:
    translator = NLLBTranslator(get_settings())

    assert translator._build_context_prefix([]) == ""
