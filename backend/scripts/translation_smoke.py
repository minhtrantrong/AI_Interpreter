from __future__ import annotations

import asyncio

from app.config import get_settings
from app.services.translator import NLLBTranslator


async def main() -> None:
    translator = NLLBTranslator(get_settings())
    samples = (
        ("Hello, how are you today?", "en", "vi"),
        ("Hôm nay thời tiết rất đẹp.", "vi", "en"),
        ("今日はいい天気ですね。", "ja", "vi"),
        ("오늘 만나서 반갑습니다.", "ko", "en"),
    )
    try:
        for text, source, target in samples:
            result = await translator.translate(text, source, target, [])
            print(f"{source}->{target}: {result}")
    finally:
        await translator.close()


if __name__ == "__main__":
    asyncio.run(main())
