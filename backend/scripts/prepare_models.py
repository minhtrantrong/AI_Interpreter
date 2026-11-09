from __future__ import annotations

import argparse
import os
from pathlib import Path

# NLLB chỉ phát hành trọng số PyTorch. Transformers mặc định tạo một thread nền
# để upload bản safetensors lên Hub, khiến script chuẩn bị model không thoát.
os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "true")

from ctranslate2.converters import TransformersConverter
from transformers import AutoTokenizer


SOURCE_MODEL = "facebook/nllb-200-distilled-600M"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tải và chuyển NLLB-200 sang CTranslate2 INT8/FP16",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/nllb-200-distilled-600m-ct2"),
    )
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    TransformersConverter(SOURCE_MODEL).convert(
        str(output),
        quantization="int8_float16",
        force=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        SOURCE_MODEL,
        src_lang="eng_Latn",
        use_fast=False,
    )
    tokenizer.save_pretrained(output)
    print(f"Model đã sẵn sàng tại {output}")


if __name__ == "__main__":
    main()
