from __future__ import annotations

import argparse

from faster_whisper.utils import download_model


MODELS = {
    "small": "Systran/faster-whisper-small",
    "turbo": "large-v3-turbo",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tải trước model Faster Whisper vào cache local"
    )
    parser.add_argument(
        "--model",
        choices=(*MODELS, "all"),
        default="all",
        help="Model cần chuẩn bị",
    )
    args = parser.parse_args()
    selected = MODELS if args.model == "all" else {args.model: MODELS[args.model]}

    for key, model in selected.items():
        path = download_model(model)
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
