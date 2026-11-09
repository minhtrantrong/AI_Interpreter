from __future__ import annotations

import argparse
import asyncio
import json

import websockets


async def run(url: str, text: str, source: str, target: str) -> None:
    uri = f"{url}?source={source}&target={target}&asr_model=small"
    async with websockets.connect(uri, max_size=2**20) as socket:
        while True:
            event = json.loads(await asyncio.wait_for(socket.recv(), timeout=60))
            print(event)
            if event["type"] == "session.ready":
                break

        await socket.send(json.dumps({"type": "translate_text", "text": text}))
        while True:
            event = json.loads(await asyncio.wait_for(socket.recv(), timeout=60))
            print(event)
            if event["type"] in {"translation.final", "translation.error", "error"}:
                break


def main() -> None:
    parser = argparse.ArgumentParser(description="Kiểm tra dịch text qua WebSocket")
    parser.add_argument(
        "--url",
        default="ws://127.0.0.1:8000/ws/realtime",
    )
    parser.add_argument("--text", default="Hello, how are you today?")
    parser.add_argument("--source", default="en", choices=("auto", "vi", "en", "ja", "ko"))
    parser.add_argument("--target", default="vi", choices=("vi", "en", "ja", "ko"))
    args = parser.parse_args()
    asyncio.run(run(args.url, args.text, args.source, args.target))


if __name__ == "__main__":
    main()
