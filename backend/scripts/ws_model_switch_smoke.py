from __future__ import annotations

import argparse
import asyncio
import json

import websockets


async def receive_until(socket: websockets.ClientConnection, event_type: str) -> dict:
    while True:
        event = json.loads(await asyncio.wait_for(socket.recv(), timeout=180))
        print(event)
        if event["type"] == "error":
            raise RuntimeError(event.get("message", "Backend trả về lỗi"))
        if event["type"] == event_type:
            return event


async def run(url: str) -> None:
    uri = f"{url}?source=vi&target=en&asr_model=small"
    async with websockets.connect(uri, max_size=2**20) as socket:
        await receive_until(socket, "session.ready")

        for model in ("turbo", "small"):
            await socket.send(
                json.dumps({"type": "set_asr_model", "model": model})
            )
            event = await receive_until(socket, "session.model")
            if event.get("asr_model") != model:
                raise RuntimeError(f"Backend không chuyển sang model {model}")

    print("PASS: chuyển Small -> Turbo -> Small trên cùng WebSocket")


def main() -> None:
    parser = argparse.ArgumentParser(description="Kiểm tra chuyển model STT realtime")
    parser.add_argument(
        "--url",
        default="ws://127.0.0.1:8000/ws/realtime",
    )
    args = parser.parse_args()
    asyncio.run(run(args.url))


if __name__ == "__main__":
    main()
