from __future__ import annotations

import argparse
import asyncio
import json

import numpy as np
import websockets


async def run(url: str, seconds: float, noise_rms: float, model: str) -> None:
    uri = f"{url}?source=auto&target=vi&asr_model={model}"
    random = np.random.default_rng(42)
    unexpected: list[dict] = []

    async with websockets.connect(uri, max_size=2**20) as socket:
        print(json.loads(await socket.recv()))
        print(json.loads(await socket.recv()))
        for _ in range(round(seconds * 10)):
            samples = random.normal(0, noise_rms, 1_600)
            frame = (np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()
            await socket.send(frame)
            await asyncio.sleep(0.1)
        await socket.send(json.dumps({"type": "commit"}))

        while True:
            event = json.loads(await asyncio.wait_for(socket.recv(), timeout=10))
            print(event)
            if event["type"].startswith(("transcript.", "translation.")):
                unexpected.append(event)
            if event["type"] == "utterance.end":
                break

    if unexpected:
        raise RuntimeError(f"Im lặng tạo ra text ngoài ý muốn: {unexpected}")
    print(
        f"PASS: {seconds:g} giây audio RMS={noise_rms:g} "
        "không tạo transcript/bản dịch"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Kiểm tra chống hallucination khi im lặng")
    parser.add_argument(
        "--url",
        default="ws://127.0.0.1:8000/ws/realtime",
    )
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--model", choices=("small", "turbo"), default="small")
    parser.add_argument(
        "--noise-rms",
        type=float,
        default=0.0,
        help="Mức RMS của nhiễu trắng giả lập (0 là im lặng tuyệt đối)",
    )
    args = parser.parse_args()
    asyncio.run(run(args.url, args.seconds, args.noise_rms, args.model))


if __name__ == "__main__":
    main()
