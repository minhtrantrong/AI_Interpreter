from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import websockets


def load_pcm16(path: Path) -> np.ndarray:
    samples, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    if sample_rate != 16_000:
        duration = samples.size / float(sample_rate)
        target_size = round(duration * 16_000)
        old_axis = np.linspace(0, duration, samples.size, endpoint=False)
        new_axis = np.linspace(0, duration, target_size, endpoint=False)
        samples = np.interp(new_axis, old_axis, samples)
    return (np.clip(samples, -1, 1) * 32767).astype("<i2")


async def run(
    path: Path,
    source: str,
    target: str,
    model: str,
    realtime: bool,
    url: str,
) -> None:
    uri = f"{url}?source={source}&target={target}&asr_model={model}"
    pcm = load_pcm16(path)
    frame_samples = 1_600

    async with websockets.connect(uri, max_size=2**20) as socket:
        print(json.loads(await socket.recv()))
        print(json.loads(await socket.recv()))
        for offset in range(0, pcm.size, frame_samples):
            await socket.send(pcm[offset : offset + frame_samples].tobytes())
            if realtime:
                await asyncio.sleep(frame_samples / 16_000)
        # Longer than the default 1.4 s VAD finalize window.
        await socket.send(np.zeros(32_000, dtype="<i2").tobytes())

        while True:
            event = json.loads(await asyncio.wait_for(socket.recv(), timeout=120))
            print(event)
            if event["type"] == "utterance.end":
                break


def main() -> None:
    parser = argparse.ArgumentParser(description="Test model thật qua WebSocket")
    parser.add_argument("audio", type=Path)
    parser.add_argument("--source", default="auto")
    parser.add_argument("--target", default="en")
    parser.add_argument("--model", choices=("small", "turbo"), default="small")
    parser.add_argument(
        "--url",
        default="ws://127.0.0.1:8000/ws/realtime",
        help="WebSocket endpoint cần kiểm tra",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Gửi từng frame theo đúng tốc độ phát của microphone",
    )
    args = parser.parse_args()
    asyncio.run(
        run(
            args.audio,
            args.source,
            args.target,
            args.model,
            args.realtime,
            args.url,
        )
    )


if __name__ == "__main__":
    main()
