from __future__ import annotations

import asyncio
import json

import numpy as np
import websockets


async def main() -> None:
    uri = "ws://127.0.0.1:8000/ws/realtime?source=auto&target=en"
    speech = np.full(8000, 8000, dtype="<i2").tobytes()
    silence = np.zeros(12800, dtype="<i2").tobytes()

    async with websockets.connect(uri, max_size=2**20) as socket:
        print(json.loads(await socket.recv()))
        print(json.loads(await socket.recv()))
        await socket.send(speech)
        await socket.send(silence)
        while True:
            event = json.loads(await socket.recv())
            print(event)
            if event["type"] == "utterance.end":
                break


if __name__ == "__main__":
    asyncio.run(main())
