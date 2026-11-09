from __future__ import annotations

import numpy as np


class PCM16Decoder:
    """Decode a byte stream of little-endian PCM16 samples without losing odd bytes."""

    def __init__(self) -> None:
        self._remainder = b""

    def decode(self, payload: bytes) -> np.ndarray:
        data = self._remainder + payload
        usable_length = len(data) - (len(data) % 2)
        self._remainder = data[usable_length:]
        if usable_length == 0:
            return np.empty(0, dtype=np.float32)
        pcm = np.frombuffer(data[:usable_length], dtype="<i2")
        return pcm.astype(np.float32) / 32768.0

    def reset(self) -> None:
        self._remainder = b""


def root_mean_square(samples: np.ndarray) -> float:
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float32))))
