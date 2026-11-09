import numpy as np

from app.audio import PCM16Decoder, root_mean_square


def test_pcm16_decoder_preserves_odd_byte_between_frames() -> None:
    decoder = PCM16Decoder()
    encoded = np.array([0, 16384, -16384], dtype="<i2").tobytes()

    first = decoder.decode(encoded[:3])
    second = decoder.decode(encoded[3:])

    assert np.allclose(first, [0.0])
    assert np.allclose(second, [0.5, -0.5])


def test_root_mean_square() -> None:
    samples = np.array([0.5, -0.5], dtype=np.float32)
    assert root_mean_square(samples) == 0.5
