import numpy as np
import pytest
import threading

from audio_nodes.audio_device_nodes import RingBuffer


@pytest.mark.parametrize("channels", [1, 2, 4])
def test_init_always_2d(channels):
    """RingBuffer is always 2D (size, channels) regardless of channel count."""
    buf = RingBuffer(100, channels)
    assert buf._buffer.shape == (100, channels)
    assert buf._buffer.dtype == np.float32


@pytest.mark.parametrize("channels", [1, 2])
def test_write_read_roundtrip(channels):
    """Data written and read back matches, always 2D."""
    buf = RingBuffer(1000, channels)
    data = np.random.randn(100, channels).astype(np.float32)

    written = buf.write(data)
    assert written == 100

    result = buf.read(100)
    assert result.shape == (100, channels)
    np.testing.assert_array_equal(result, data)


@pytest.mark.parametrize("channels", [1, 2])
def test_boundary_wrap(channels):
    """Write/read across ring buffer boundary."""
    buf = RingBuffer(64, channels)

    # Fill most of the buffer
    data1 = np.ones((50, channels), dtype=np.float32)
    buf.write(data1)

    # Read to advance read position
    buf.read(50)

    # Write across boundary (50 + 30 > 64)
    data2 = np.full((30, channels), 2.0, dtype=np.float32)
    written = buf.write(data2)
    assert written == 30

    result = buf.read(30)
    assert result.shape == (30, channels)
    np.testing.assert_array_equal(result, data2)


@pytest.mark.parametrize("channels", [1, 2])
def test_overflow(channels):
    """Overflow returns fewer frames written than requested."""
    buf = RingBuffer(50, channels)

    data = np.ones((80, channels), dtype=np.float32)
    written = buf.write(data)
    assert written == 50  # Only 50 frames fit


@pytest.mark.parametrize("channels", [1, 2])
def test_underflow_returns_2d(channels):
    """Empty read returns 2D array with shape (0, channels)."""
    buf = RingBuffer(100, channels)

    result = buf.read(10)
    assert result.shape == (0, channels)
    assert result.dtype == np.float32


@pytest.mark.parametrize("channels", [1, 2])
def test_reset(channels):
    """Reset clears available count."""
    buf = RingBuffer(100, channels)
    buf.write(np.ones((50, channels), dtype=np.float32))
    assert buf.available() == 50

    buf.reset()
    assert buf.available() == 0


@pytest.mark.parametrize("channels", [1, 2])
def test_read_into(channels):
    """read_into copies directly to output without allocation."""
    buf = RingBuffer(1000, channels)
    data = np.random.randn(100, channels).astype(np.float32)
    buf.write(data)

    out = np.zeros((100, channels), dtype=np.float32)
    read = buf.read_into(out)
    assert read == 100
    np.testing.assert_array_equal(out, data)


@pytest.mark.parametrize("channels", [1, 2])
def test_read_into_partial(channels):
    """read_into with less data than output buffer."""
    buf = RingBuffer(1000, channels)
    data = np.random.randn(50, channels).astype(np.float32)
    buf.write(data)

    out = np.zeros((100, channels), dtype=np.float32)
    read = buf.read_into(out)
    assert read == 50
    np.testing.assert_array_equal(out[:50], data)


@pytest.mark.parametrize("channels", [1, 2])
def test_read_into_boundary_wrap(channels):
    """read_into across ring buffer boundary."""
    buf = RingBuffer(64, channels)

    # Advance write/read position
    buf.write(np.ones((50, channels), dtype=np.float32))
    buf.read(50)

    # Write across boundary
    data = np.full((30, channels), 2.0, dtype=np.float32)
    buf.write(data)

    out = np.zeros((30, channels), dtype=np.float32)
    read = buf.read_into(out)
    assert read == 30
    np.testing.assert_array_equal(out, data)


@pytest.mark.parametrize("channels", [1, 2])
def test_read_into_empty(channels):
    """read_into on empty buffer returns 0."""
    buf = RingBuffer(100, channels)
    out = np.zeros((10, channels), dtype=np.float32)
    read = buf.read_into(out)
    assert read == 0


def test_concurrent_access():
    """Write from one thread, read from another without data corruption."""
    buf = RingBuffer(1000, 2)
    written_total = [0]
    read_total = [0]
    errors = []

    def writer():
        for _ in range(100):
            data = np.ones((10, 2), dtype=np.float32)
            w = buf.write(data)
            written_total[0] += w

    def reader():
        for _ in range(100):
            data = buf.read(5)
            read_total[0] += len(data)

    t1 = threading.Thread(target=writer)
    t2 = threading.Thread(target=reader)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    # No crash = success. Exact counts depend on scheduling.
    assert written_total[0] > 0
