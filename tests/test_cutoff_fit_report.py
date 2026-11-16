"""Offline report packing must preserve trace order and fit precision."""

import base64
import gzip

import numpy as np
import pytest

from scripts.report_cutoff_fits import pack_arrays


def test_browser_arrays_preserve_alignment_shape_and_nan():
    arrays = {
        "accepted": np.array([1, 0, 1], dtype="uint8"),
        "population": np.arange(30, dtype="<f4").reshape(2, 3, 5) / 31,
        "parameters": np.array([[.1234567890123456, np.nan, -2.34567890123456]],
                               dtype="<f8"),
    }
    packed = pack_arrays(arrays)
    raw = gzip.decompress(base64.b64decode(packed["payload"]))
    assert len(raw) == packed["byte_length"]
    for key, expected in arrays.items():
        spec = packed["layout"][key]
        assert spec["offset"] % 8 == 0
        actual = np.frombuffer(raw, dtype=spec["dtype"], count=spec["count"],
                               offset=spec["offset"]).reshape(spec["shape"])
        np.testing.assert_array_equal(actual, expected)
    assert pack_arrays(arrays) == packed


def test_browser_arrays_reject_unsupported_dtype():
    with pytest.raises(ValueError, match="Unsupported browser dtype"):
        pack_arrays({"values": np.array([1, 2], dtype="<i8")})
