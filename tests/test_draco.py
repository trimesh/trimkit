import time

import numpy as np
import pytest
from helpers import LITTLE_ENDIAN, runs_here

import trimkit

pytestmark = pytest.mark.skipif(
    not LITTLE_ENDIAN,
    reason="draco refuses big-endian machines, as test_draco_refuses_only_big_endian asserts",
)


def primitive(n: int = 1000, seed: int = 0):
    """
    A glTF primitive as its parts: faces, every attribute array, each
    attribute's DracoKind and the glTF name it would carry.
    """
    rng = np.random.default_rng(seed)
    normal = rng.normal(size=(n, 3)).astype(np.float32)
    named = [
        (
            "POSITION",
            trimkit.exchange.DracoKind.POSITION,
            (rng.random((n, 3)) * [10, 1, 0.1]).astype(np.float32),
        ),
        ("NORMAL", trimkit.exchange.DracoKind.NORMAL, normal / np.linalg.norm(normal, axis=1)[:, None]),
        ("TEXCOORD_0", trimkit.exchange.DracoKind.TEXCOORD, rng.random((n, 2)).astype(np.float32)),
        ("COLOR_0", trimkit.exchange.DracoKind.COLOR, rng.integers(0, 256, (n, 4)).astype(np.uint8)),
        ("_SIGNED", trimkit.exchange.DracoKind.GENERIC, rng.integers(-1000, 1000, n).astype(np.int16)),
        ("_BYTE", trimkit.exchange.DracoKind.GENERIC, rng.integers(-128, 128, (n, 2)).astype(np.int8)),
    ]
    faces = rng.integers(0, n, (2 * n, 3))
    names = [n for n, _, _ in named]
    kinds = np.array([k for _, k, _ in named], dtype=np.int64)
    attributes = [a for _, _, a in named]
    return faces, attributes, kinds, names


def test_round_trip():
    faces, attributes, kinds, names = primitive()
    buffer = trimkit.exchange.draco_encode(faces, attributes, kinds)
    assert isinstance(buffer, bytes)
    # draco attribute i is attributes[i]
    decoded_faces, decoded = trimkit.exchange.draco_decode(buffer, np.arange(len(attributes), dtype=np.int64))
    # order and count are kept exactly, so accessors written before encoding
    # stay valid and the decoded vertices line up with the source
    assert np.array_equal(decoded_faces, faces)
    assert isinstance(decoded, tuple) and len(decoded) == len(attributes)
    for name, value, out in zip(names, attributes, decoded, strict=True):
        assert out.dtype == value.dtype
        assert out.shape == value.shape
        if name == "NORMAL":
            # normals are octahedral encoded: bound the angle, not the axes
            dot = np.einsum("ij,ij->i", out, value)
            assert np.degrees(np.arccos(np.clip(dot, -1, 1))).max() < 0.5
        elif value.dtype.kind == "f":
            # quantized onto 2**bits steps across the attribute's extent
            bits = {"POSITION": 14, "TEXCOORD_0": 12}[name]
            step = np.ptp(value) / (2**bits - 1)
            assert np.abs(out - value).max() <= 0.5 * step + 1e-7
        else:
            assert np.array_equal(out, value)


def test_decode_subset_in_any_order_and_views():
    faces, attributes, kinds, _ = primitive(100)
    buffer = trimkit.exchange.draco_encode(faces, attributes, kinds)
    # a subset, in any order, from bytes, a memoryview or a uint8 array
    wanted = np.array([4, 0, 4], dtype=np.int64)
    for source in (buffer, memoryview(buffer), np.frombuffer(buffer, dtype=np.uint8)):
        decoded_faces, decoded = trimkit.exchange.draco_decode(source, wanted)
        assert np.array_equal(decoded_faces, faces) and len(decoded) == 3
        assert np.array_equal(decoded[0], attributes[4]) and np.array_equal(decoded[2], attributes[4])
        assert decoded[1].shape == attributes[0].shape
    # nothing wanted still decodes the faces
    decoded_faces, decoded = trimkit.exchange.draco_decode(buffer, np.zeros(0, dtype=np.int64))
    assert np.array_equal(decoded_faces, faces) and decoded == ()


def test_rejects():
    faces, attributes, kinds, _ = primitive(100)
    position = [attributes[0]]
    one = np.array([trimkit.exchange.DracoKind.POSITION], dtype=np.int64)
    with pytest.raises(TypeError):
        trimkit.exchange.draco_encode(faces, [attributes[0].astype(np.float64)], one)
    with pytest.raises(ValueError):
        trimkit.exchange.draco_encode(faces + 100, position, one)
    with pytest.raises(ValueError, match="like the rest"):
        trimkit.exchange.draco_encode(
            faces, [np.zeros(100, np.float32), np.zeros(99, np.float32)], np.array([4, 4], dtype=np.int64)
        )
    with pytest.raises(ValueError, match="one entry per attribute"):
        trimkit.exchange.draco_encode(faces, position, np.array([0, 0], dtype=np.int64))
    with pytest.raises(ValueError, match="DracoKind"):
        trimkit.exchange.draco_encode(faces, position, np.array([5], dtype=np.int64))
    with pytest.raises(ValueError, match="at least one"):
        trimkit.exchange.draco_encode(faces, [], np.zeros(0, dtype=np.int64))
    buffer = trimkit.exchange.draco_encode(faces, attributes, kinds)
    with pytest.raises(ValueError, match="no draco attribute 99"):
        trimkit.exchange.draco_decode(buffer, np.array([99], dtype=np.int64))
    with pytest.raises(ValueError, match="not a draco attribute"):
        trimkit.exchange.draco_decode(buffer, np.array([-1], dtype=np.int64))


def test_corrupt_buffers_raise_not_crash():
    faces, attributes, kinds, _ = primitive(200)
    buffer = trimkit.exchange.draco_encode(faces, attributes, kinds)
    rng = np.random.default_rng(1)
    for i in range(300):
        if i % 2:
            corrupt = buffer[: rng.integers(0, len(buffer))]
        else:
            raw = np.frombuffer(buffer, dtype=np.uint8).copy()
            hit = rng.integers(0, len(raw), 8)
            raw[hit] = rng.integers(0, 256, 8)
            corrupt = raw.tobytes()
        try:
            decoded_faces, (position,) = trimkit.exchange.draco_decode(corrupt, np.array([0], dtype=np.int64))
        except ValueError:
            continue
        # a buffer that still decodes must still be internally consistent
        assert decoded_faces.max(initial=-1) < len(position)


def test_empty_primitive_refused():
    # found by Atheris: draco refuses an attribute with no values, which was
    # dereferenced without a check
    with pytest.raises(ValueError, match="without vertices"):
        trimkit.exchange.draco_encode(
            np.zeros((0, 3), dtype=np.int64), [np.zeros((0, 3), np.float32)], np.array([0], dtype=np.int64)
        )


def test_attribute_without_components_refused():
    # found by Atheris: a (n, 0) attribute indexed past an empty vector in draco
    with pytest.raises(ValueError, match="must be"):
        trimkit.exchange.draco_encode(
            np.zeros((0, 3), dtype=np.int64), [np.zeros((10, 0), np.float32)], np.array([0], dtype=np.int64)
        )


@pytest.mark.parametrize("bits", [0, 1, 8, 14, 22, 24])
@pytest.mark.parametrize("name", ["POSITION", "TEXCOORD", "COLOR", "GENERIC"])
def test_quantization_bits_bound_the_error(name, bits):
    # the one thing that decides how much precision survives: at any bits, no
    # value moves past half a step across the extent, and 0 keeps every bit
    runs_here("exchange.draco_encode")
    kind = trimkit.exchange.DracoKind[name]
    value = np.random.default_rng(bits).random((500, 3)).astype(np.float32) * 10
    faces = np.arange(498, dtype=np.int64).reshape((-1, 3))
    buffer = trimkit.exchange.draco_encode(faces, [value], np.array([kind], np.int64), quantization_bits=bits)
    _, (out,) = trimkit.exchange.draco_decode(buffer, np.zeros(1, np.int64))
    if bits == 0:
        assert np.array_equal(out, value)
    else:
        step = np.ptp(value) / (2**bits - 1)
        # float32 rounds too, by an ulp of the largest value
        assert np.abs(out - value).max() <= 0.5 * step + 2 * np.spacing(np.float32(10))


def test_quantization_bits_only_touch_floats():
    runs_here("exchange.draco_encode")
    value = np.random.default_rng(0).integers(0, 200, (30, 3)).astype(np.uint8)
    faces = np.arange(30, dtype=np.int64).reshape((-1, 3))
    kind = np.array([trimkit.exchange.DracoKind.COLOR], np.int64)
    buffer = trimkit.exchange.draco_encode(faces, [value], kind, quantization_bits=2)
    assert np.array_equal(trimkit.exchange.draco_decode(buffer, np.zeros(1, np.int64))[1][0], value)


@pytest.mark.parametrize(
    "value",
    [
        np.array([-(2**31), -2] * 3, np.int32),
        np.array([1, 2**31 - 1] * 3, np.int32),
        # 8 GiB of table to count these, before draco-symbol-encoding.patch
        np.array([0, 2**31 - 2] * 3, np.uint32),
        np.random.default_rng(0).integers(0, 2**31 - 1, 498, dtype=np.int64).astype(np.uint32),
    ],
)
def test_integers_of_the_widest_range_round_trip(value):
    runs_here("exchange.draco_encode")
    faces = np.arange(len(value), dtype=np.int64).reshape((-1, 3))
    kind = np.array([trimkit.exchange.DracoKind.GENERIC], np.int64)
    buffer = trimkit.exchange.draco_encode(faces, [value], kind)
    assert np.array_equal(trimkit.exchange.draco_decode(buffer, np.zeros(1, np.int64))[1][0], value)


RANGE_TOO_WIDE = "attributes\\[0\\] has a range of integers over 2\\*\\*31 - 2"


@pytest.mark.parametrize(
    "value, message",
    [
        # the largest symbol there is, which draco read as -1 and wrote through null
        (np.array([0, -(2**31)] * 3, np.int32), RANGE_TOO_WIDE),
        (np.array([-(2**31), 2**31 - 1] * 3, np.int32), RANGE_TOO_WIDE),
        # one past the widest: draco would write this, then refuse to read it
        (np.array([0, 2**31 - 1] * 3, np.uint32), RANGE_TOO_WIDE),
        # a narrow range, past what int32 holds
        (np.array([2**31, 2**31 + 5] * 3, np.uint32), "Failed to encode point attributes"),
    ],
)
def test_integers_draco_cannot_store_refused(value, message):
    runs_here("exchange.draco_encode")
    faces = np.arange(6, dtype=np.int64).reshape((-1, 3))
    kind = np.array([trimkit.exchange.DracoKind.GENERIC], np.int64)
    with pytest.raises(ValueError, match="draco failed to encode: " + message):
        trimkit.exchange.draco_encode(faces, [value], kind)


@pytest.mark.timing
def test_wide_integers_cost_no_table():
    # 8 GiB and 3.6 s here, before draco-symbol-encoding.patch
    runs_here("exchange.draco_encode")
    kind = np.array([trimkit.exchange.DracoKind.GENERIC], np.int64)
    value = np.random.default_rng(0).integers(0, 2**31 - 1, 498, dtype=np.int64).astype(np.uint32)
    faces = np.arange(498, dtype=np.int64).reshape((-1, 3))
    start = time.perf_counter()
    trimkit.exchange.draco_encode(faces, [value], kind)
    assert time.perf_counter() - start < 0.5


@pytest.mark.parametrize("bad", [-1, 25, 30])
def test_quantization_bits_refused(bad):
    faces = np.zeros((1, 3), np.int64)
    kind = np.array([trimkit.exchange.DracoKind.POSITION], np.int64)
    with pytest.raises(ValueError, match="quantization_bits"):
        trimkit.exchange.draco_encode(faces, [np.zeros((1, 3), np.float32)], kind, quantization_bits=bad)
