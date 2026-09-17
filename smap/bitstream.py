"""Lossless bitstream (de)serialization for SonarMap token streams.

Two modes:
  - "none": uint16 tokens, no entropy coding (~910 kbps on FMA 10×10s)
  - "lzma": split_pack10 + lzma preset 9 extreme (~160 kbps, 5.7× smaller)

Both modes are bit-exact: unpack(pack(tokens)) == tokens.

Container format (little-endian):
    MAGIC (4B: "SMBS") | VERSION (1B) | MODE (1B) | N_TOKENS (4B) | payload

Mode "none" payload: tokens.astype(uint16).tobytes()
Mode "lzma" payload:
    patch_clen (4B) | meta_clen (4B) | patch_count (4B)
    lzma(u10_packed(patch_codes)) | lzma(u16(meta_tokens))

Patch reinterleave relies on deterministic grammar: PVQ_N_STAGES patch codes
follow every fbin token (anchor) and every dfbin token (target).
"""
from __future__ import annotations

import lzma
import numpy as np
from numpy.typing import NDArray

from smap.config import (
    FBIN_OFFSET, DDB_OFFSET,
    DFBIN_OFFSET, PATCH_OFFSET,
    TIME_SHIFT_OFFSET,
    PVQ_N_STAGES,
)

_MAGIC = b"SMBS"
_VERSION = 1
_MODE_NONE = 0
_MODE_LZMA = 1

MODES = ("none", "lzma")


def pack(tokens: NDArray, mode: str = "lzma") -> bytes:
    arr = np.asarray(tokens, dtype=np.int64).ravel()
    n = int(arr.shape[0])
    header = _MAGIC + bytes([_VERSION])

    if mode == "none":
        return header + bytes([_MODE_NONE]) + n.to_bytes(4, "little") + arr.astype(np.uint16).tobytes()
    if mode == "lzma":
        return header + bytes([_MODE_LZMA]) + n.to_bytes(4, "little") + _encode_lzma(arr)
    raise ValueError(f"unknown mode: {mode!r} (expected one of {MODES})")


def unpack(data: bytes) -> NDArray[np.int64]:
    if len(data) < 10 or data[:4] != _MAGIC:
        raise ValueError("bad magic — not a SonarMap bitstream")
    version = data[4]
    if version != _VERSION:
        raise ValueError(f"unsupported version {version}")
    mode = data[5]
    n = int.from_bytes(data[6:10], "little")
    payload = data[10:]

    if mode == _MODE_NONE:
        arr = np.frombuffer(payload, dtype=np.uint16).astype(np.int64)
        if arr.shape[0] != n:
            raise ValueError(f"token count mismatch: {arr.shape[0]} vs {n}")
        return arr
    if mode == _MODE_LZMA:
        return _decode_lzma(payload, n)
    raise ValueError(f"unknown mode id: {mode}")


def _encode_lzma(tokens: NDArray) -> bytes:
    is_patch = (tokens >= PATCH_OFFSET) & (tokens < TIME_SHIFT_OFFSET)
    patch = (tokens[is_patch] - PATCH_OFFSET).astype(np.uint16)
    meta = tokens[~is_patch].astype(np.uint16)

    patch_c = lzma.compress(_pack_u10(patch), preset=9 | lzma.PRESET_EXTREME)
    meta_c = lzma.compress(meta.tobytes(), preset=9 | lzma.PRESET_EXTREME)

    sub = (
        len(patch_c).to_bytes(4, "little")
        + len(meta_c).to_bytes(4, "little")
        + int(patch.shape[0]).to_bytes(4, "little")
    )
    return sub + patch_c + meta_c


def _decode_lzma(payload: bytes, n_tokens: int) -> NDArray[np.int64]:
    patch_clen = int.from_bytes(payload[0:4], "little")
    meta_clen = int.from_bytes(payload[4:8], "little")
    patch_count = int.from_bytes(payload[8:12], "little")
    off = 12
    patch_raw = lzma.decompress(payload[off:off + patch_clen])
    off += patch_clen
    meta_raw = lzma.decompress(payload[off:off + meta_clen])

    patch = _unpack_u10(patch_raw, patch_count).astype(np.int64)
    meta = np.frombuffer(meta_raw, dtype=np.uint16).astype(np.int64)

    out = np.empty(n_tokens, dtype=np.int64)
    mi = 0
    pi = 0
    oi = 0
    n_meta = meta.shape[0]
    while mi < n_meta:
        v = int(meta[mi])
        out[oi] = v
        oi += 1
        mi += 1
        if (FBIN_OFFSET <= v < DDB_OFFSET) or (DFBIN_OFFSET <= v < PATCH_OFFSET):
            for _ in range(PVQ_N_STAGES):
                out[oi] = int(patch[pi]) + PATCH_OFFSET
                oi += 1
                pi += 1
    if oi != n_tokens or pi != patch_count:
        raise ValueError(f"reinterleave mismatch: oi={oi}/{n_tokens}  pi={pi}/{patch_count}")
    return out


def _pack_u10(arr: NDArray) -> bytes:
    a = arr.astype(np.uint16)
    n = int(a.shape[0])
    pad = (-n) % 4
    if pad:
        a = np.concatenate([a, np.zeros(pad, dtype=np.uint16)])
    a64 = a.reshape(-1, 4).astype(np.uint64)
    packed = a64[:, 0] | (a64[:, 1] << 10) | (a64[:, 2] << 20) | (a64[:, 3] << 30)
    out = np.zeros((packed.shape[0], 5), dtype=np.uint8)
    for k in range(5):
        out[:, k] = (packed >> (8 * k)) & 0xFF
    return out.tobytes()


def _unpack_u10(data: bytes, n: int) -> NDArray[np.uint16]:
    if n == 0:
        return np.zeros(0, dtype=np.uint16)
    n_blocks = (n + 3) // 4
    buf = np.frombuffer(data, dtype=np.uint8)
    if buf.size != n_blocks * 5:
        raise ValueError(f"u10 payload size {buf.size} != expected {n_blocks * 5}")
    arr = buf.reshape(n_blocks, 5)
    packed = np.zeros(n_blocks, dtype=np.uint64)
    for k in range(5):
        packed |= arr[:, k].astype(np.uint64) << (8 * k)
    out = np.empty(n_blocks * 4, dtype=np.uint16)
    mask = np.uint64((1 << 10) - 1)
    out[0::4] = (packed & mask).astype(np.uint16)
    out[1::4] = ((packed >> 10) & mask).astype(np.uint16)
    out[2::4] = ((packed >> 20) & mask).astype(np.uint16)
    out[3::4] = ((packed >> 30) & mask).astype(np.uint16)
    return out[:n]


def kbps(data: bytes, duration_s: float) -> float:
    return len(data) * 8.0 / duration_s / 1000.0 if duration_s > 0 else 0.0
