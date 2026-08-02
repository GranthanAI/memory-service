"""
app/utils/compression.py

zstd compression and decompression utilities for reducing memory storage footprint.

Uses the zstandard library to compress/decompress summary text or other long strings
stored in Redis. zstd offers superior compression ratios and speed compared to zlib.
"""

import zstandard as zstd


def compress_string(text: str, level: int = 3) -> bytes:
    """
    Compresses a UTF-8 string using zstd compression.
    Default compression level is 3 (balanced speed/ratio).
    """
    if not text:
        return b""
    cctx = zstd.ZstdCompressor(level=level)
    return cctx.compress(text.encode("utf-8"))


def decompress_to_string(compressed: bytes) -> str:
    """
    Decompresses zstd-compressed bytes back into a UTF-8 string.
    """
    if not compressed:
        return ""
    dctx = zstd.ZstdDecompressor()
    return dctx.decompress(compressed).decode("utf-8")
