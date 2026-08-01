import zlib

def compress_string(text: str, level: int = 6) -> bytes:
    """
    Compresses a UTF-8 string to zlib bytes.
    Useful for reducing cache storage footprint for long summaries in Redis.
    """
    if not text:
        return b""
    return zlib.compress(text.encode("utf-8"), level=level)

def decompress_to_string(compressed: bytes) -> str:
    """
    Decompresses zlib bytes back to a UTF-8 string.
    """
    if not compressed:
        return ""
    return zlib.decompress(compressed).decode("utf-8")
