"""Kafka compression codecs. Other services (billing, auth, ...) may publish gzip, snappy, lz4 or zstd batches; aiokafka needs
the `cramjam` package for the last three. Without it a consumer that meets such a batch raises on every poll and never
advances. Checked once at startup so the problem shows up at deploy time, not as a stuck consumer later."""

import logging

logger = logging.getLogger("memory_service.events.codecs")


def missing_codecs() -> list[str]:
    import aiokafka.codec as codec

    checks = {"gzip": codec.has_gzip, "lz4": codec.has_lz4, "snappy": codec.has_snappy, "zstd": codec.has_zstd}
    return [name for name, available in checks.items() if not available()]


def check_kafka_codecs() -> None:
    """Raises RuntimeError naming the missing codecs (fail fast)."""
    missing = missing_codecs()
    if missing:
        logger.critical("kafka_codec_missing: %s", ", ".join(missing))
        raise RuntimeError(f"Kafka codecs not available: {', '.join(missing)} (install cramjam)")
    logger.info("kafka_codecs_ok")
