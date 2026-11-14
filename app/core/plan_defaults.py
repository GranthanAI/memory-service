"""Bundled copy of the Free plan's limits (billing-service config/plans.yaml, catalog 2026-11-09).

Used only when billing cannot be reached and no earlier entitlement is cached (integration plan 4.7).
tests/unit/test_entitlements_cap.py checks that this copy matches plans.yaml whenever the billing repo is next to this one.
-1 means unlimited everywhere.
"""

FREE_LIMITS: dict[str, int] = {
    "messages_per_day": 25,
    "best_per_day": 0,
    "branches_per_tree": 20,
    "messages_per_conversation": 50,
    "memory_facts": 50,
    "context_tokens": 16000,
    "documents_per_tree": 3,
    "file_size_mb": 10,
    "youtube_per_month": 3,
    "storage_mb": 100,
    "custom_modes": 3,
    "mode_instruction_chars": 1000,
}
