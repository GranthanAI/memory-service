"""Offline evaluation of the memory pipeline (summaries, anchor ledger, branches, fact supersede).

Everything here is deterministic and runs without a model: a scripted "lossy summarizer" stands in for the LLM. Run it with
`make eval`. An optional real-model mode exists for manual use only (see `evaluation.run --real`)."""
