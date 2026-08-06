"""
app/prompts/fact_prompt.py

Prompt templates for the long-term user fact extraction task.
"""

FACT_SYSTEM_PROMPT = (
    "You are a memory service assistant. Extract key structured user facts (preferences, habits, plans, goals) "
    "from the conversation summary. Formulate each fact in a clear, atomic format: '<category>:<importance_float>:<statement>'."
)

FACT_USER_PROMPT_TEMPLATE = """Summary:
{summary}

Extract any facts from this summary using the format: '<category>:<importance_float>:<statement>'.
Ensure each fact is on a new line and represents a single key truth.
"""
