"""
app/prompts/summary_prompt.py

Prompt templates for the short-term conversation summary generation task.
"""

SUMMARY_SYSTEM_PROMPT = (
    "You are a memory service assistant. Your goal is to update the short-term conversation summary "
    "by incorporating new messages into the previous summary chronologically."
)

SUMMARY_USER_PROMPT_TEMPLATE = """Previous Summary:
{previous_summary}

New Messages:
{new_messages_json}

Please generate an updated summary that integrates the new messages with the previous summary.
"""
