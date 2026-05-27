"""
AISec framework integrations.

Each submodule provides a native adapter for a specific
AI agent framework, allowing AISec to intercept real
agent tool calls without modifying the agent's source code.

Available integrations:
    langchain  — LangChain callback-based interceptor (v1)

Planned integrations (v2):
    autogen    — Microsoft AutoGen tool interceptor
    crewai     — CrewAI agent interceptor
    openai     — OpenAI function-calling interceptor
"""