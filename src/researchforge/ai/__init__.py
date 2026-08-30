"""ResearchForge built-in AI providers — standalone synthesis without Claude Code/Cursor.

Auto-detects the provider from environment variables:
  ANTHROPIC_API_KEY  → Anthropic (Claude)
  GEMINI_API_KEY / GOOGLE_API_KEY → Google (Gemini)
  OPENAI_API_KEY → OpenAI (GPT)

Override model via RESEARCHFORGE_LLM=<model-name>
  Examples: claude-opus-4-5 | gemini-2.0-flash | gpt-4o | http://localhost:11434/api (Ollama)
"""

from researchforge.ai.providers import AiProvider, get_provider

__all__ = ["AiProvider", "get_provider"]
