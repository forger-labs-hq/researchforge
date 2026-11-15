"""ResearchForge built-in AI providers — standalone synthesis without Claude Code/Cursor.

Auto-detects the provider from environment variables:
  ANTHROPIC_API_KEY  → Anthropic (Claude)
  GEMINI_API_KEY / GOOGLE_API_KEY → Google (Gemini)
  OPENAI_API_KEY → OpenAI (GPT)

Override model via RESEARCHFORGE_LLM=<model-name>
  Examples: claude-opus-4-5 | gemini-3.6-flash | gpt-4o | ollama/qwen3:14b (local, no key;
  server at OLLAMA_HOST, default http://localhost:11434)
"""

from researchforge.ai.providers import AiProvider, get_provider

__all__ = ["AiProvider", "get_provider"]
