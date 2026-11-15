"""Local Ollama models: resolution, the native /api/chat call, and zero token cost."""

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest

from researchforge.ai.providers import OllamaProvider, OpenAIProvider, get_provider
from researchforge.ai.usage import AiCall, Usage, recording
from researchforge.config.settings import DEFAULT_MODEL_PRICES
from researchforge.kaggle.dashboard_state import _tokens
from researchforge.reporting.economics import token_spend

KEYS = ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY", "OLLAMA_HOST")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (*KEYS, "RESEARCHFORGE_LLM"):
        monkeypatch.delenv(key, raising=False)


class _FakeOllama(BaseHTTPRequestHandler):
    reply: dict[str, Any] = {}
    status = 200
    seen: list[dict[str, Any]] = []

    def do_POST(self) -> None:  # noqa: N802 — http.server's naming
        length = int(self.headers["Content-Length"])
        type(self).seen.append({"path": self.path, "body": json.loads(self.rfile.read(length))})
        payload = json.dumps(type(self).reply).encode()
        self.send_response(type(self).status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args: Any) -> None:
        pass


@pytest.fixture
def ollama() -> Iterator[str]:
    _FakeOllama.seen = []
    _FakeOllama.status = 200
    server = HTTPServer(("127.0.0.1", 0), _FakeOllama)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


class TestResolution:
    def test_ollama_prefix_selects_a_local_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RESEARCHFORGE_LLM", "ollama/qwen3:14b")
        provider = get_provider()
        assert isinstance(provider, OllamaProvider)
        assert provider.name == "ollama/qwen3:14b"

    def test_native_api_url_is_ollama_not_openai(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The URL the docs used to show (…/api) is Ollama's own API, not OpenAI's."""
        monkeypatch.setenv("RESEARCHFORGE_LLM", "http://gpu-box:11434/api")
        provider = get_provider(model_hint="llama3.1")
        assert isinstance(provider, OllamaProvider)
        assert provider._host == "http://gpu-box:11434"

    def test_openai_compatible_url_stays_openai(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RESEARCHFORGE_LLM", "http://localhost:11434/v1")
        assert isinstance(get_provider(), OpenAIProvider)

    def test_provider_hint_needs_a_model(self) -> None:
        with pytest.raises(RuntimeError, match="Name the Ollama model"):
            get_provider(provider_hint="ollama")
        provider = get_provider(provider_hint="ollama", model_hint="gemma3:12b")
        assert isinstance(provider, OllamaProvider)

    def test_host_comes_from_ollama_host(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OLLAMA_HOST", "10.0.0.5:11434")
        assert OllamaProvider("qwen3")._host == "http://10.0.0.5:11434"


class TestGenerate:
    def test_calls_native_chat_and_records_usage(self, ollama: str) -> None:
        _FakeOllama.reply = {
            "message": {"content": '<think>plan</think>\n{"title": "lgbm"}'},
            "prompt_eval_count": 120,
            "eval_count": 30,
        }
        with recording() as ledger:
            text = OllamaProvider("qwen3:14b", host=ollama).generate("sys", "user", max_tokens=64)
        assert text == '{"title": "lgbm"}'
        sent = _FakeOllama.seen[0]
        assert sent["path"] == "/api/chat"
        assert sent["body"]["stream"] is False
        assert sent["body"]["options"] == {"num_predict": 64}
        assert [m["role"] for m in sent["body"]["messages"]] == ["system", "user"]
        (call,) = ledger.calls
        assert (call.provider, call.model) == ("ollama", "qwen3:14b")
        assert (call.usage.input_tokens, call.usage.output_tokens) == (120, 30)

    def test_missing_model_says_how_to_pull_it(self, ollama: str) -> None:
        _FakeOllama.status = 404
        _FakeOllama.reply = {"error": "model 'qwen3:14b' not found"}
        with pytest.raises(RuntimeError, match="ollama pull qwen3:14b"):
            OllamaProvider("qwen3:14b", host=ollama).generate("s", "u")

    def test_server_down_says_how_to_start_it(self) -> None:
        with pytest.raises(RuntimeError, match="ollama serve"):
            OllamaProvider("qwen3", host="http://127.0.0.1:9", timeout=2).generate("s", "u")


class TestCost:
    def test_research_economics_prices_local_tokens_at_zero(self) -> None:
        call = AiCall("planning", "ollama", "gpt-oss:20b", Usage(1000, 500))
        spend = token_spend([call], {"gpt": DEFAULT_MODEL_PRICES["gpt-4o"]})
        assert spend.usd == 0.0
        assert spend.unpriced_models == []

    def test_overnight_dashboard_prices_local_tokens_at_zero(self) -> None:
        calls = [
            {
                "provider": "ollama",
                "model": "gpt-oss:20b",
                "input_tokens": 9000,
                "output_tokens": 1000,
                "purpose": "proposal_round_1",
            }
        ]
        tokens = _tokens(calls, "ollama")
        assert tokens["estimated_cost_usd"] == 0.0
        assert tokens["has_unpriced"] is False
        assert tokens["unobservable"] is False
