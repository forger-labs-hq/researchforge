"""Token accounting for model calls.

Providers report usage into a context-local ledger rather than returning it, so
that adding metering did not require changing ``generate()``'s signature at
every call site and in every test double. The trade is deliberate: a caller that
wants the numbers opts in by opening a :func:`recording` block, and a caller that
does not is unaffected and unslowed.

Tokens are the durable fact and dollars are derived. Prices change, vary by
contract, and are configuration rather than measurement — so what gets stored is
the token count and the model that produced it, and money is computed at read
time from rates the user can see and correct.
"""

from __future__ import annotations

import contextvars
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

__all__ = [
    "AiCall",
    "Ledger",
    "Usage",
    "record_call",
    "recording",
]


@dataclass(frozen=True)
class Usage:
    """Tokens consumed by one model call."""

    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class AiCall:
    """One model call, attributed to the work that asked for it."""

    purpose: str
    """What the call was for — planning, synthesis, observation, evaluation."""

    provider: str
    model: str
    usage: Usage
    duration_seconds: float = 0.0

    estimated: bool = False
    """True when the tokens were sized rather than reported.

    A loop driven from an IDE spends its tokens inside that IDE, where
    ResearchForge cannot observe them — all it knows is how much context it
    handed over and how much came back. That is worth showing, and worth never
    confusing with a metered count.
    """


@dataclass
class Ledger:
    """Calls recorded inside one :func:`recording` block."""

    calls: list[AiCall] = field(default_factory=list)

    @property
    def input_tokens(self) -> int:
        return sum(c.usage.input_tokens for c in self.calls)

    @property
    def output_tokens(self) -> int:
        return sum(c.usage.output_tokens for c in self.calls)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


_ledger: contextvars.ContextVar[Ledger | None] = contextvars.ContextVar(
    "researchforge_ai_ledger", default=None
)


@contextmanager
def recording() -> Iterator[Ledger]:
    """Collect every model call made inside the block.

    Nesting is intentionally *not* additive: an inner block captures its own
    calls and the outer block does not see them. A caller that opens a block is
    asking "what did *this* cost", and silently double-counting into an
    enclosing total would make both figures wrong.
    """
    ledger = Ledger()
    token = _ledger.set(ledger)
    try:
        yield ledger
    finally:
        _ledger.reset(token)


def record_call(
    purpose: str,
    provider: str,
    model: str,
    usage: Usage,
    duration_seconds: float = 0.0,
) -> None:
    """Note a model call, if anyone is listening. A no-op otherwise."""
    ledger = _ledger.get()
    if ledger is None:
        return
    ledger.calls.append(
        AiCall(
            purpose=purpose,
            provider=provider,
            model=model,
            usage=usage,
            duration_seconds=duration_seconds,
        )
    )


def current_purpose() -> str:
    """The purpose label calls should be filed under right now."""
    return _purpose.get()


_purpose: contextvars.ContextVar[str] = contextvars.ContextVar(
    "researchforge_ai_purpose", default="other"
)


@contextmanager
def purpose(label: str) -> Iterator[None]:
    """Attribute model calls made inside the block to a named activity.

    Set at the call site that knows *why* it is calling — the provider only
    knows that it was called.
    """
    token = _purpose.set(label)
    try:
        yield
    finally:
        _purpose.reset(token)
