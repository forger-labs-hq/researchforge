"""Live progress indicator for long-running CLI operations.

Provides a lightweight, dependency-free progress display that works in
any terminal (TTY or not, Unicode or ASCII).

Usage:
    with LiveProgress("Running baseline") as p:
        p.phase("Creating worktree…")
        # ... long work ...
        p.phase("Running benchmark…")
        # ... more work ...
    # prints ✓ Running baseline  (2m 14s) on exit
"""

from __future__ import annotations

import sys
import threading
import time
from types import TracebackType


class LiveProgress:
    """Thread-backed elapsed-time indicator with phase labels.

    On a TTY: rewrites a single line with a spinner and elapsed time.
    On non-TTY (CI, piped output): prints each phase as a timestamped line.
    """

    _SPINNER_UNICODE = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    _SPINNER_ASCII = "-\\|/"

    def __init__(self, label: str, *, enabled: bool = True) -> None:
        self._label = label
        self._enabled = enabled and sys.stderr.isatty()
        self._plain = enabled and not sys.stderr.isatty()
        self._phase = label
        self._start = time.monotonic()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        # Detect Unicode support
        try:
            "⠋".encode(sys.stderr.encoding or "utf-8")
            self._spinner = self._SPINNER_UNICODE
        except (UnicodeEncodeError, LookupError):
            self._spinner = self._SPINNER_ASCII

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> LiveProgress:
        if self._enabled:
            self._thread.start()
        elif self._plain:
            self._print_plain(f"Starting: {self._label}")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._stop.set()
        if self._enabled:
            self._thread.join(timeout=0.5)
            elapsed_str = self._fmt_elapsed()
            mark = "✗" if exc_type else "✓"
            sys.stderr.write(f"\r  {mark} {self._label}  ({elapsed_str})\n")
            sys.stderr.flush()
        elif self._plain and not exc_type:
            self._print_plain(f"Done: {self._label}  ({self._fmt_elapsed()})")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def phase(self, message: str) -> None:
        """Update the current phase label shown in the spinner / plain log."""
        self._phase = message
        if self._plain:
            self._print_plain(message)

    def note(self, message: str) -> None:
        """Print a line that stays on screen while the spinner keeps running.

        A phase is transient — it is overwritten by the next one — so a result
        worth keeping (an experiment's measurement, a failure) has to be
        written over the spinner's line and followed by a newline, leaving the
        spinner to redraw itself underneath.
        """
        if self._enabled:
            sys.stderr.write("\r" + " " * 100 + "\r" + message + "\n")
            sys.stderr.flush()
        elif self._plain:
            self._print_plain(message)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _fmt_elapsed(self) -> str:
        secs = int(time.monotonic() - self._start)
        m, s = divmod(secs, 60)
        return f"{m}m {s:02d}s" if m else f"{s}s"

    def _print_plain(self, message: str) -> None:
        elapsed = self._fmt_elapsed()
        sys.stderr.write(f"  [{elapsed:>5}] {message}\n")
        sys.stderr.flush()

    def _spin(self) -> None:
        chars = self._spinner
        i = 0
        while not self._stop.wait(0.1):
            elapsed_str = self._fmt_elapsed()
            sys.stderr.write(f"\r  {chars[i % len(chars)]} {self._phase}  ({elapsed_str})   ")
            sys.stderr.flush()
            i += 1
