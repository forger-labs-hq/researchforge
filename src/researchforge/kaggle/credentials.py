"""Kaggle API credential management.

Safely handles Kaggle API tokens:
  - Checks ~/.kaggle/kaggle.json existence and validity
  - Interactively walks the user through setup if credentials are missing
  - Validates credentials with a harmless read-only API call
  - Never logs, prints, or embeds keys anywhere
"""

from __future__ import annotations

import json
import stat
import subprocess
from pathlib import Path


class CredentialError(Exception):
    """Credential-related error; message is user-facing."""


_KAGGLE_JSON = Path.home() / ".kaggle" / "kaggle.json"


# ---------------------------------------------------------------------------
# Read / validate
# ---------------------------------------------------------------------------


def get_kaggle_credentials() -> dict[str, str]:
    """Load credentials from ~/.kaggle/kaggle.json.

    Returns:
        Dict with 'username' and 'key'.

    Raises:
        CredentialError: If the file is missing or malformed.
    """
    if not _KAGGLE_JSON.exists():
        raise CredentialError(
            f"Kaggle credentials not found at {_KAGGLE_JSON}.\n"
            "Run setup or follow https://kaggle.com/settings → API → Create New Token."
        )

    try:
        with _KAGGLE_JSON.open() as fh:
            creds = json.load(fh)
    except json.JSONDecodeError as exc:
        raise CredentialError(f"Invalid JSON in {_KAGGLE_JSON}: {exc}") from exc

    if "username" not in creds or "key" not in creds:
        raise CredentialError(
            f"{_KAGGLE_JSON} must contain 'username' and 'key' fields.\n"
            "Download a fresh token from https://kaggle.com/settings → API."
        )
    return {str(k): str(v) for k, v in creds.items()}


def validate_credentials() -> bool:
    """Return True if credentials exist and pass a harmless read-only API call."""
    try:
        get_kaggle_credentials()
        result = subprocess.run(
            ["kaggle", "config", "view"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return result.returncode == 0
    except (CredentialError, subprocess.TimeoutExpired, FileNotFoundError):
        return False


# ---------------------------------------------------------------------------
# Interactive setup — called when credentials are missing or invalid
# ---------------------------------------------------------------------------


def setup_credentials_interactive() -> bool:
    """Walk the user through creating ~/.kaggle/kaggle.json interactively.

    Returns True when credentials are working, False if the user skips.
    In non-interactive mode (Claude Code shell) prints clear instructions
    for the user to follow manually instead of blocking on input().
    Never prints the key value.
    """
    import sys

    if not sys.stdin.isatty():
        print(
            "\n🔑  Kaggle credentials not found and cannot be set up interactively.",
            file=sys.stderr,
        )
        print(
            "   Please set up credentials manually:\n"
            "   1. Go to https://www.kaggle.com/settings → API → Create New Token\n"
            "   2. Move the downloaded kaggle.json to ~/.kaggle/kaggle.json\n"
            "   3. Run: chmod 600 ~/.kaggle/kaggle.json\n"
            "   Then retry.",
            file=sys.stderr,
        )
        return False
    print("\n🔑  Kaggle API credentials not found.")
    print("   ResearchForge needs a Kaggle API token to download data and kernels.")
    print()
    print("   Steps:")
    print("   1. Open https://www.kaggle.com/settings (log in if needed)")
    print("   2. Scroll to the API section → click 'Create New Token'")
    print("   3. A file called kaggle.json will download to your computer")
    print("   4. Paste the path to that file below, OR enter username + key manually")
    print()

    choice = _prompt(
        "How would you like to set up credentials?",
        choices=["paste file path", "enter username and key manually", "skip"],
        default="paste file path",
    )

    if choice == "skip":
        print("⚠️  Skipping credential setup — some features will not work.")
        return False

    if choice == "paste file path":
        return _setup_from_file()
    else:
        return _setup_manually()


def _setup_from_file() -> bool:
    """Copy a downloaded kaggle.json to ~/.kaggle/kaggle.json."""
    while True:
        try:
            raw = input("  Path to downloaded kaggle.json: ").strip()
        except EOFError:
            return False

        if not raw:
            print("  ⚠  No path entered.")
            continue

        src = Path(raw).expanduser()
        if not src.exists():
            print(f"  ⚠  File not found: {src}")
            continue

        try:
            with src.open() as fh:
                creds = json.load(fh)
            if "username" not in creds or "key" not in creds:
                print("  ⚠  File does not contain 'username' and 'key'. Try again.")
                continue
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  ⚠  Could not read file: {exc}")
            continue

        _write_credentials(creds)
        print(f"  ✅ Credentials saved for user: {creds['username']}")
        return _verify_and_report()


def _setup_manually() -> bool:
    """Ask for username and key, write kaggle.json without ever echoing the key."""
    try:
        username = input("  Kaggle username: ").strip()
    except EOFError:
        return False

    if not username:
        print("  ⚠  Username is required.")
        return False

    # Use getpass so the key is never echoed to the terminal
    import getpass

    try:
        key = getpass.getpass("  Kaggle API key (input hidden): ")
    except (EOFError, KeyboardInterrupt):
        return False

    if not key:
        print("  ⚠  API key is required.")
        return False

    _write_credentials({"username": username, "key": key})
    print(f"  ✅ Credentials saved for user: {username}")
    return _verify_and_report()


def _write_credentials(creds: dict[str, str]) -> None:
    """Write credentials to ~/.kaggle/kaggle.json with owner-only permissions."""
    _KAGGLE_JSON.parent.mkdir(parents=True, exist_ok=True)
    _KAGGLE_JSON.write_text(json.dumps(creds, indent=2))
    # Owner read/write only — key must never be world-readable
    _KAGGLE_JSON.chmod(stat.S_IRUSR | stat.S_IWUSR)


def _verify_and_report() -> bool:
    """Run a harmless API call to confirm the credentials work."""
    print("  Verifying with Kaggle API...", end=" ", flush=True)
    ok = validate_credentials()
    if ok:
        print("✅ Credentials valid!")
    else:
        print("❌ Verification failed.")
        print("  Check that you joined the competition and accepted its rules on kaggle.com.")
    return ok


def _prompt(question: str, choices: list[str], default: str) -> str:
    choices_str = " / ".join(choices)
    prompt = f"  {question} [{choices_str}] (default: {default}): "
    while True:
        try:
            raw = input(prompt).strip().lower()
        except EOFError:
            return default
        if not raw:
            return default
        matches = [c for c in choices if c.startswith(raw)]
        if len(matches) == 1:
            return matches[0]
        if raw in choices:
            return raw
        print(f"  ⚠  Choose one of: {choices_str}")
