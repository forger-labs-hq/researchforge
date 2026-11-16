"""Turn a pulled public notebook into a round proposal, without running anything here.

A notebook's code cells become one script, run on the competition's compute like
any round. IPython-only lines are rewritten so the script runs as plain Python:
``!cmd`` and ``%pip`` become subprocess calls, ``%%bash`` / ``%%writefile`` cells
become their Python equivalents, and other magics are commented out. The
notebook's attached datasets go into ``dataset_sources``.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

_SHIM = (
    "# ---- ResearchForge: notebook → script shim ----\n"
    "try:\n"
    "    display  # noqa: B018\n"
    "except NameError:\n"
    "    display = print\n"
)


def find_notebook(kernel_dir: Path) -> Path | None:
    """The notebook (or script) pull-kernel saved in ``kernel_dir``."""
    files = [p for p in kernel_dir.iterdir() if p.is_file() and p.suffix in (".ipynb", ".py")]
    named = [p for p in files if p.stem == kernel_dir.name]
    ranked = named or sorted(files, key=lambda p: (p.suffix != ".ipynb", p.name))
    return ranked[0] if ranked else None


def _shell(indent: str, cmd: str) -> str:
    return f'{indent}__import__("subprocess").run({cmd.strip()!r}, shell=True, check=False)'


def _line(line: str) -> str:
    # `%name` and `!cmd` only: a line such as `% (a, b)` continues a `%` format
    # expression, and `!= x` continues a comparison.
    m = re.match(r"^(\s*)(!(?!=)|%(?=[A-Za-z]))(.*)$", line)
    if not m:
        return line
    indent, mark, rest = m.groups()
    if mark == "!":
        return _shell(indent, rest)
    if rest.startswith(("pip ", "conda ")):
        return _shell(indent, rest)
    if rest.startswith("cd "):
        return f'{indent}__import__("os").chdir({rest[3:].strip()!r})'
    return f"{indent}pass  # notebook magic, not run as a script: %{rest}"


def cell_to_script(src: str) -> str:
    """One code cell as plain Python."""
    first, _, body = src.partition("\n")
    head = first.strip()
    if head.startswith(("%%bash", "%%sh", "%%script bash")):
        return _shell("", body)
    if head.startswith("%%writefile"):
        parts = head.split()
        name = parts[-1] if len(parts) > 1 else "cell.txt"
        mode = "a" if "-a" in parts[1:-1] else "w"
        return f"with open({name!r}, {mode!r}) as _f:\n    _f.write({body!r})"
    if head.startswith("%%"):
        return f"# notebook cell magic, not run as a script: {head}\n" + cell_to_script(body)
    return "\n".join(_line(x) for x in src.split("\n"))


def notebook_code(path: Path) -> str:
    """The notebook's code cells in order as one script (a .py kernel as it is)."""
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".py":
        return text
    cells = []
    for c in json.loads(text).get("cells", []):
        if c.get("cell_type") != "code":
            continue
        src = c.get("source") or ""
        src = "".join(src) if isinstance(src, list) else str(src)
        if src.strip():
            cells.append(cell_to_script(src))
    return "\n\n\n".join(cells)


def check_syntax(code: str) -> str:
    """'' when ``code`` parses, else where it does not."""
    try:
        ast.parse(code)
    except SyntaxError as exc:
        bad = (exc.text or "").strip()
        return f"line {exc.lineno}: {exc.msg}" + (f" — {bad[:120]}" if bad else "")
    return ""


def build_proposal(
    kernel_dir: Path,
    ref: str,
    *,
    title: str,
    description: str = "",
    rationale: str = "",
    expected_impact: str = "",
    hypothesis: str = "",
    parent: int | None = None,
    branch: str = "model",
    append: str = "",
    extra_datasets: list[str] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """(proposal, notes). Raises ValueError when the result is not valid Python."""
    nb = find_notebook(kernel_dir)
    if nb is None:
        raise ValueError(f"no .ipynb or .py in {kernel_dir}")
    meta_path = kernel_dir / "kernel-metadata.json"
    meta_raw = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    meta = meta_raw if isinstance(meta_raw, dict) else {}
    code = (
        f"'''Round from the public notebook {ref}, its code cells in order"
        + (" plus scoring code appended by the loop" if append else "")
        + ".'''\n\n"
        + _SHIM
        + "\n\n"
        + notebook_code(nb)
    )
    if append:
        code += "\n\n\n# ---- ResearchForge: appended code ----\n" + append
    problem = check_syntax(code)
    if problem:
        raise ValueError(f"the script does not parse ({problem})")
    from researchforge.kaggle.api import kernel_inputs

    # Everything the notebook attaches goes with the round: datasets, other notebooks'
    # outputs, Kaggle Models, and its Docker image and GPU type.
    inputs = kernel_inputs(
        {**meta, "dataset_sources": [*(meta.get("dataset_sources") or []), *(extra_datasets or [])]}
    )
    notes = []
    if not append:
        notes.append(
            "no scoring code appended: unless the notebook prints the metric itself, the round "
            "records no score (append scoring code with --append)"
        )
    proposal = {
        "title": title,
        "description": description or f"Runs the public notebook {ref} as a round.",
        "code_changes": code,
        "rationale": rationale,
        "expected_impact": expected_impact,
        "source_credit": ref,
        "parent_round_num": parent,
        "branch_tag": branch,
        "dataset_sources": inputs.pop("dataset_sources", []),
        **inputs,
        "hypothesis": hypothesis,
    }
    return proposal, notes
