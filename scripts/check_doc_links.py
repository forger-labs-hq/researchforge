"""Check that every relative Markdown link and heading anchor resolves.

    python scripts/check_doc_links.py [file ...]

Defaults to the README, the docs/ pages, and the example READMEs. Anchors are
slugified the way GitHub does, so a renamed heading is caught here rather than
by a reader hitting a dead link.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT = [
    "README.md",
    *sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "docs").glob("*.md")),
    *sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "examples").glob("*/README.md")),
]


def slug(heading: str) -> str:
    return re.sub(r"[^\w\s-]", "", heading.strip().lower()).replace(" ", "-")


def headings(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    return [slug(m.group(2)) for m in re.finditer(r"^(#{1,6})\s+(.*)$", text, re.M)]


def main() -> int:
    files = [Path(a) for a in sys.argv[1:]] or [ROOT / f for f in DEFAULT]
    anchors = {f.resolve(): headings(f) for f in files if f.is_file()}

    problems: list[str] = []
    for file in files:
        if not file.is_file():
            problems.append(f"{file}: not found")
            continue
        for target in re.findall(r"\]\(([^)\s]+)\)", file.read_text(encoding="utf-8")):
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            path_part, _, anchor = target.partition("#")
            resolved = file.resolve() if not path_part else (file.parent / path_part).resolve()
            if not resolved.exists():
                problems.append(f"{file}: missing target {target}")
                continue
            known = anchors.get(resolved)
            if anchor and known is not None and anchor not in known:
                problems.append(f"{file}: no heading for {target}")

    for problem in problems:
        print(problem)
    print(f"{len(files)} file(s) checked, {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
