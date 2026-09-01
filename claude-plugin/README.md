# Claude Code assets

The ResearchForge Claude skills live **inside the Python package** — at
[`src/researchforge/claude/skills/`](../src/researchforge/claude/skills/) —
so that `pip install researchforge` carries them and
`researchforge init --claude` (or `researchforge claude install`) can copy
them into a project's `.claude/skills/`.

This directory exists only as a pointer for people who expect the spec's
`claude-plugin/` layout. What the skills may and may not do is enforced by the
engine, not by the skills — see
[Security & Isolation](../README.md#security--isolation--what-runs-where).

No hooks ship in Phase 1: the spec defines no hook behavior, and skills are
deliberately not a security boundary — enforcement lives in the Python
engine.
