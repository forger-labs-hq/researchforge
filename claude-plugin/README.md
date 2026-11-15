# Claude Code assets

The ResearchForge Claude skills live **inside the Python package** — at
[`src/researchforge/claude/skills/`](../src/researchforge/claude/skills/) —
so that `pip install researchforge` carries them and
`researchforge init --claude` (or `researchforge claude install`) can copy
them into a project's `.claude/skills/`.

The same folder is also a Claude Code plugin. Its manifest is
[`src/researchforge/claude/.claude-plugin/plugin.json`](../src/researchforge/claude/.claude-plugin/plugin.json),
and this repository is its marketplace
([`.claude-plugin/marketplace.json`](../.claude-plugin/marketplace.json)):

```text
/plugin marketplace add forger-labs-hq/researchforge
/plugin install researchforge@researchforge
```

Plugin skills are namespaced (`/researchforge:overnight`). They drive the CLI,
so `pip install researchforge` is still needed. Use either the plugin or
`researchforge claude install`, not both, or every skill appears twice.

This directory exists only as a pointer for people who expect the spec's
`claude-plugin/` layout. What the skills may and may not do is enforced by the
engine, not by the skills — see
[Security & Isolation](../README.md#security--isolation--what-runs-where).

No hooks ship: skills are deliberately not a security boundary — enforcement
lives in the Python engine.
