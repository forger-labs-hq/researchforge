# Architecture — Phase 1 (complete)

> This document describes the codebase at the **Phase 1 open-source beta**.
> See [demo.md](demo.md) to watch it run, [research-mode.md](research-mode.md)
> for the research workflow, and [experiment-mode.md](experiment-mode.md) for the
> contract → baseline → funnel → shipping workflow, including
> [driving the loop from Claude Code or Cursor](experiment-mode.md#driving-the-loop-from-claude-code-or-cursor).
> The security model is in the README under
> [Security & Isolation](../README.md#security--isolation--what-runs-where).

## Module layout

```
src/researchforge/
├── cli.py           # root Typer app: doctor / init / status + sub-app mounting
├── config/          # paths.py (.researchforge/ layout), settings.py (pipeline knobs)
├── domain/          # framework-agnostic pydantic models
│   ├── project.py       # Project, ProjectMode, ProjectStatus
│   ├── paper.py         # Paper (spec paper schema)
│   ├── hypothesis.py    # Hypothesis (spec hypothesis schema)
│   ├── evidence.py      # EvidenceClaim (published_claim | interpretation | speculation)
│   ├── landscape.py     # ResearchLandscape, ResearchDirection, PaperAnnotation
│   └── repo_scan.py     # RepoScan, CompatibilityStatus
├── project/         # project create/show (service + cli)
├── repository/      # read-only repo scanner + `repo scan`
├── research/        # retrieval + synthesis handshake
│   ├── arxiv_client.py  # Atom parsing, paging, 3s politeness, retries
│   ├── dedup.py         # id- and title-based deduplication
│   ├── ranking.py       # deterministic TF-IDF cosine relevance
│   ├── queries.py       # fallback query generation (>=3 distinct)
│   ├── text.py          # shared tokenizer/stopwords
│   ├── search_service.py# fetch → dedup → rank → persist orchestration
│   ├── context_export.py# synthesis bundle w/ embedded JSON Schemas
│   ├── importers.py     # artifact validation + import (the enforcement boundary)
│   └── cli.py           # `research` + `papers` sub-apps
├── hypotheses/      # `hypotheses import/list/show`
├── contract/        # researchforge.yaml wizard/validate/approve (immutable versions)
├── execution/       # worktrees, runners, evaluation, path guard, funnel, ranking,
│                    #   validation, baseline — the enforcement layer
├── experiments/     # experiment-plan handshake (context export, patch import, CLI)
├── shipping/        # ship branch (clean reconstruction) + ship pr (gh seam)
├── reporting/       # research + engineering reports, research package, `paper package`
├── claude/          # packaged Claude Code skills + manifest-based installer
│   └── skills/      #   12 SKILL.md assets shipped inside the wheel
├── analytics/       # opt-in local-only beta analytics (spec §20)
├── storage/         # sqlite persistence boundary (one repository module per entity)
└── utils/           # system_checks (doctor), output (JsonOption/echo), artifact_io
```

`claude/skills/` is copied into a project's `.claude/skills/` by
`researchforge init --claude` / `researchforge claude install`. A sha256
manifest (`.researchforge/claude-skills-manifest.json`) tracks what was
installed: user-modified or foreign files are never overwritten or removed
without `--force`. Skills are UX only — no security property depends on
them.

## Shipping safety invariants (Phase 1D)

- `ship branch` builds the winning commit in a temporary detached worktree
  and materializes it with `git branch` — the user's HEAD/checkout is never
  touched, the branch's only parent is the frozen baseline commit, and its
  diff is asserted to equal the validated changed files. Nothing is pushed.
- `ship pr` is opt-in twice (contract `shipping.allow_draft_pr` + typed
  confirmation), pushes exactly one ref, and always opens a **draft** PR.
- Every shipped artifact is recorded in the `deliverables` table.

## Key design decisions

**Claude proposes; Python enforces.** Synthesis artifacts (landscape,
hypotheses) are validated by `research/importers.py` in layers — safe parse,
pydantic schema, referential integrity, uniqueness, cross-field rules —
transactionally, with field-level errors. Paper↔hypothesis back-links are
always recomputed by the CLI, never accepted from artifacts. The context
bundle embeds `model_json_schema()` from the same models the importers use,
so producer and validator cannot drift. Free text from artifacts and paper
abstracts is data: stored and rendered, never executed or interpolated into
commands (spec §18 prompt-injection posture).

**Acceptance criteria live in the type system where possible.**
`NoveltyConfidence` has no `high` member (novelty guarantees are
unrepresentable); `Hypothesis.evidence_status` is a computed field
(supported/unsupported cannot be claimed, only derived); landscape models
forbid unknown keys.

**Deterministic retrieval.** Ranking is TF-IDF cosine with IDF computed over
the fetched candidate set — no model call, fully reproducible, labeled
advisory. Query generation has a deterministic fallback so the CLI works
standalone; Claude normally supplies better queries via `-q`.

**Storage pattern.** One table per entity; the full pydantic model is stored
in a `record` JSON column plus a few extracted columns for keys and sorting.
Schema is versioned (`meta.schema_version`, currently 2) with additive
`CREATE TABLE IF NOT EXISTS` migrations applied by `ensure_schema` on every
open — older databases upgrade silently.

## The `.researchforge/` on-disk contract

```
.researchforge/
├── researchforge.db      # sqlite: meta, projects, repo_scans, papers,
│                         #   search_runs, landscape, evidence_claims, hypotheses
├── config.json           # optional ResearchSettings overrides
├── synthesis/            # created by `research context`
│   ├── context.json      #   CLI → Claude
│   ├── landscape.yaml    #   Claude → CLI (imported, validated)
│   └── hypotheses.yaml   #   Claude → CLI (imported, validated)
└── reports/
    └── research-report.md
```

`worktrees/` and `artifacts/` (spec §9.2) remain deferred: **future Phase 1C
execution code creates them lazily on first use** — nothing assumes `init`
pre-created them.

## Storage assumptions

One `Project` row per `.researchforge/` directory (single-project model).
All Phase 1A tables carry a `project_id` column, so multi-project support
later is a query change, not a schema rewrite. Re-running `research search`
replaces the stored paper set, but is refused (without `--force`) once
hypotheses cite papers, to avoid orphaned citations.
