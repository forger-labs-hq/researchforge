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

## Kaggle dashboard: setup, keys and steering

The live dashboard (`kaggle/dashboard.py`, one HTML shell in `dashboard_app.html`) can do
everything `/overnight` does in a chat. Each surface calls the same Python, so a project
started on one surface can be continued on the other.

| Module | What it owns | Files |
|---|---|---|
| `kaggle/new_project.py` | The **New competition** wizard: default answers from `gather-facts`, the `researchforge kaggle …` commands setup runs, the keys the chosen setup needs | `.overnight_<slug>/setup_draft.json` (facts, answers, the running job; deleted once setup succeeds) |
| `kaggle/keys.py` + `keys_cli.py` | **Keys & accounts**: one registry of every credential, its status, set/remove, and a harmless test call | `~/.kaggle/access_token` or `kaggle.json`; `~/.config/researchforge/auto_run.env` (mode 600); `~/.config/researchforge/key_tests.json` (results only) |
| `kaggle/steering.py` | **Steer**: the person's directions (`standing` or for the `next` round), marked read by `prepare-round`, answered by the proposer | `.overnight_<slug>/steering.json` |

**Endpoints.** POST endpoints need the page's per-process token and a localhost `Host`.
`/api/keys/{set,test,remove,colab}` and `/api/new/{facts,draft,setup,discard,cloud-gpus}`
are global. `/api/steer` and `/api/loop` (pause, resume, re-synthesize, add idea) take a
project. Setup runs as a background job in the dashboard process. Each step is a
`researchforge kaggle` subprocess run in the competition's folder, with the agent-host
markers removed so setup does not claim it ran inside Claude Code or Cursor. The job's
progress is in `/api/state` (`jobs`) and is mirrored into the draft.

**Secrets.** A value crosses the network once, from the page to `set_key`, and is never
returned. `/api/state` carries names, where each one is set, and the last test result.
`keys set` and `driver-login` refuse to run without a terminal, so no agent can feed them a
value. In an IDE, the overnight skill finds what is missing with `keys status`, opens the
page with `keys open <name>`, and waits with `keys wait <name>`.

**Directions.** `prepare-round` calls `steering.mark_round(round)`. It marks open `next`
directions as read by that round, and closes those an earlier round read as applied. The
active ones go out as `directions` (with `reply_needed`) and at the head of `person_notes`,
which auto-run's prompt also includes. The proposer replies with `kaggle steer --reply`.
A direction is text for the proposer only: it never answers a question, approves a plan or
submits.

## Kaggle rounds on Colab, round time limit and alerts

- `kaggle/colab.py` `run_round` runs a round detached: `colab new` (GPU, high-RAM from
  `colab_config.json`), `colab upload` of the wrapped script, a `colab exec` that starts it
  under `bash` in its own session (exit code to `/content/rf_round/rc`, output to
  `out.log`), then a `colab exec` check every minute until it ends or the round time limit
  passes (then it is killed). Each `exec` renews the runtime token that `colab run` never
  renewed. Up to `MAX_FAILED_POLLS` failed checks in a row are tolerated. The log and the
  submission file (`SUBMISSION_FILE:` or a fresh `submission.*`) are downloaded, and
  `colab stop` always runs. `BenchmarkRunner._run_colab_full` keeps the file under
  `round_outputs/`, which makes the round submittable.
- `kaggle/round_target.py`: `brief()` is `prepare-round`'s `compute` (where the code runs,
  `time_limit_minutes`, rules); `refusal()` stops a Colab round that pushes a Kaggle notebook
  in `execute-round` and `run-parallel`, before a round number is used.
- `run_config.round_time_limit_seconds()` (`round_time_limit_minutes` in
  `overnight_setup.json`, default 30) replaces the fixed 1800 s in `execute-round`,
  `run-parallel` and the in-process loop.
- `control.notify_topic()` resolves the alert topic: `NTFY_TOPIC`, the project's
  `ntfy_topic` (or a `ntfy_topic` setup answer), then `notify.json`. `notifications.service`
  sends nothing without a topic. `run_config.notify_view()` / `send_test_alert()` back Run
  configs → Notifications and `POST /api/notify-test`.

## Unscored rounds, submission questions and the notebook board

- `benchmark_runner.UNSCORED_REASON`: a run that finished and kept a submission but printed no
  score (Kaggle kernel, Colab or local). The result is not passed but carries `outputs`, so
  `round_outputs.json` records it. `UNSCORED_PATTERN` also matches the older wordings.
- `submission.unscored_rounds()` lists those rounds that are submittable; `offer_missed()` (each
  monitor check and after `execute-round`) asks once about them and about the best round.
- `control.submission_round()` / `ask_submission()`: a yes/no question asking to submit round N
  becomes kind `submission_approval` with `answer_key round:N`, whoever asks it (`kaggle ask`,
  `kaggle offer-submission`, the loop). Only that kind submits on "yes".
- `dashboard_state._overnight_experiment(…, prior_best)`: with no baseline score, a round marked
  improved is checked against the best earlier round; `status: "unscored"` for unscored rounds.
- `kernel_watcher.run_watch_cycle` lists notebooks by date created, best score and date run;
  `WatcherConfig.kernel_runs` (ref → last run seen) finds re-run notebooks. Claims are compared
  with `hypothesis_pool.our_best_public_lb()` (the setup baseline only when nothing is scored).
- `kernel_board.update_board()` writes `kernel_board.json` (≤200 rows): claim (title, else the
  notebook, re-read when it re-runs, ≤25 downloads per check), `kaggle_rank`, `gap` to our best
  LB, `updated_at`, and `tried` (rounds crediting it or naming it in their code, with local and
  LB scores). `ahead_of_us()` feeds `prepare-round`; the dashboard renders it on Hypotheses.
- `kaggle monitor --kernel-minutes` (default 60) runs the notebook sensor between full cycles.
- `submission.submit_round` refuses before calling Kaggle: a round already in
  `submissions_made.jsonl`, another submission in flight (`submitting.lock`, O_EXCL, stale after
  15 min), or `daily_allowance()` at 0 — the "N submissions remaining today" of today's (UTC)
  last reply. The dashboard shows it as `submissions.daily`.

## Repo auto-run (`researchforge auto-run`)

- `autorun/unattended.py` holds the unattended machinery for a repository project. It reuses
  the Kaggle auto-run's state file, usage-limit handling, driver readiness and `AgentCLIProvider`,
  pointed at `.researchforge/`: `auto_run.json` (`RepoAutoRunConfig`), `auto_run_state.json`
  (current, last and history of launches, usage limit), `auto_run.log`, `monitor_state.json`
  and `loop_control.json` (pause).
- `monitor_cycle()` runs `launch_watch` (SIGTERM to a launch past `max_minutes` + 30),
  `auto_run` (`gates()` → `launch()`), `papers` (`research search --since <last scan>`) and
  `briefing`. The service runs one cycle per `poll_minutes` from a launcher under
  `~/.local/share/researchforge/launchers/` (label `com.researchforge.repo.<name>-<hash>`).
- The launch child (`python -m researchforge.autorun.unattended <root>`) runs approved plans
  with `start_run`/`execute_run`, then `run_autorun(max_rounds=rounds left)` inside
  `ai.service.provider_override(driver)`. With `ask`, the gate declines every plan
  (`AutorunConfig.ask_every_plan`), so the plan stays `planned` and shows as Approve plan.
  With a standing approval it runs like `autorun --yes`. `_LimitWatch` turns a driver's
  usage-limit error into `usage_limit` state, and gates hold launches until the reset.
- `set_approval("standing")` needs the typed word; the CLI also needs a TTY and refuses under
  `RESEARCHFORGE_AUTO_RUN`. The dashboard calls `POST /api/repo/auto` (settings, approval,
  start, stop, pause, resume, service, monitor), and `dashboard_state` puts `status()` under
  `repo.auto_run`.
