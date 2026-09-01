# Changelog

## 0.3.0 — What it costs, and running it from your editor (2026-09-01)

Two gaps 0.2.0 left open: the loop needed an API key even when you were already
paying for one inside your editor, and nothing anywhere said what a run cost.

### The loop without an API key

- **`autorun --dry-run`** answers "where would you go next?" — which node the
  frontier would expand and which hypotheses it ranked there — without planning,
  running, or calling a provider. An agent driving the loop by hand gets
  ResearchForge's search instead of re-deriving one by eye from a results table.
  `--json` for programmatic use.
- **`experiment plan --parent exp-NNN`** compounds onto a measured winner from
  the handshake path: the exported repository files are the ones that experiment
  leaves behind, so the change is written against what it already won. Unknown
  ids fail before any AI call or file write.
- **Provider resolution is lazy** in `plan_all_hypotheses`. A round with nothing
  to plan no longer demands a key it was never going to use, which is what
  blocked keyless execution-only rounds.
- **`researchforge-autorun`** skill and Cursor rule drive the loop from the
  editor: dry-run for the next move, plan and import it, run it, repeat.
- **`init --cursor` installs the workflow rules**, not only the gateway —
  matching what `--claude` has always done.

### What a run costs

- **Time economics** on the dashboard: compute split by stage and *by how each
  experiment ended*, so what the failures cost is visible rather than only what
  the winner gained; how many full benchmarks never ran because screening or the
  stall rule stopped them first; and which changes a hard limit caught.
- **Work avoided is shown as its arithmetic** — skipped runs times this
  project's own measured average full benchmark, with both halves on screen. A
  project with no completed benchmark to average reports no figure rather than
  zero.
- **No counterfactual claims.** There is deliberately no "this would have taken
  a human N hours": that number cannot be checked, and one unfalsifiable claim
  discredits the measured ones next to it. A test fails the build if the phrase
  reappears in the dashboard.
- **The hub shows compute and token spend per project**, so you can see which
  project has been running rather than only thinking; the monitor's overview tab
  carries the same summary, and the full breakdown stays on its dashboard tab.

### Token accounting

- **Every provider now reports usage.** Anthropic, Gemini and OpenAI all
  returned token counts that were being discarded on the line that extracted the
  text. Providers file them into a context-local ledger rather than changing
  `generate()`'s signature, so callers that want the numbers opt in and the rest
  are unaffected. Usage is recorded even when a call raises — a request that
  failed after the prompt was read still cost money.
- **Calls are attributed to what asked for them** — planning, synthesis,
  observation, evaluation, merge, queries, dockerfile — and stored per project.
- **Prices live in `.researchforge/config.json`** under `model_prices`, dollars
  per million tokens matched by longest model-name prefix, so `gpt-4o-mini` is
  not billed at `gpt-4o`'s rate and a dated release is priced by its family.
  Defaults ship for common models and are overridable for negotiated rates.
- **A model with no rate is named, not billed at zero.** The dashboard says the
  dollar figure is a floor and lists the unpriced models; the hub omits the
  dollar pill entirely. A zero meaning "unknown" reads as "free".
- **Benchmark compute converts to dollars only if you say what an hour costs**
  (`local_compute_usd_per_hour`), since an owned laptop has no marginal hourly
  cost and a rented GPU has an invoice.
- **IDE-driven planning is sized, not metered.** When Claude Code or Cursor
  makes the call, ResearchForge never sees the response — so it records the size
  of the exchange it *can* see: context handed over, plan and patches returned.
  These are stored flagged as estimates, kept out of the metered total, never
  priced, and rendered with a tilde everywhere they appear.

**Compatibility:** schema 8 adds `ai_calls` and its `estimated` column;
migration is automatic on the first write. Projects that have not been written
to since upgrading report no model calls, which is accurate — usage is captured
as calls happen and cannot be reconstructed for runs that already finished.

## 0.2.0 — Phase 2: standalone AI and the autonomous loop (2026-08-30)

0.1.0 needed a human at every step and Claude Code to think. This release
removes both: ResearchForge runs from a plain API key, and `autorun` searches
for hours unattended without giving up the gates, constraints, or records that
make a result worth trusting.

### Standalone — no IDE required

- **Built-in AI providers**: Anthropic, Google Gemini and OpenAI ship as core
  dependencies with auto-detection from whichever key is set, `--provider` to
  choose explicitly, `RESEARCHFORGE_LLM` to override the model, and Ollama for
  local inference. Every step that previously required an IDE now has a CLI
  path; the Claude Code and Cursor integrations remain, as one option of three.
- **`research synthesize` / `hypotheses generate`**: landscape and hypotheses
  generated directly from retrieved papers, validated against the same schemas
  as the IDE path before import.
- **`generate eval-script`**: writes `benchmarks/evaluate.py` and a tunable
  `src/config.py` for a repo that has no benchmark, and commits them — an
  uncommitted eval script is invisible to the worktrees experiments run in.
  `--existing` adopts a script you already have.
- **`generate dockerfile`**: heuristic by default (no key needed), AI-tailored
  with `--provider`.
- **`experiment plan --synthesize`**: plan and patches authored in one command.

### `researchforge autorun` — the loop

- **Unattended rounds**: plan, run, record, re-synthesize from measured results,
  repeat. Stopping conditions are per-plan `--stall`, loop-wide
  `--global-stall`, `--max-rounds`, `--max-hours` and `--target`; `--resume`
  continues an interrupted loop with the same counters and budget.
- **Human gates survive autonomy**: the contract and the first batch are still
  typed by a human. `--yes` skips only the per-round confirmations after that,
  and hard constraints are enforced at 3am exactly as they are by hand.
- **Experiments became a DAG**: `parent_experiment_ids` replaces the single
  parent, so a round can build on any earlier winner and two independent
  winners can be merged into one experiment to test whether their gains add.
  Parent patches compose in topological order; conflicts are caught at import.
- **Graph search, not a queue drain**: the unit of work is a *(hypothesis,
  node)* pair — the same idea on a different ancestor is a different
  measurement — bounded by trying each pair once and never applying a
  hypothesis on top of itself. Re-synthesis fires on exhaustion rather than
  round number, so a second `autorun` on an explored graph starts working
  instead of exiting.
- **Node selection is UCB1** (`--explore`): 0 always expands the current best,
  higher values revisit under-explored branches so a stalled leader cannot trap
  the loop.
- **`--observe`**: the AI reads each run's own `stdout.log` and writes a short
  observation, so a failure that failed for an interesting reason leaves
  something readable rather than only a status code.
- **Results-grounded re-synthesis**: new hypotheses account for what improved,
  what violated a constraint and what failed. They are added rather than
  replacing the set, and near-duplicates of tested ideas are dropped.

### Patches are written from the source

- The contract's `editable_paths` are exported **in full** — real contents read
  from git, prioritized by the hypothesis's wording when the budget cannot fit
  everything, with the remainder listed by name.
- The AI returns the **complete new content** of each file it changes and
  ResearchForge computes the diff with git, which cannot produce a diff that
  fails to apply to the state it came from. This replaced a long tail of
  `corrupt patch` and `No such file or directory` plan rejections.
- A compounding plan sees its **parent's** files, not the baseline's: the
  lineage is replayed in a scratch worktree and the prompt names which
  experiments are already baked into what the model is reading.

### Honesty and legibility

- **`NO CHANGE` badge**: every percentage is measured against the frozen
  baseline, which is what makes two cards comparable — but it means a child that
  changed nothing still wears its parent's gain. A node measuring exactly what
  its ancestors measured is now badged, with a tooltip saying the number is
  inherited.
- **Derived audit trail**: `audit log` / `audit export` reconstruct project
  history from the records the workflow already writes, and report *gate
  findings* — plans that executed without a recorded approval. There is
  deliberately no separate audit log file; a second write path can drift from
  what it describes.
- **Readable experiment graph**: sibling edges share a trunk, parents are
  centred on their children, and layers too tall to stack wrap into sub-columns
  with edges routed over a rail, so wide runs grow sideways rather than downward.
- **Nothing runs silently**: planning shows its phase and a clock and then
  states the plan's shape; execution reports worktree setup, patch application,
  screening and the full benchmark, and every finished experiment leaves a
  permanent one-line verdict. Rounds narrate which node they are expanding and
  why.

### Shipping

- **`ship pr` asks where the pull request should go** and pins that repository
  for every `gh` call instead of relying on `gh`'s base-repo inference, which
  can silently prefer `upstream` over your own fork. `--remote` and `--repo-url`
  choose non-interactively.
- **Replay by default**: the winning change is replayed as a single commit on
  the target's base branch, so the PR carries the change rather than the
  baseline history it was measured on. `--as-measured` pushes the shipped commit
  instead, and diverging files are reported before anything is pushed.
- **Opening a PR on a repository you cannot push to always requires a typed
  confirmation**, which `--yes` cannot bypass.

### Setup that recovers itself

- `baseline run` diagnoses setup failures instead of reporting them: it reads
  stderr and retries against a rule set covering flat-layout `pyproject.toml`,
  stale pip, missing compilers, CUDA mismatches, resolver conflicts, network and
  disk failures, and a benchmark script missing from git.
- When every venv fix fails and Docker is available, it can generate a
  Dockerfile, switch the contract to Docker mode, re-approve, and retry there.

### Also

- **`baseline run --n-runs N`** freezes the *mean* of N runs and records the
  spread, so a noisy benchmark's improvements are not measured against one lucky
  run. **`baseline reset`** refuses without `--force` once experiments have been
  measured against the baseline, and never rewrites their records.
- **`validate --n` / `--stdev-max`** refuse a finalist whose repeats spread too
  wide, however good the average.
- **`hypotheses review/approve/reject`** as an optional gate — rejection is the
  half with teeth, and the refusal names the command that undoes it.
- **`papers export` / `import`** move a literature set between projects without
  re-querying arXiv; **`research search --since`** filters inside the query so
  the candidate budget is not spent on papers that get discarded.
- **`env_overrides`** in a plan change configuration without a patch file. Note
  the asymmetry, which silently breaks compounding if you don't know it: patches
  compose down a lineage, `env_overrides` do not.
- **`researchforge run <plan.yaml>`** and **`baseline status`** aliases;
  `--help` groups commands into named panels.
- **`examples/yolov5-detection`**: a real ML target — YOLOv5su on COCO128, mAP as
  the objective, per-image latency as a hard constraint — rather than a
  deterministic toy.

**Compatibility:** existing single-parent experiment records migrate
automatically. Contracts from 0.1.0 remain valid; `stall`, `mode: auto` and
`target_value` are optional additions.

## 0.1.0 — Phase 1 open-source beta (2026-07-24)

The complete local pipeline, Claude-first.

- **Research intelligence**: arXiv discovery (dedup + deterministic
  ranking), the Claude↔CLI synthesis handshake (landscape + hypotheses with
  graded evidence), citation-backed research report.
- **Experiment contract**: `researchforge.yaml` wizard, 14 semantic
  validation rules, typed approval into immutable versions with drift
  detection.
- **Isolated execution**: detached git worktrees per attempt, Docker
  (locked-down defaults) and `.venv` runners, path guard with run-time
  re-check, process-group timeouts, secrets redaction.
- **Experiment funnel**: Claude-authored patch variants through screening →
  full benchmark → repeated validation; hard constraints; Pareto ranking
  with honesty caveats; rejected/failed experiments preserved.
- **Shipping**: clean branch reconstructed from the frozen baseline
  (pre-ship re-validation, post-conditions asserted), opt-in draft PR via
  gh, engineering report, research package (BibTeX, outline,
  reproducibility bundle).
- **Claude Code experience**: 12 installable project skills
  (`researchforge init --claude`), manifest-based installer that never
  clobbers user edits.
- **Beta tooling**: tested launch-demo examples, opt-in local-only
  analytics, security notes.
