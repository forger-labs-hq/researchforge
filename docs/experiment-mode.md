# Experiment mode — contract, baseline, funnel, shipping

Phases 1B–1D turn a hypothesis into a validated, shippable change. The same
division of labor as [research mode](research-mode.md) applies: **Claude
proposes; the Python engine enforces.** Nothing an author writes reaches
execution or the database without passing code-side validation, and no
prompt can bypass the path guard, the contract, or the approval gates.

## The flow

```text
researchforge contract generate            # draft researchforge.yaml from project + scan
  (edit, especially execution.full_command)
researchforge contract validate            # schema + 14 semantic rules, repeat-safe
researchforge contract approve             # typed approval -> immutable contract version
        ▼
researchforge baseline run                 # frozen baseline in a detached worktree
  (--n-runs 5 on a noisy benchmark: freezes the mean and records the spread)
        ▼
researchforge hypotheses review            # OPTIONAL: reject what isn't worth testing
        ▼
researchforge experiment plan hyp-001      # export context.json for Claude
Claude writes plan.yaml + patches/*.patch  # one unified diff per variant
  (or: experiment plan hyp-001 --synthesize — an API key does the same, and
   ResearchForge writes the diffs itself; see "How a patch gets written")
researchforge experiment import ...        # 6-layer validation; protected patches
                                           #   are recorded as rejected, never run
researchforge experiment approve plan-001  # typed approval, worst-case wall time shown
researchforge experiment run plan-001      # screening baseline -> screening -> full,
                                           #   one experiment at a time
researchforge results show run-001         # ranking, Pareto trade-offs, rejected history
researchforge validate run-001             # repeated finalist runs -> validated
  (--n 5 for more repeats; --stdev-max 0.01 refuses a result that swings)
        ▼
researchforge ship branch                  # pre-ship confirmation, then a clean local
                                           #   branch on the frozen baseline (never pushed)
researchforge report build                 # engineering report (spec §16)
researchforge ship pr                      # OPT-IN: push that one branch + DRAFT PR
researchforge paper package                # research bundle (BibTeX, outline, data)
```

`researchforge status` shows the next step at every stage. All commands
support `--json`.

## How a patch gets written

A variant is a unified diff, but nobody is asked to write one blind. Whichever
path authors the plan, the author is first shown the source it is editing.

`experiment plan --synthesize` and every round of `autorun` export the contract's
`editable_paths` **in full** — real file contents, read out of git rather than
the working tree, prioritized by the hypothesis's own words when the budget
cannot fit everything, with whatever does not fit still listed by name. The AI
returns the **complete new content** of each file it wants to change, and
ResearchForge computes the diff itself with git.

That split matters: a model asked to produce a diff has to invent line numbers
and context for a file it cannot see, which is how "corrupt patch" and
"No such file or directory" happen. A model asked to rewrite a file it *has*
been shown does not, and git cannot generate a diff that fails to apply to the
state it was generated from.

When a plan builds on a parent, the files are shown **as that parent leaves
them** — the ancestor patches are replayed in a scratch worktree first, and the
prompt names which experiments are already baked into what the model is reading.
Otherwise the model would edit the baseline's version of a line its patch will
never meet.

## Or let the loop drive it

The flow above is the manual path, and it is worth walking once to see what
each gate is for. After that, `autorun` does the same thing repeatedly without
you: it picks a place in the graph to expand, plans hypotheses there, runs them,
records what happened, and asks for new hypotheses when it runs out of moves —
round after round until it stalls, reaches the target, or runs out of hours.

```bash
researchforge autorun --target 0.85 --max-hours 8 --yes
researchforge autorun --resume        # after a Ctrl-C, same stall counter and budget
```

Neither planning nor running is silent. Each round names the node it is
expanding and why, keeps a clock on the AI call, and prints a one-line verdict
per experiment as it finishes, so a long round is visibly working rather than
apparently hung.

What it does *not* do is remove the gates. The contract and the first batch
still need your typed approval; `--yes` is what skips that first-batch prompt,
and it is the only prompt a loop has — every later round runs unattended
regardless. Hard constraints are enforced at 3am exactly as they are by hand, and a
rejected experiment is still recorded rather than quietly dropped. See
[the autorun section in the README](../README.md#the-autonomous-loop) for the
flags.

## Experiments form a DAG, not a chain

An experiment declares `parent_experiment_ids`. With no parents it is measured
against the frozen baseline; with one it builds on that ancestor; with several
it composes all of their patches. That means:

- **Compounding.** Round 3 can build on round 1's winner rather than only on
  the most recent result.
- **Merging.** Two independent winners can be combined into one experiment to
  see whether their gains actually add up. When the patches conflict
  mechanically, the AI is asked for a single merged patch instead — and that
  patch goes through the same import validation as any other.
- **Backtracking.** `autorun --compound` selects which node to expand with
  UCB1, so a stalled leader does not trap the loop: `--explore 0` always
  expands the current best, while higher values revisit under-explored
  branches — including ones that never beat the baseline, which the default
  holds back (see below).

Patches are composed in topological order, so an ancestor's change always
applies before a descendant's. Conflicts between a parent set are detected at
import, before anything runs. The graph is rendered by `researchforge
dashboard` (static HTML) and by the live monitor — winning paths highlighted,
rejected variants shown in context.

One asymmetry is worth knowing: **patches compose down a lineage, `env_overrides`
do not.** A run injects only that experiment's own overrides, so a variant whose
change is expressed as an env override is not inherited by anything built on it.
Where a change needs to compound, put it in the patch.

## What the loop counts as "still to try"

A hypothesis is not spent the moment it is planned once. What is spent is a
hypothesis **at a node** — the same idea applied on top of a different ancestor
is a different experiment with a different measurement, which is exactly the
move a person makes after a result comes in. Two rules keep that from
degenerating:

1. A (hypothesis, node) pair is tried once; re-planning it would reproduce an
   experiment already in the graph, patch for patch.
2. A hypothesis already applied anywhere in a node's lineage is not applied on
   top of itself.

Within a round, the available hypotheses are ranked — ideas that improved the
metric somewhere first, then never-tried ones, then what is left ordered by how
little it managed and how often it landed on its parent's exact value. How many
of them a round commits to depends on the clock: with `--max-hours` the round
takes as many as the remaining time can pay for, and without one it takes a
single move, measures, and re-selects.

Re-synthesis is driven by **exhaustion, not by the round number**. When the
winning branches have nothing left to try, the loop asks for new hypotheses; a
second invocation of `autorun` therefore starts working again instead of exiting
with "no hypotheses left to try". Only when new ideas cannot be generated either
does it fall back to the branches that never won — an ablation is worth more
than stopping, but it is the last resort rather than the next move, because
running an idea without the gains already banked is a step backwards. Passing
`--explore` above 0 lifts that floor deliberately: asking for under-explored
branches is asking for exactly the nodes it holds back.

`--observe` adds one more thing to each round: the AI reads the run's own
`stdout.log` and writes a short observation, stored on the experiment and in
the research log. It is how a run that "failed" for an interesting reason
leaves a trace a human can read later.

## Every decision is recoverable after the fact

```bash
researchforge audit log --last 20        # oldest first; --kind filters
researchforge audit export trail.json
```

The trail is **derived**, not a second log file: the events are read back out
of the records the workflow already wrote — contract approvals, baseline
measurements, plan approvals, benchmark executions, experiment decisions,
hypothesis reviews, searches, deliverables. There is nothing to keep in sync
and nothing to tamper with separately. `audit log` also reports plans that ran
without a recorded approval, which is the part an auditor asks about.

**Monitor live:** `researchforge serve --background` (after
`pip install "researchforge[serve]"`) starts a local, **read-only** web
monitor at `http://127.0.0.1:9000` (a free port is picked automatically if
that one is busy) — overview, research state, an experiments page that
refreshes as each funnel stage completes, and the live dashboard charts.
`experiment run`/`start` auto-start it on a TTY and print the URL; manage
it with `serve --status` / `serve --stop`. It binds 127.0.0.1 only by
default and opens the database in read-only mode, so watching can never
interfere with a run.

**Run lifecycle at a glance:**

| I want to… | Command |
|---|---|
| import + approve + run in one step | `experiment start plan.yaml` (one typed approval) |
| stop a running batch | Ctrl-C — safe; worktrees are isolated |
| continue an interrupted run | `experiment resume run-XXX` |
| discard an interrupted run | `experiment abandon run-XXX` (finished results are kept) |
| cancel a not-yet-run plan | `experiment cancel plan-XXX` |
| start the next batch | `experiment plan <hyp-id>` → `experiment start …` |

## The contract is the boundary

`researchforge.yaml` freezes the evaluation: primary metric + direction,
hard constraints, editable vs protected paths, commands, limits, network
mode, and the shipping flags. Approval hashes the file; any later edit is
detected as drift and execution refuses until re-approval creates the next
immutable version. Execution always uses the stored snapshot, never the
disk file.

## Isolation and honesty guarantees

- Every baseline/experiment/validation attempt runs in its own **detached
  git worktree** — the user's checkout and branches are never touched.
- Patches are validated with `git apply --check`; changed files are
  extracted by git, never trusted from the author; the **path guard**
  rejects protected/non-editable changes at import *and* re-checks after
  apply at run time (plus symlink refusal), before any command runs.
- Screening results are only compared to a **screening baseline** run with
  the same command; full/validation results compare to the frozen full
  baseline (same benchmark, same environment mode — enforced).
- `validated` structurally requires the full run plus repeated validation
  attempts: a one-off result can never be called validated. `--stdev-max`
  goes further and refuses a finalist whose repeats spread wider than the
  given standard deviation, however good its average looks — without it, a
  benchmark that swings can turn luck into a "validated" claim.
- `baseline reset` refuses to drop a frozen baseline that experiments have
  already been measured against unless you pass `--force`, because their
  recorded improvements are stated against it. The experiment records are
  never rewritten: they keep the baseline value they were measured against.
- Rejected and failed experiments are first-class records; they appear in
  `results show`, the engineering report, and the research package.
- Every percentage on the graph is measured against the **frozen baseline**,
  never against a parent, which is what makes two cards comparable. The cost is
  that a child inherits its parent's gain, so a node that measured exactly what
  it was built on is badged `NO CHANGE`: the number is real, the experiment's own
  contribution was nothing the benchmark could see.

## Shipping safety

- `ship branch` requires `shipping.allow_branch_creation` in the approved
  contract and a typed confirmation; it re-runs the full benchmark once
  (pre-ship confirmation) and refuses to ship on failure, constraint
  violation, or unconfirmed improvement. The branch is one clean commit on
  the frozen baseline (post-conditions asserted in code) and is **local
  only** — nothing is pushed.
- `ship pr` is opt-in twice: `shipping.allow_draft_pr` in the contract AND
  a typed `push` confirmation. It pushes exactly one ref and always opens
  a **draft** PR whose body is generated from recorded data.
- Tests: the contract's `test_command` runs before every evaluation; the
  commit and PR state explicitly that no new tests were authored (test
  authoring is a Claude-assisted step in Phase 1E).
