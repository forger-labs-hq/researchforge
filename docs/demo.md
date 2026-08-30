# Demos — see it work before you trust it

Two walkthroughs. They teach different things, and you can run either first.

| | [Demo 1 — YOLOv5](#demo-1--a-real-model-yolov5-on-coco128) | [Demo 2 — simple-python](#demo-2--the-mechanics-offline-and-deterministic) |
|---|---|---|
| What it is | A real detection model on real images | A tiny deterministic classifier |
| Why run it | See a genuinely better model get **rejected** for breaking a latency budget | See every mechanic — approval gates, isolation, ranking, shipping — in ten minutes |
| Needs an API key | For hypotheses and `autorun`; not for the baseline | No |
| Needs the network | Yes — torch, the checkpoint, COCO128 | No |
| Time to first number | ~10 min (first run installs torch) | Under a minute |
| Your numbers match this page | Accuracy yes, **timings no** — hardware-specific | Exactly |

Both drive from Claude Code, Cursor, or the CLI with an API key. The commands
below are the CLI; each step names the IDE skill that does the same thing.

---

# Demo 1 — a real model: YOLOv5 on COCO128

[`examples/yolov5-detection`](../examples/yolov5-detection/README.md) evaluates
YOLOv5su on COCO128 through ultralytics. It exists to show the two things a toy
benchmark cannot: a metric that is stable while its **latency is not**, and a
model that is genuinely more accurate and gets thrown out anyway.

**Before you start:** Python 3.12+, git, ~2 GB free for torch and the dataset,
and `researchforge doctor` reporting green. An AI key
(`ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, or `OPENAI_API_KEY`) is needed from
step 5 onward, not before.

### Step 1 — Copy the example and commit it

```bash
pip install researchforge
cp -r examples/yolov5-detection /tmp/yolo && cd /tmp/yolo
git init -b main && git add . && git commit -m "baseline"
```

**Why:** every experiment runs in a git worktree taken from a commit, so the
example has to be a repository before anything can measure it.

### Step 2 — Create the project

> **Skill:** `/researchforge-start` (Claude Code) · `@researchforge-start` (Cursor)

```bash
researchforge init
researchforge project create --mode improve_repository \
  --objective "Improve YOLOv5su mAP@0.5 on COCO128 while keeping inference under 200ms"
researchforge repo scan .
```

**What happens:** `.researchforge/` is created in `/tmp/yolo` and the scan
reports the language, dependencies, the benchmark it found
(`benchmarks/evaluate.py`), and which paths look safe to edit.

### Step 3 — Review and approve the contract

> **Skill:** `/researchforge-baseline` · `@researchforge-baseline` — read the contract
> **yourself**; approving it is a typed confirmation.

```bash
cp researchforge.example.yaml researchforge.yaml
# EDIT hard_constraints.inference_ms for YOUR machine — the file explains how
researchforge contract validate
researchforge contract approve
```

**What the contract fixes:** `map50` is the metric to maximize, `inference_ms`
is a hard ceiling, `src/config.py` is editable, and `benchmarks/` is protected —
so a variant can change what the model does but never how it is judged.

**Do not copy the 200 ms budget from this page.** It is deliberately close to
the measured baseline so the trade-off is visible, which also means a slower
machine would fail its own baseline. Step 4 gives you your own number.

### Step 4 — Freeze the baseline

```bash
researchforge baseline run --n-runs 3
```

**What happens:** the first run builds an isolated venv, installs torch, and
fetches the checkpoint and COCO128 — give it time; later runs reuse both. It
then evaluates three times and freezes the **mean** as the immutable reference.

**What you'll see:** `map50` identical across the three runs, `inference_ms`
varying. On an idle Apple-silicon laptop the reference was `map50 = 0.7395` at
~165 ms/image; the same config measured 340 ms/image during a large
`pip install`. That spread is why `--n-runs 3` is here rather than a single run.

**Then:** set `hard_constraints.inference_ms` from *your* mean with headroom,
`researchforge contract approve` again, and re-run the baseline.

### Step 5 — Find papers and turn them into hypotheses

> **Skill:** `/researchforge-papers`, then `/researchforge-landscape` and
> `/researchforge-hypotheses`

```bash
export ANTHROPIC_API_KEY=sk-ant-...   # or GEMINI_API_KEY / OPENAI_API_KEY
researchforge research search          # AI writes the arXiv queries from your objective
researchforge papers list
researchforge research synthesize      # landscape + hypotheses, validated on import
researchforge hypotheses list
```

**What happens:** arXiv is queried, results are deduplicated and ranked, and the
AI groups them into research directions and testable hypotheses. Everything it
writes is schema-validated before it is stored.

**Optional gate:** `researchforge hypotheses review` walks them one at a time.
Reviewing is optional — a hypothesis plans fine while `speculative`. *Rejecting*
is what changes behaviour: a rejected hypothesis is skipped by planning, import,
and `autorun`.

### Step 6 — Turn one hypothesis into experiments and run them

> **Skill:** `/researchforge-plan` then `/researchforge-run`

```bash
researchforge experiment plan hyp-001 --synthesize   # AI writes plan.yaml + patches
researchforge run .researchforge/experiments/plan.yaml
```

**What happens:** the AI is shown the real contents of the editable paths and
returns whole rewritten files; ResearchForge generates the diffs with git, so a
patch can never be written against a file the model never saw. `run` imports the
plan, takes **one typed approval**, then executes each variant — screening pass
first, full benchmark only if screening survives.

**What you'll see — the point of this demo:** on the measured hardware,
`MODEL = "yolov5mu.pt"` scores the best mAP anything reached, **0.7683 against
the 0.7395 baseline**, and is **rejected** at 291 ms against a 200 ms budget. A
framework that reported only the primary metric would have called it the winner.

### Step 7 — Read the results, including the losers

> **Skill:** `/researchforge-results`

```bash
researchforge results show run-001
researchforge dashboard --open
```

**What you'll see:** the ranking, the constraint violation that killed
`yolov5mu`, and the variants that simply did not help — all kept on the record.
`CONF = 0.25`, the common default, costs 0.13 mAP here; `IMGSZ = 960` is worse
*and* slower than 640 because the checkpoint was trained at 640.

### Step 8 — Validate the winner

> **Skill:** `/researchforge-validate`

```bash
researchforge validate run-001 --n 5 --stdev-max 0.01
```

**Why the flags:** on a latency-bound benchmark this is the whole question.
`--stdev-max` refuses to certify a finalist whose repeats scatter wider than the
margin it claims to have won by — the difference between a real improvement and
a lucky run.

### Step 9 — Ship it

> **Skill:** `/researchforge-ship`

```bash
researchforge ship branch                 # one commit on the frozen baseline
git log --oneline researchforge/*
researchforge report build                # full evidence chain
```

**What you get:** a clean local branch containing only the validated change,
never pushed. `researchforge ship pr` optionally opens a **draft** PR — it asks
which repository to push to and shows the exact files first.

### Or hand the whole thing to the loop

Steps 5–8 can run unattended instead:

```bash
researchforge serve --background          # live monitor — start it BEFORE autorun
researchforge autorun --target 0.85 --max-hours 8 --yes --observe
```

Watch it at the printed URL. What you should see is the loop expanding whichever
node currently leads, compounding on it when a variant wins, and asking for new
hypotheses when that branch runs dry — not a single pass through a list. Cards
badged `NO CHANGE` are the useful surprise: a variant that ran cleanly and moved
the metric by exactly nothing, which usually means the knob it turned is not
wired to the benchmark at all.

Cautious first run: `researchforge autorun --max-rounds 1 --observe` does one
round, has the AI read each run's output, and stops — enough to see the shape of
the loop, and of your bill, before committing a night to it.

The gates survive: the contract and the first batch still need your typed
approval, and hard constraints reject a variant at 3am exactly as they would
with you watching. Ctrl-C is expected — `researchforge autorun --resume`
continues with the same stall counter and time budget.

---

# Demo 2 — the mechanics, offline and deterministic

[`examples/simple-python`](../examples/simple-python/README.md) is a sentiment
classifier with a benchmark that returns the same numbers on every machine, so
**your output will match this page exactly**. No API key and no network are
needed to reach a frozen baseline; the three canonical variants are written for
you. [`examples/docker-python`](../examples/docker-python/README.md) is the same
demo under Docker isolation.

### Step 1 — Set up

```bash
pip install researchforge
cp -r examples/simple-python /tmp/demo && cd /tmp/demo
git init -b main && git add . && git commit -m "baseline"

researchforge init --claude                  # Claude Code skills → .claude/skills/
researchforge init --cursor                  # Cursor rules       → .cursor/rules/
researchforge init                           # neither: the standalone CLI path
```

Install once for the whole machine instead with `researchforge all install --user`
(both IDEs, `~/.claude/skills/` and `~/.cursor/rules/`); `researchforge all status`
says what is installed, modified, or missing.

### Step 2 — Enter an objective

> **Skill:** `/researchforge-start` · `@researchforge-start` — "Improve this
> classifier's F1 without exceeding the latency budget."

```bash
researchforge project create --mode improve_repository \
  --objective "Improve sentiment classification F1 without exceeding the latency budget"
researchforge repo scan .
```

### Step 3 — Approve the contract and freeze the baseline

> **Skill:** `/researchforge-baseline` — review the contract **yourself**.

```bash
cp researchforge.example.yaml researchforge.yaml
researchforge contract validate
researchforge contract approve        # typed approval → immutable contract
researchforge baseline run            # frozen: f1 = 0.75, p95 = 72 ms
```

This benchmark is deterministic, so one run is enough. On a real one it will not
be — `researchforge baseline run --n-runs 5` freezes the mean and records the
spread. `researchforge baseline reset --confirm` drops it if you need to measure
a new one.

**You can stop here without an API key** and still have a frozen baseline, a
scanned repo, and an audit trail. Steps 4–5 need a key or an IDE.

### Step 4 — Papers and hypotheses

> **Skill:** `/researchforge-papers`, `/researchforge-landscape`, `/researchforge-hypotheses`

```bash
export ANTHROPIC_API_KEY=sk-ant-...    # or GEMINI_API_KEY / OPENAI_API_KEY
researchforge research search           # AI-generated, domain-specific arXiv queries
researchforge research synthesize       # landscape + hypotheses, auto-imported
```

Prefer your IDE to author them? Export the context instead and let it write the
files:

```bash
researchforge research context          # writes .researchforge/synthesis/context.json
# Claude / Cursor reads context.json and writes landscape.yaml + hypotheses.yaml
researchforge research landscape --import .researchforge/synthesis/landscape.yaml
researchforge hypotheses import .researchforge/synthesis/hypotheses.yaml
```

Both paths end in the same validated records. Details in
[research mode](research-mode.md).

### Step 5 — Run competing variants

> **Skill:** `/researchforge-plan` then `/researchforge-run` — one patch per
> variant against `src/config.py`; `benchmarks/` is protected and enforced at
> import *and* again at run time.

```bash
researchforge experiment plan hyp-001 --synthesize   # AI writes plan.yaml + patches/
researchforge run .researchforge/experiments/plan.yaml
# = import + one typed approval + run (screening → full, one variant at a time)
```

Without a key, `researchforge experiment plan hyp-001` exports a context file for
an IDE to fill in. `--all --synthesize` plans every pending hypothesis at once.

**What you'll see** with the three canonical variants: `NORMALIZE = True` reaches
**f1 0.90** inside the budget; `NGRAM_EXPANSION = True` reaches f1 0.82 but
**312 ms > 200 ms** and is rejected on the hard constraint; a broken import
**fails** and is recorded as a failure rather than discarded.

### Step 6 — Failures are preserved

> **Skill:** `/researchforge-results` — losers are findings, not noise.

```bash
researchforge results show run-001    # ranking, the violation, the failure
researchforge dashboard --open        # the same story as charts (static HTML)
```

### Step 7 — Validate the winner

> **Skill:** `/researchforge-validate`

```bash
researchforge validate run-001
researchforge validate run-001 --n 5 --stdev-max 0.01   # stricter: more repeats,
                                                        # refuse a result that swings
```

### Step 8 — Clean branch and report

> **Skill:** `/researchforge-ship`

```bash
researchforge ship branch             # researchforge/<hypothesis-slug>
git log --oneline researchforge/*     # single commit on the baseline
researchforge report build            # .researchforge/reports/engineering-report.md
researchforge paper package           # optional: the research bundle
```

### Steps 4–8, unattended

```bash
researchforge serve --background      # live monitor first — watch it work
researchforge autorun --target 0.95 --max-hours 4 --yes
```

Each round picks a node in the experiment graph to expand, plans hypotheses
against **that node's** state, runs them, records the outcome in
`.researchforge/research-log.md`, and asks for new hypotheses **grounded in the
results it just measured** once the promising branches are exhausted — so round 2
pursues what worked rather than re-rolling round 1. It stops on `--target`, on
`--global-stall` rounds without improvement, on `--max-rounds`, or on
`--max-hours`. How it chooses is in
[experiment mode](experiment-mode.md#what-the-loop-counts-as-still-to-try).

---

## Lost? Two commands always answer it

```bash
researchforge status                  # where the project stands and the exact next command
researchforge paths                   # every location this project uses on disk
```

## What happened, after the fact

```bash
researchforge audit log               # every step, oldest first
researchforge audit log --kind contract_approved
researchforge audit export trail.json
```

The trail is read back out of the project's own records rather than kept as a
separate log, so it cannot disagree with what actually happened. It also flags
plans that ran without a recorded approval.

## Where to go next

- [experiment mode](experiment-mode.md) — the contract, the funnel, the search
  loop, and shipping, in depth
- [research mode](research-mode.md) — papers, landscape, hypotheses, and
  re-synthesis
- [`examples/yolov5-detection`](../examples/yolov5-detection/README.md) — the
  measured numbers behind demo 1, and the noise note in full
