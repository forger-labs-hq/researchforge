---
name: researchforge-autorun
description: Drive the autonomous research loop round after round — ask the engine which node to expand, plan there, run, read the result, repeat. Use when the user wants to keep improving a repository over many rounds, or wants autorun without an AI API key.
---

# Drive the research loop

`researchforge autorun` runs this loop by itself when an API key is configured.
Without one, **you** are the intelligence and the engine is still the lab: it
chooses where to search, validates what you write, runs the benchmark, and
records the result. Your job is the one part it cannot do without a provider —
writing the plan and the patches.

Never guess at the next move. The engine's search knows things a results table
does not show: which ideas were already tried at which node, what is in each
node's ancestry, and how much budget is left.

## 1. Ask where to go next

```bash
researchforge autorun --dry-run --json
```

This spends nothing — no AI call, no experiment runs. It returns the node the
loop would expand, the hypotheses it would try there in ranked order, and the
exact command to take that step:

- `recommended_node` — the experiment to build on, or `baseline`;
- `next_command` — run this next;
- `merge_opportunities` — if set, THIS IS THE HIGHEST PRIORITY MOVE;
- `dag_state` — total, improved, baseline, current best.

(Kaggle competitions use the `/overnight` skill and `researchforge kaggle prepare-round`,
not this loop.)

## 2. Plan at that node

Run the `command` from step 1. With a parent it looks like:

```bash
researchforge experiment plan hyp-002 --parent exp-008 --json
```

The exported context's `repository` section shows the files **as `exp-008`
leaves them**. Write the change against those contents.

**To combine two independent improvements (merge)** — when `merge_opportunities` is set:

```bash
researchforge experiment plan --merge exp-001,exp-003 --json
```

This writes and imports the merge plan itself (both patches composed, no AI) and prints its
`plan_id`; the AI is asked to author a combined patch only if the two edit the same lines.
Go straight to step 3's approve and run.

## 3. Import, approve, run

```bash
researchforge experiment import .researchforge/experiments/plan.yaml --json
researchforge experiment approve <plan-id>     # the user's decision
researchforge experiment run <plan-id> --json
```

Approval belongs to the user. Show them the plan (`researchforge experiment show <plan-id> --json`)
and the worst-case wall time, and let them run the approve command themselves unless they
explicitly approved this plan in this conversation — only then pass `--yes`.

**When the local benchmark is only a signal** (it cannot measure what the change targets),
the unattended loop can record every result instead of rejecting on the local metric:
`researchforge autorun --no-metric-gate`. Hard constraints still reject.

## 4. Read the result, then go back to step 1

```bash
researchforge results show --json
```

In unattended `autorun`, every result is also pushed to ntfy (after
`researchforge notify setup`). Did anything improve? Does `merge_opportunities` list a pair?

Then return to step 1.

## 5. When the graph is exhausted

`needs_resynthesis: true` means no hypothesis has anywhere left to apply:

```bash
researchforge research synthesize --from-results
```

Then return to step 1.

## Available flags

```
autorun --dry-run           Preview next round without executing. Use this every time.
autorun --no-metric-gate    Record every result; reject only on hard constraints
autorun --merge             Propose combining two independent winners each round
autorun --yes               Skip the round-1 approval gate (overnight/unattended)
autorun --max-hours N       Wall-clock budget
autorun --max-rounds N      Hard cap on rounds
autorun --explore N         UCB1 constant for node selection (0 = always best)
```

## Unattended (no chat open)

When the person wants the loop to keep running without this chat, point them to
`researchforge auto-run settings --on --driver claude|cursor|byok`, then
`researchforge auto-run service`.
Each plan it writes waits for their **Approve plan**. A standing approval is something they
type themselves with `researchforge auto-run approval standing`. Never run that for them.

## Stopping

Stop when: objective target is reached, several rounds produce no improvement,
only retreat moves remain. Say which happened, show the best experiment, and offer
`/researchforge-validate` then `/researchforge-ship` — both the user's decision.

## Rules

- **Never choose the next node yourself** when `--dry-run` can tell you.
- Approvals belong to the user — never pass `--yes` without explicit approval.
- Ground every summary in stored data — never invent metrics.
- The Python engine is the boundary — never work around a validation error.
