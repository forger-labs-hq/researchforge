# yolov5-detection — the real-ML ResearchForge demo

[`examples/simple-python`](../simple-python/README.md) proves the mechanics with
a deterministic toy benchmark. This one runs the same loop on a **real model and
real images**: YOLOv5su evaluated on COCO128 through ultralytics.

The difference that matters is honesty about noise. The toy benchmark returns
the same numbers on every machine. Here, detection accuracy is deterministic but
**inference latency is not** — and the demo is built around that fact rather
than hiding it.

## What's in here

| File | What it is |
|---|---|
| `src/config.py` | The four knobs experiments patch: `MODEL`, `CONF`, `IOU`, `IMGSZ` |
| `benchmarks/evaluate.py` | Writes `artifacts/results.json`: `map50` as the primary metric, `inference_ms`, `map50_95`, `recall`, `precision` as secondaries. `--quick` evaluates at 320px as the screening stage |
| `researchforge.example.yaml` | A ready-to-review contract — copy it to `researchforge.yaml`. Budget: `inference_ms <= 200` |
| `Dockerfile` | For `execution.mode: docker`; venv is the default |
| `requirements.txt` | ultralytics and torch |

## Before you start

- Python 3.12+ and git; run `researchforge doctor` first if anything misbehaves.
- ~2 GB free disk for torch, the checkpoint, and COCO128.
- Network access for the first run (later runs reuse everything).
- An AI key — `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, or `OPENAI_API_KEY` — only
  from step 5 onward. Steps 1–4 need no key.

---

## Step 1 — Copy it out and make it a repository

```bash
pip install researchforge
cp -r examples/yolov5-detection /tmp/yolo && cd /tmp/yolo
git init -b main && git add . && git commit -m "baseline"
```

Experiments run in git worktrees taken from a commit, so this has to be a
repository before anything can measure it.

## Step 2 — Create the project

```bash
researchforge init
researchforge project create --mode improve_repository \
  --objective "Improve YOLOv5su mAP@0.5 on COCO128 while keeping inference under 200ms"
researchforge repo scan .
```

The scan reports the language, the dependencies, the benchmark it found, and
which paths look safe to edit.

## Step 3 — Review and approve the contract

```bash
cp researchforge.example.yaml researchforge.yaml
# EDIT hard_constraints.inference_ms for YOUR machine — see "The noise" below
researchforge contract validate
researchforge contract approve            # typed approval → immutable contract
```

The contract makes `map50` the metric to maximize, `inference_ms` a hard
ceiling, `src/config.py` editable, and `benchmarks/` protected — so a variant can
change what the model does but never how it is judged.

## Step 4 — Freeze the baseline

```bash
researchforge baseline run --n-runs 3      # first run installs torch + fetches COCO128
researchforge baseline show                # mean and coefficient of variation
```

The first run is slow: it builds an isolated venv, installs torch, and downloads
the checkpoint and dataset. Three repeats rather than one because latency here
swings with whatever else the machine is doing.

If venv setup fails (torch is the usual culprit), `baseline run` reads the error,
retries with corrected setup commands, and can generate a Dockerfile and switch
the contract to Docker mode if every venv fix fails.

**Now go back and set `hard_constraints.inference_ms` from your own mean**, with
enough headroom that ordinary machine noise cannot flip a verdict, then
`researchforge contract approve` and re-run the baseline.

## Step 5 — Papers and hypotheses

```bash
export ANTHROPIC_API_KEY=sk-ant-...        # or GEMINI_API_KEY / OPENAI_API_KEY
researchforge research search              # AI writes the arXiv queries from your objective
researchforge research synthesize          # landscape + hypotheses, validated on import
researchforge hypotheses list
```

## Step 6 — Plan and run experiments

```bash
researchforge experiment plan hyp-001 --synthesize    # AI writes plan.yaml + patches
researchforge run .researchforge/experiments/plan.yaml
```

One typed approval, then each variant runs screening-first: a 320px pass, and a
full evaluation only if screening survives.

## Step 7 — Results, validation, shipping

```bash
researchforge results show run-001
researchforge dashboard --open
researchforge validate run-001 --n 5 --stdev-max 0.01
researchforge ship branch
researchforge report build
```

`--stdev-max` refuses to certify a finalist whose repeats scatter wider than the
margin it claims to have won by — which, on a latency-constrained benchmark, is
the difference between a real improvement and a lucky run.

## Or hand it the night

```bash
researchforge serve --background           # live monitor — start it BEFORE autorun
researchforge autorun --target 0.85 --max-hours 8 --yes --observe
```

`researchforge status` names the next command at any point; the full walkthrough
with explanations is in [docs/demo.md](../../docs/demo.md).

---

## Measured behavior

Measured on an Apple-silicon laptop, **CPU only** (`torch.backends.mps.is_available()`
was `False`), ultralytics 8.4.131, torch 2.13.0, Python 3.14. `map50` values will
reproduce exactly; the millisecond figures will not — see the noise note below.

| Change in `src/config.py` | map50 | inference (ms/img) | map50_95 | Outcome |
|---|---|---|---|---|
| baseline (`yolov5su`, conf 0.001, iou 0.6, imgsz 640) | 0.7395 | 165 | 0.5555 | frozen reference |
| `MODEL = "yolov5mu.pt"` | **0.7683** | 291 | 0.5842 | best accuracy, **violates the 200 ms budget → rejected** |
| `IOU = 0.7` | 0.7342 | 164 | 0.5575 | not better on the primary metric |
| `IMGSZ = 960` | 0.6913 | 333 | 0.4658 | worse *and* slower — and rejected |
| `CONF = 0.25` | 0.6140 | 158 | 0.4834 | much worse: a plausible knob that badly hurts mAP |
| screening pass (`--quick`, imgsz 320) | 0.6250 | 88 | 0.4660 | 3× faster, tracks the full result |

Four things this example demonstrates that the toy one cannot:

1. **The obvious win is the one you can't ship.** `yolov5mu` is genuinely more
   accurate — +0.029 mAP — and it is rejected anyway, because it breaks the
   200 ms budget. A framework that reported only the metric would call it the
   winner. (Two separate runs measured it at 282 ms and 291 ms; both lose.)
2. **A reasonable-sounding change can be badly wrong.** Raising `CONF` to the
   common default of 0.25 costs 0.13 mAP. mAP rewards keeping low-confidence
   detections, which is exactly why the baseline sits at 0.001.
3. **More pixels is not more accuracy.** `IMGSZ = 960` is *worse* than 640,
   because the checkpoint was trained at 640. It costs double the time for less
   accuracy — a result you only learn by measuring.
4. **Screening works.** At 320px the ordering is preserved at a third of the
   cost, so bad variants die before spending a full evaluation.

## The noise, and what to do about it

The identical baseline config measured **165 ms/image** on an idle machine and
**340 ms/image** while a large `pip install` was running — a 2× swing, same code,
same data. `map50` was 0.7395 both times, to four decimal places.

A cold first run is the worst case, and the 200 ms budget in the shipped
contract is deliberately close to the measured baseline so that the trade-off is
visible — which also means a loaded machine can push the baseline itself over
the line. So: **do not copy the 200 ms constraint from this page.** Measure your
own baseline, average it (step 4), and set the constraint from *your* mean.

## Running it under Docker instead

venv is the default. To use the included Dockerfile:

```yaml
execution:
  mode: docker
```

Docker is CPU-only here — neither Apple MPS nor CUDA is reachable from a
portable container — so its numbers are slower than venv numbers on the same
machine. That is fine, because the baseline is frozen in whichever mode you
chose and ResearchForge refuses to compare results across modes. What you must
not do is baseline in one mode and measure experiments in the other.
`researchforge generate dockerfile` writes a Dockerfile for a repo that has none.
