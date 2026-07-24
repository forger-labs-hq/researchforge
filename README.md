# ResearchForge

*From papers to proof.*

ResearchForge turns a research question — or an "improve my repository"
goal — into evidence. It finds relevant papers, helps Claude form testable
hypotheses, benchmarks competing implementations against a frozen baseline
in isolated local workspaces, and delivers the strongest supported result
as a clean branch, an engineering report, or a research package.

Inspired by Andrej Karpathy's
[autoresearch](https://github.com/karpathy/autoresearch), where an agent
autonomously runs training experiments against a fixed benchmark overnight —
ResearchForge generalizes that loop to any repository with a measurable
benchmark, and grounds it in literature: papers → hypotheses → controlled
experiments → a validated, reproducible result.

## How it works

Three parties with strict roles:

| Who | Does what |
|---|---|
| **You** | set the objective; approve the benchmark contract, each experiment plan, and anything that ships |
| **Claude** (in Claude Code) | reads the papers, writes the research landscape, hypotheses, and experiment patches; explains results |
| **The Python engine** | everything that must be trustworthy: search, validation, isolated execution, ranking, shipping — no prompt can bypass it |

No model API key is required — Claude Code *is* the model. Every artifact
Claude writes is schema-validated before it is stored, every experiment
runs in a detached git worktree (your checkout is never touched), and
"validated" is only ever earned by repeated benchmark runs.

## Get started (two minutes)

Everything runs on your machine — nothing is uploaded anywhere.

```bash
# 1. Install ResearchForge (Python 3.12+):
pip install "researchforge[serve]"       # or: pipx install "researchforge[serve]"

# 2. Make ResearchForge available in EVERY Claude Code session (recommended):
researchforge claude install --user      # skills -> ~/.claude/skills/
```

(Skip step 2 if you like — the first time you run `researchforge` in a
terminal it offers this install with one keystroke, and never asks again.)

That's the whole setup. Now open **any** Claude Code session and say what
you want:

> `/researchforge-start` — *"Can uncertainty-aware routing outperform fixed
> routing?"* — or — *"Improve this repo's F1 without hurting latency —
> work in ~/projects/my-repo."*

Claude asks **where the project should live** (any folder — a cloned repo
to improve, or an empty directory for pure research), initializes it there,
and walks the whole journey: it runs the CLI, shows you what it found, and
**asks before anything is approved, executed, or shipped**. If you ever
wonder where things stand, `researchforge status` names the exact next
step — and the **hub** at http://127.0.0.1:9000 shows every project, where
it lives, and what's happening, all the time.

Prefer per-project skills instead of the global install? `cd` into the
folder and run `researchforge init --claude`, then start Claude Code **in
that folder** (desktop app: open the folder as the session's project — a
home-screen session without a folder can't see project skills).

Everything is **directory-scoped**: the database, worktrees, artifacts, and
dashboard live under the project folder. Run commands from any subfolder
(they walk up to the project root like git), point at another project with
`researchforge -C /path/to/project <command>`, and print the full location
map with `researchforge paths`.

**Working on the ResearchForge source itself?**

```bash
git clone https://github.com/forger-labs-hq/researchforge
cd researchforge
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## See it work first

[docs/demo.md](docs/demo.md) is a ten-minute, fully offline walkthrough
against [examples/simple-python](examples/simple-python/README.md) — a
deterministic benchmark where one variant genuinely improves F1, one is
rejected for violating the latency budget, and one fails (all three stay in
the record). [examples/docker-python](examples/docker-python/README.md) is
the same demo under Docker isolation.

## Watch your experiments (dashboard + live monitor)

Two ways to *see* how experiments perform against the baseline:

```bash
researchforge dashboard --open       # one self-contained HTML snapshot
```

A single static file (no server, no JS libraries, nothing leaves your
machine) with an autoresearch-style **progress chart** — every experiment as
a chronological dot, kept improvements annotated, a running-best step
line — plus per-experiment bars vs the baseline, the trade-off scatter with
the hard-constraint line, the funnel with drop-offs, and validation spread.

The static dashboard also opens with **summary stats** (best score and its
delta vs baseline, experiments kept · discarded · errored, the Pareto
frontier) and the **experiment tree** — a graph from the baseline through
every branch of experiments (see "experiments on experiments" below).

```bash
pip install "researchforge[serve]"
researchforge serve --background     # live monitor for THIS project (URL printed)
```

A local web monitor that follows runs **as they happen**: overview with the
next action, collapsible research sessions with the full recorded detail
(directions, evidence, limitations, underexplored aspects), per-run
execution timelines with work locations on disk, the experiment tree with
**click-through drill-down pages** for every experiment (lineage, decision,
executions, artifacts on disk), the live chart dashboard, and a JSON API
(`/api/state`). Once the extra is installed, `experiment run`/`start`
auto-start it and print the URL. Manage it with `researchforge serve
--status` / `--stop`. The server opens the database **read-only** and binds
127.0.0.1 only by default — watching can never interfere with a run.

### The hub — every project, one dashboard

```bash
researchforge hub --background       # http://127.0.0.1:9000 — all projects
```

Projects live in whatever folders you initialize them in, and it is easy to
forget which. The **hub** is one machine-wide page listing every project
with its **folder location**, status, and live activity; click through to
any project's full monitor (sessions, runs, experiment tree, drill-downs).
Every `researchforge init` registers the project, so new projects appear
automatically — and once the `serve` extra is installed, **any researchforge
command quietly ensures the hub is running**, so `http://127.0.0.1:9000`
is simply always there (set `RESEARCHFORGE_NO_HUB=1` to opt out). Commands
run in a subfolder of a project also walk up to find it (like `git`) and
print `Using project at <root>` so you always know which project you're in.

## Journey A — research an idea

You have a question; ResearchForge grounds it in literature. From Claude
Code, `/researchforge-start` with your question does all of this; the CLI
equivalent (where **you** write the synthesis files against exported
schemas) is:

```bash
researchforge project create --mode explore_research_idea --objective "..."
researchforge research search        # arXiv: fetch -> dedup -> rank -> store
researchforge research context       # exports context.json with the schemas
# Claude (or you) writes landscape.yaml + hypotheses.yaml — validated on import:
researchforge research landscape --import .researchforge/synthesis/landscape.yaml
researchforge hypotheses import .researchforge/synthesis/hypotheses.yaml
researchforge report build           # citation-backed report
researchforge paper package          # optional: BibTeX, outline, evidence matrix
```

**You end up with:** a research landscape (directions + graded evidence:
published claim vs interpretation vs speculation), testable hypotheses, a
citation-backed Markdown report, and optionally a full research bundle.
Details: [docs/research-mode.md](docs/research-mode.md)

## Journey B — improve a repository

Everything in Journey A, then benchmarked experiments on your code. Every
consequential step is **your** typed approval — Claude cannot approve, run,
or ship anything by itself.

**1. Define and freeze the evaluation:**

```bash
researchforge project create --mode improve_repository --objective "..."
researchforge repo scan
researchforge contract generate      # edit researchforge.yaml, then:
researchforge contract approve       # YOUR approval -> immutable contract
researchforge baseline run           # frozen reference measurement
```

**2. Run experiments** (Claude writes `plan.yaml` + one patch per variant
after `researchforge experiment plan hyp-001`; manually you write them
against the exported schema):

```bash
researchforge experiment start .researchforge/experiments/plan.yaml
# = import + ONE typed approval (shows worst-case wall time) + run
```

**Experiments on experiments (branching):** a plan entry can declare
`parent: exp-001` (or the key of another entry in the same plan) to build
**on top of** a previous experiment instead of the baseline — refine a
winner, or rescue a rejected idea by combining it with something else. The
engine validates the whole chain at import (a child that doesn't apply on
its parent's state is refused), executes it root-first in isolation, and
still measures **honestly against the frozen baseline**. Ship a branched
winner and the chain is composed into one clean commit. The dashboard
draws the whole tree, like Karpathy's autoresearch UI.

**3. Inspect and validate:**

```bash
researchforge results show run-001   # ranking, trade-offs, rejections
researchforge validate run-001       # repeated runs earn "validated"
```

**4. Accept the result → what ships (and how the PR happens):**

```bash
researchforge ship branch    # clean LOCAL branch on the frozen baseline —
                             # one commit, nothing pushed, inspect with git
researchforge report build   # engineering report: the full evidence chain
researchforge ship pr        # OPT-IN: push + open a DRAFT PR on YOUR repo
```

`ship pr` is how the pull request gets created, and it only works when all
three gates open: `shipping.allow_draft_pr: true` in the **approved**
contract, the **`gh` CLI authenticated** to your remote, and a typed
`push` confirmation from you. It pushes exactly one branch and opens a
**draft** PR for human review — nothing is ever pushed without those
gates. Prefer full control? Stop after `ship branch` and push/PR yourself
with plain git. Details: [docs/experiment-mode.md](docs/experiment-mode.md)

**Open-source repositories (no push access):** cloned someone else's repo?
`ship pr` detects that you can't push to origin and switches to the fork
workflow — behind a typed `fork` confirmation that spells out exactly what
happens: a **public fork** is created (or reused) under your GitHub
account, one branch is pushed *to the fork*, and a **draft** PR is opened
on the upstream repository. One caveat to review first: if you committed a
benchmark locally to establish the baseline, that commit is part of your
branch's history and **will appear in the PR** — check `git log` (the CLI
warns you when the branch carries extra commits). And read the project's
CONTRIBUTING.md before opening PRs on repos you don't maintain — a draft
PR with a reproducible, validated improvement is a good contribution; ten
of them are spam.

### Managing experiment runs

| I want to… | Command |
|---|---|
| start a batch (one command) | `researchforge experiment start plan.yaml` |
| watch it live | `researchforge serve --background` (URL is printed) |
| see ALL projects + their folders | `researchforge hub --background` → http://127.0.0.1:9000 |
| stop a running batch | **Ctrl-C** — always safe (isolated worktrees) |
| continue an interrupted run | `researchforge experiment resume run-001` |
| discard an interrupted run | `researchforge experiment abandon run-001` |
| run another batch | `researchforge experiment plan hyp-002` → start again |
| build on a previous experiment | `parent: exp-001` in the next plan entry |
| see what's next, always | `researchforge status` |
| see where everything lives | `researchforge paths` |
| reset the whole project | `rm -rf .researchforge researchforge.yaml` |

Everything is local; nothing outside your repository is ever created. To
redefine just the objective on existing data:
`researchforge project create --force-update`.
[docs/claude-mode.md](docs/claude-mode.md) explains exactly what the Claude
skills do and cannot do.

## Supported repositories (beta)

The improve-repository journey currently expects:

- a **git** repository with **user-owned or trusted code** (isolation is
  local, not a hostile-code sandbox — [docs/security.md](docs/security.md));
- a **Python 3.11+** single project or single target service;
- an existing **Dockerfile** or simple Python dependency metadata
  (`requirements.txt` / `pyproject.toml`);
- a **machine-readable benchmark** (a command that writes JSON metrics)
  with **bounded runtime**;
- no production infrastructure required.

The explore-research-idea journey works anywhere. Repositories outside this
matrix are reported honestly by `researchforge repo scan` rather than
half-supported.

## Beta feedback

This is a narrow-but-complete beta — reports shape what gets built next.
Use the issue templates ([bug](.github/ISSUE_TEMPLATE/bug_report.yml),
[setup failure](.github/ISSUE_TEMPLATE/setup_failure.yml),
[beta feedback](.github/ISSUE_TEMPLATE/beta_feedback.yml)). Optionally,
`researchforge analytics enable` records **local-only** coarse events —
nothing is transmitted — and `researchforge analytics show` computes the
beta metrics you can choose to include in a report.

## More documentation

- [docs/demo.md](docs/demo.md) — the launch demo, step by step
- [docs/claude-mode.md](docs/claude-mode.md) — working from Claude Code
- [docs/research-mode.md](docs/research-mode.md) — the research journey (CLI)
- [docs/experiment-mode.md](docs/experiment-mode.md) — contract, funnel, shipping (CLI)
- [docs/security.md](docs/security.md) — security model and honest limitations
- [docs/architecture.md](docs/architecture.md) — code layout
- [docs/RESEARCHFORGE_PHASED_BUILD_SPEC.md](docs/RESEARCHFORGE_PHASED_BUILD_SPEC.md) — the product spec

## License

Apache-2.0 — see [LICENSE](LICENSE).
