# Changelog

## 0.4.7 — Kaggle's scores on every round, steered rounds first, Submit any round (2026-11-19)

- **Kaggle's score sits on the round it scored, and the next proposal reads it.** Every
  submission's public leaderboard score is matched to its round ("RF R25: …", or "round_14" in
  older descriptions). Runs has a **Kaggle LB** column and the DAG node shows "LB 0.13 (local
  0.40)"; a round Kaggle is still scoring says so, and one Kaggle finished without a score
  (usually a notebook that failed or ran out of time there) shows **no score**. `prepare-round`
  gives the agent a `leaderboard` table (local metric vs LB, newest first) and the API-key loop
  gets the same as a context section, with the rule to trust the LB where the two disagree.
  Enveda's rounds score about 0.93 locally but 0.10–0.47 on the LB; until now no proposal knew.
- **A submission Kaggle failed says why, and the next round is told to fix it.** Kaggle lists
  a notebook that ran out of time on the hidden test set as "complete" with no score; only its
  API says "exceeded the allowed runtime". The submissions listing now carries that reason, so
  Gemma's R24 and R26 read **timed out** (with Kaggle's text) in Runs, the DAG and
  Submissions instead of "no score", the failure push includes it, and the `leaderboard`
  table gives proposals `lb_error` with the rule to remove the cause (a shorter run, fewer
  calls, a time guard) before building on such a round.
- **A round whose Kaggle notebook is gone stops waiting.** Gemma's R27 and R33 had been
  "collected when it finishes" since their notebooks outlived the round's wait, but Kaggle no
  longer has those notebooks (404), so they never finished. A pending round whose notebook
  Kaggle reports missing for more than 12 hours is now closed as failed, with the reason.
- **A steered offer's own words no longer end up in Kaggle's description.** The submit step
  took the round's title from the offer's text up to its last apostrophe, so a note like
  "didn't beat" ran into the description (R37). The title now comes from the experiment log.
- **Submission offers for your steered rounds come first.** In 🔔, an offer for a round that
  followed one of your directions or pinned ideas now sits at the top of its project, marked
  🧭 with an amber edge and the direction it followed. It used to sort by age, so with dozens of
  newer offers (Enveda had 32) the RL round you asked for was buried. The project's row in 🔔
  says how many offers were steered by you.
- **Steered rounds are offered even when they don't beat the best.** A round that followed
  your direction or pinned idea and wrote a submission file now gets a Submit? offer, labelled
  with what it followed and, when it fell short, by how much ("followed your direction; didn't
  beat the best: 0.4000 vs 0.6000 (R45)"). You asked for that work, so you judge it. Gemma's RL
  rounds R36 and R37 tied the best of the time and were never offered before. Failed rounds,
  rounds already asked about and the conservative submission policy are left alone.
- **Submit any round from Runs or the DAG.** Every finished round that left something Kaggle
  takes has a **Submit** button in Runs and in its DAG node detail; it asks once, then sends the
  round (reusing its open offer if it has one). Rounds already sent show **submitted ✓**.
  If Kaggle refuses, nothing is used and the offer stays in 🔔.
- **A round is offered for submission once.** The engine and the agent driving an auto-run
  launch both offer improved rounds, seconds apart, so a round could show up twice in 🔔
  (Name that LLM's R11 did). Offering a round that already has an open offer now returns that
  offer instead of asking again.

## 0.4.6 — Token metering that counts, and kernel ideas through Claude (2026-11-19)

- **Kernel-idea extraction works through the Claude driver.** The prompt (a list of notebook
  titles, one "- " per line) was passed as an argument, which Claude Code read as an unknown
  option, so every extraction failed within a second and was logged as a 0-token call. The
  prompt now goes on stdin.
- **Insights counts the tokens auto-run actually uses.** Claude launches ran with text
  output, so the rounds themselves were never metered. They now ask Claude Code for its JSON
  reply: the log still gets the reply's text, and `token_usage.jsonl` gets one `auto_run` row
  per model the launch used, with cached input counted (it was dropped, so a call showed
  about 2 input tokens) and Claude Code's own reported cost. The launch record carries
  `cost_usd`. Sensor calls are logged under the model that answered instead of "default".
- **Insights says which tokens are which.** The tile reads **Tokens in / out**. The summary
  says what input and output mean, how much input came from cache, and which part of the cost
  was reported by Claude Code (the API-equivalent price; on a subscription it counts toward
  your plan's limits) and which was estimated from list prices. The per-model table adds
  "of which cached" and Cost columns. Rounds you drive from your own Claude Code chat still
  can't be metered, and the page says so.

## 0.4.5 — Up next: steer the queue, and auto-run keeps going (2026-11-18)

- **Auto-run keeps experimenting.** Launches now run back to back: when one ends, the next
  starts within 10 minutes instead of at the next 6-hour monitor cycle (`back_to_back` in Run
  configs; off goes back to one launch per cycle). The default daily limit is 20 rounds, up
  from 6, which is what stopped Enveda by mid-afternoon. Open questions and submission offers
  never stop it; only a pause, a question marked hold, the daily limit, the Kaggle GPU cap or the
  driver's usage limit do. The project badge for open questions reads "questions for you"
  instead of "waiting for you", which looked like the loop had stopped.
- **Up next: one numbered queue, and what the next round does.** The Steer page opens with
  **Next round R…**: what it will do (your direction, a pinned idea or the engine's top idea)
  and when it starts, or what is stopping it. Below it one list numbered 1, 2, 3…: your
  directions (the newest on top), then the ideas you pinned, then the engine's top 10. Every row
  has ↑/↓ (or drag); moving one of the engine's ideas up pins it, moving the last pinned idea
  down unpins it. Pin from **📌 Pin** on Hypotheses too (`researchforge kaggle pin`). The same
  Up next card sits at the top of Runs and Hypotheses. Pinned ideas go before the engine's
  ranking in every loop.
- **Hypotheses: the filters act on what you see first.** The idea list now comes before the
  public-notebook table, which starts folded; it used to fill the screen above the list, so a
  filter seemed to do nothing.
- **The top of Up next is followed, not just read.** A round's proposal names the item it
  follows (`"follows"`) or says why it cannot (`"declines"`); `execute-round` and
  `run-parallel` refuse a proposal that does neither, and the API-key loop is told once, then
  refused. A declined item stays at the top with the reason shown under it. The API-key loop
  now sees your directions at all; it did not before.
- **A next-round direction stays until a round follows it.** It used to close after the next
  round read it, whatever that round did. It can now ask for several rounds
  (`kaggle steer --rounds N`, or "for the next 3 rounds" on the page), and once done it stays
  on the Steer page for a day with the rounds that followed it and the agent's reply, instead
  of disappearing behind "Show applied".
- **🧭 What your steering led to is marked.** Rounds that followed one of your directions or
  pinned ideas get a 🧭 badge and an amber edge on Runs, an amber ring on the DAG, and a
  **Steered by** line in the round's details, with the direction it followed (or the reason it
  declined). Runs has a **Steered by you** filter, and Hypotheses a **🧭 Yours** filter for
  your pinned ideas and the ideas steered rounds tested. Each round's record is kept in
  `steered_rounds.json`.
- **Submissions left today counts every submission.** Kaggle's reply does not always include
  how many remain; the count used to come from an earlier reply and miss the later ones (it
  showed 3 of 5 left with 1 left). It now subtracts the submissions made after that reply,
  and never shows more than the daily limit less today's submissions.
- **A refused submission is recorded and names its round**, in `submissions_refused.jsonl`
  with Kaggle's full reply, so a bare 400 can be traced afterwards.

## 0.4.4 — Only notebooks that wrote a submission are offered (2026-11-18)

- **A notebook round is offered for submission only when it wrote a submission file.** Any
  finished Kaggle notebook used to count, so a notebook that ran out of time and wrote no
  `submission.csv` was offered, and approving it got a bare 400 Bad Request from Kaggle.
  Approving such a round now looks in the notebook's Kaggle output first (older rounds kept no
  copy): it submits when the file is there and otherwise says the notebook wrote none, without
  calling Kaggle. The monitor withdraws open offers whose notebook output has no submission
  file, and keeps them when Kaggle's output listing could not be read in full.

## 0.4.3 — A ported notebook brings everything it attaches (2026-11-16)

- **Rounds attach other notebooks' outputs, Kaggle Models, a Docker image and a GPU type**, not
  only datasets. Top public notebooks build on all of these; a round used to attach the datasets
  alone, so such a notebook stopped at its first missing input and spent Kaggle GPU quota for
  nothing. `propose-from-kernel` now copies every input the notebook declares into
  `proposal.json` (`dataset_sources`, `kernel_sources`, `model_sources`, `docker_image`,
  `machine_shape`), `pull-kernel` lists them, and `execute-round` and `run-parallel` push them
  with the round's notebook. A proposal written by hand can name them with the same keys.

## 0.4.2 — Code competitions run the full loop (2026-11-16)

- **A code competition now gets the experiment loop, on Kaggle GPU.** Setup used to send every
  code competition to the ideation path, which writes a list of ideas and never creates the
  loop, so `/overnight`, `kaggle overnight` and **＋ New competition** stopped there and no
  round could run. Code competitions now take the full setup with Kaggle GPU — rounds run as
  Kaggle notebooks, the only way such a competition accepts submissions — and choosing other
  compute stops setup with the reason. Data larger than the free disk still gets the ideation
  plan.

## 0.4.1 — Projects that need you are no longer listed as Idle (2026-11-16)

- **Dashboard: projects that need you are no longer listed as Idle.** The project picker now
  groups projects as **Needs you** (questions waiting, paused, or a run stalled), **Active**
  (running or watched) and **Idle**. Before, a competition with a question waiting was filed
  under Idle even though its watcher kept running.

## 0.4.0 — Kaggle overnight, background sensors and a live dashboard (2026-11-15)

0.3.0 was never published to PyPI; this is the first release that carries it.

### Answering from the dashboard

- **An answer can change a setting.** A question may tie an answer to Run configs changes
  (`kaggle ask … --action "<answer> => compute:kaggle, max_experiments:40"`). Picking that answer
  applies them through the same setters as Run configs, logged as "answer to q-…"; a refused
  value keeps the question open with the reason. Only the person can pick such an answer: an
  unattended run is refused. Before this, "switch compute to kaggle" only recorded the words, so
  auto-run found compute unchanged and asked again. Auto-run and `/overnight` now attach the
  change to such answers instead of asking the person to edit Run configs by hand.
- **The 🔔 groups questions by competition**, one collapsible section each with its count of
  questions and submission offers; the sections you open are remembered.
- **Submissions left today** shows on every submission offer and each competition's section:
  Kaggle's reply to today's last submission, else the competition's daily limit (read once from
  Kaggle's API) minus today's submissions. A used-up day is refused before calling Kaggle.

### Kaggle overnight

- **`/overnight <competition-url>`** skill and **`researchforge kaggle overnight`**: reads the
  competition, finds starter kernels, asks five setup questions, then runs a propose → run →
  measure → keep loop on the compute you pick — Kaggle GPU, this machine, Colab, an SSH server
  (`kaggle ssh-setup`) or a RunPod / Lambda Labs GPU (`kaggle cloud-gpu-setup`). Competition data
  never downloads locally when the compute policy forbids it.
- **Claude drives the DAG**: `kaggle prepare-round` (DAG state, merge opportunity, ranked
  hypothesis pool, next batch ready), `kaggle execute-round`, `kaggle run-parallel`,
  `kaggle resynthesize` every five rounds, `kaggle propose-hypothesis`, `kaggle finish`.
  Proposals name the hypothesis they test, and that hypothesis is marked tested with the round
  and its outcome.
- **Human-gated submissions**: `experiment submission-candidates` ranks candidates by evidence;
  `experiment submit <id>` asks for confirmation and records the public LB back into the DAG.

### Background sensors

- **`researchforge kaggle monitor`** runs every sensor in one cycle: new public kernels → ideas,
  submission status, hypothesis-pool growth, hung-experiment heartbeat, arXiv scan and a morning
  briefing. Each sensor's last result is recorded in `monitor_state.json`.
- **`researchforge kaggle sensors <slug>`** turns sensors on/off and adds custom sensors (any shell
  command; exit 0 = healthy), stored in `scheduler_config.json`.
- **`kaggle scheduler-install`** runs the monitor under launchd with the base interpreter and logs
  under `~/.local/share/researchforge/logs/`, so macOS privacy controls on `~/Desktop` no longer
  stop it (exit 78).

### Live dashboard

- **`researchforge dashboard`** (or `kaggle dashboard --open/--daemon/--stop`) serves one live page
  for every competition and repository project on this machine — nested repos, worktrees and
  backups included. Home, Automation, Runs, DAG, Hypotheses, Submissions, Timeline and Insights
  read one versioned state payload, polled with ETags; refreshes keep your view.
- DAG edges come from recorded parents; nodes show their patch, rationale and linked hypothesis;
  merge candidates are drawn dashed using the engines' own merge rules.
- Projects show running / waiting for you / stalled / watching / idle, deadline countdowns, a
  run gantt, launchd state with the real error line, and token usage per model, purpose and round.
- The old static report is `researchforge dashboard --legacy`.

### Set up, sign in and steer from the dashboard

- **＋ New competition**: paste a Kaggle URL; the dashboard gathers the facts and asks the
  `/overnight` questions as a form, then runs the same setup commands, showing each step's output.
  The answers are a draft (`setup_draft.json`, `kaggle draft <slug>`) that `/overnight` in Claude
  Code or Cursor picks up, and a chat can hand to the page (`kaggle draft <slug> --start`).
- **Keys & accounts**: one page and `kaggle keys status|open|wait|set|test|remove` for Kaggle,
  Anthropic/Gemini/OpenAI, the Claude Code and Cursor tokens, Colab, RunPod, Lambda, vast.ai and
  AWS. A value is saved once and never shown again. `/overnight` asks which accounts you want and
  opens the page for the missing ones instead of asking for a key in the chat. Cloud GPU rounds
  read their provider key from the same file, so a background monitor can use it.
- **Steer**: directions for the next round or every round (`kaggle steer`), handed to the
  proposer by `prepare-round` (`directions`) and auto-run's prompt, and answered on the page.
  Pause, resume, re-synthesize and add an idea from the same page.

### Colab rounds, the round time limit and phone alerts

- **Colab rounds survive a dropped connection.** `colab run` held one connection whose access
  token expires after about 30 minutes, so every longer round died with "Connection was lost".
  A round now starts in the background on the VM (`colab new`, `upload`), is checked every
  minute with short `colab exec` calls that renew the token, and its output and submission
  file are downloaded before the VM is released (`colab stop`, always). The kept file makes
  the round eligible for a submission offer.
- **Colab only means Colab.** `prepare-round` gives the proposer `compute`: where the round
  runs, its time limit and its rules. On Colab, `execute-round` and `run-parallel` refuse a
  round that pushes a Kaggle notebook.
- **Round time limit** (Run configs → Budget, `round_time_limit_minutes`): how long one round's
  full run may take, 30 minutes by default, instead of a fixed 30 minutes.
- **Alerts reach your topic.** Questions, approvals and submission offers ignored the
  machine-wide topic (`researchforge notify setup`) and a topic given as a setup answer, so
  they were never sent. Now: `NTFY_TOPIC`, then the competition's topic (`kaggle settings
  <slug> --ntfy-topic`), then the machine-wide one. With no topic, nothing is sent: the shared
  `researchforge-overnight` fallback, readable by anyone, is gone.
- **Run configs → Notifications**: where alerts go, how to subscribe, what each level sends,
  **New topic** and **Send test alert**; the machine-wide topic is editable there too.
- Tests no longer send real alerts or add projects to your dashboard.
- `kaggle ask --options` keeps commas inside brackets or quotes in one answer (or separate
  with `|`); an answer like "Switch to local (…, which …)" was split in two.
- A failed run shows no score on the Runs page instead of 0.0000.

### Unattended auto-run for repository projects

- **`researchforge auto-run`** runs a repository project's loop with no terminal open. A
  background monitor (`auto-run service`: launchd, systemd timer or Task Scheduler) runs a
  cycle every 15 minutes and starts a launch when its checks pass. A launch runs the plans you
  approved, then rounds of the real `autorun` engine.
- **Drivers:** Claude Code, Cursor (through their CLIs, every tool off) or an API key, chosen
  per project. Keys come from the same `auto_run.env` as competitions.
- **Approval:** each plan waits for **Approve plan** and sends a phone alert; or the person types a
  standing approval (`auto-run approval standing`), the consent `autorun` already asks for.
  Unattended runs cannot give it.
- **Kaggle-side controls carried over:** pause and resume, rounds per launch and per day,
  launch time limit, waiting out a driver's usage limit, alerts to the project's own ntfy topic,
  a daily summary, and a daily scan for new arXiv papers.
- **Dashboard:** the Automation page's repo card has the service switch, driver, settings,
  Start now / Stop, Pause, approval mode, topic and each check's status.
- `autorun`'s engine gains `ask_every_plan`, and `resolve_provider` a scoped override, so the
  engine plans with the project's driver.

### Unscored notebook rounds, real submission questions and the notebook board

- **A round with no local score but a submission is kept and offered.** A public notebook for
  a hidden test set prints no score, so these rounds were marked failed and dropped their
  output; Enveda's 0.480–0.500 notebook rounds (R7, R8, R21, R22) were never offered. They are
  now recorded *unscored* (purple in Runs and the DAG), keep their notebook or file, and get a
  submission question: Kaggle's score is the only measure of them.
- **"Submit R21's output?" asked by an agent is now that round's submission question**, so your
  yes submits it. Before, it was a plain question: two yeses on Enveda submitted nothing. A
  submit question about a round with nothing to submit is refused. Agents use
  `researchforge kaggle offer-submission <slug> <round> --note "…"`.
- **Kaggle's daily submission limit is checked before submitting.** Kaggle's reply to each
  submission says how many are left today; with 0 left, an approval is refused with the reset
  time (00:00 UTC) and the question stays open. Before, R7, R8 and R21 on Enveda went to Kaggle
  and came back as an unreadable "400 Bad Request". The allowance shows on each approval and
  the Submissions page, and the "outdated kaggle" warning is no longer shown as the error.
- **A round is never submitted twice, and one competition sends one submission at a time.**
  Enveda's R26 was sent twice at 14:51 and 14:52, spending a submission.
- **Improvements count without a baseline score.** A round the loop marked improved is shown as
  improved when it beat every earlier round; the cards said "0 improved" on every project.
- **Public notebooks are checked every hour** (`monitor --kernel-minutes`, default 60), not every
  6 hours, including notebooks *re-run* since they were last seen. Each check rebuilds
  `kernel_board.json`: every notebook with its claimed LB, Kaggle's best-score order, when it
  last ran and our rounds that ran it, ranked by claimed LB. The dashboard's Hypotheses page
  shows it per project, and `prepare-round` gives the proposer `public_notebooks_ahead`.
- **"Better notebook" alerts compare with our best leaderboard score.** They compared with the
  setup baseline, which no project had, so none was ever sent.
- **The idea pool ranks notebooks claiming more than our best LB first**, highest claim first,
  ahead of local measurability and relevance. A local benchmark can miss the leaderboard
  entirely (Enveda: local 0.915, LB 0.21, public notebooks 0.49+). The Hypotheses page sorts by
  claim by default.

### Loop and notifications

- `autorun --no-metric-gate` records every result instead of auto-rejecting (for when the local
  metric is not the real score); `experiment plan --merge a,b` combines two independent winners.
- `researchforge notify setup --topic <topic>` persists ntfy settings for every process; every
  experiment result, merge opportunity and briefing is pushed, and logged to
  `~/.config/researchforge/notification_log.jsonl`.
- Producers write to one resolved `.overnight_<slug>` directory per competition instead of
  whichever folder a command ran from, and JSON state is written atomically.

### Steer a running loop without stopping it

- **`kaggle status | pause | resume | respond | compute | add-hypothesis | settings`**, and the
  same verbs as `/overnight status`, `/overnight pause`, … in the skill. A pause stops
  `prepare-round`, `execute-round` and `run-parallel` from starting new rounds; a round already
  running finishes and sensors keep going.
- **Questions never block the loop.** Mid-loop questions and submission offers are recorded in
  `pending_interaction.json`; the loop keeps working on anything they do not block, and
  `kaggle respond` answers them (domain answers flow into the next proposals).
- **Notification level** (`verbose` / `balanced` / `minimal`) and **submission policy**
  (`aggressive` / `balanced` / `conservative`) per competition. Nothing is ever submitted
  automatically — a submission offer records your decision only.
- **Daily Kaggle GPU cap** (`kaggle settings --kaggle-gpu-daily-cap N`): `execute-round` refuses
  on Kaggle compute once today's cap is reached.
- **Kernel-watcher ideas are tagged by relevance** — builds on the best path, repeats a failure,
  or a new direction — and `prepare-round` ranks them that way, with your own ideas first.
- **`autorun --kaggle`** records every result, tags how far each local metric can be trusted, marks
  what is worth Kaggle GPU time, and lists submission candidates at the end. **`autorun
  --judgment`** asks per experiment instead (a prompt, or a queued question answered with
  `experiment judge <id> yes|no|skip`). Evidence goes to `.researchforge/kaggle_evidence.json`.
- **More compute:** vast.ai (`VAST_API_KEY`) and AWS EC2 (through the `aws` CLI) alongside RunPod and
  Lambda; `scheduler-install` now also installs a systemd user timer on Linux and a Task Scheduler
  task on Windows.
- Fixed: "no improvement" round notifications were never sent (an invalid format string raised
  inside the notifier and was swallowed).

### Auto-run, questions and run configs

- **Auto-run** (`kaggle auto-run <slug> --on`, off by default): `auto_run_sensor` starts rounds on
  untested ideas with no chat open, driven by headless Claude Code (`claude -p`, allowed only
  `researchforge kaggle …` commands and edits inside the project), the Cursor agent
  (`cursor-agent -p`) or your API key (`kaggle run --rounds N`). A launch starts only when setup is
  done, the loop isn't paused, no question holds it, no round is running, and the daily limit and
  Kaggle GPU cap allow it; `--max-minutes` stops a long launch. It never submits.
  With no flag it shows its status; also `--dry-run`, `--now` and `--stop`.
- **One driver per project, switchable any time** (`kaggle driver <slug> <driver>`). The driver also
  does the sensors' AI work through one-shot, tool-less calls (`--no-sensors` keeps them on a
  key), and every round records which driver ran it. `kaggle driver-login <claude|cursor|byok|colab>`
  signs a driver in from your own terminal; tokens and keys go to
  `~/.config/researchforge/auto_run.env` (mode 600). `kaggle auto-run-key <NAME>` stores one value.
- **Questions come to you**: `kaggle ask <slug> "…" --options a,b --blocks "<idea>" [--hold]` pushes
  to your phone and the dashboard's 🔔 and Home; only the ideas it names wait.
  `kaggle respond <slug> <id> --dismiss` closes one unanswered and its ideas run again. A
  standalone repository project's plans waiting for approval are listed too: **Approve plan** asks
  you to type `approve`, as `researchforge experiment approve` does; unattended runs never approve
  plans. A repo project beside a competition gets no plan approvals.
- **Run configs** in the dashboard: every setting for a competition, editable on the fly through the
  same setters as the CLI, with a change log.
- **Metric direction is measured**: read from the leaderboard's order, with its source recorded and a
  guess flagged; lower-is-better is handled everywhere. `kaggle settings --metric-direction` fixes it.
- **Every compute option is offered at setup**; `--gpu` accepts local, kaggle, colab, ssh, cloud and
  competition, and an unknown or unconfigured target stops setup.
- **Colab without clicking Run all**, through Google's Colab CLI: each round rents a fresh VM with
  the GPU your plan allows (T4, L4, G4, A100, H100, optional high-RAM), downloads the data there
  with your Kaggle key, runs and releases the VM. `kaggle colab-setup <slug> --plan pro --gpu L4`
  and `kaggle driver-login colab`; without a sign-in, rounds wait. Used on a real Colab account
  (a T4 VM up in about 15 s, Titanic data downloaded on the VM, score read back, VM released);
  Colab Enterprise is not supported yet.
- **One history per competition**: the Kaggle loop is a competition's only engine, and auto-run
  always continues it. A repo project beside it is folded in (`kaggle merge-history`, also run by
  the monitor each cycle): its finished experiments appear in the same graph, dotted and marked
  imported, and never count as best or as a parent; its untested hypotheses join the pool. With
  no baseline, a round must beat the best earlier round to count as improved or be offered for
  submission.
- **Notebook score claims**: notebook ideas carry the LB score their author claims and its gap to
  your best public score, ranked by it but never treated as a measurement.
- **`kaggle archive <slug> R7`** (reversible) and **`kaggle delete-round`** (confirmed, not
  available to unattended runs); **`kaggle pull-kernel <slug> <owner/notebook>`** downloads a
  public notebook into `reference_kernels/`.
- Fixed: a round whose process died left a lock that blocked every later round — the lock is now
  released and the round recorded as interrupted; the monitor runs a cycle missed while the Mac
  slept within a minute of waking.
- Fixed: running `/overnight` again on a set-up project redid setup — it now resumes (`--refresh`
  starts over).

### Install and models

- **Claude Code plugin.** The repository is a plugin marketplace:
  `/plugin marketplace add forger-labs-hq/researchforge`, then
  `/plugin install researchforge@researchforge`. It ships the same skills as
  `researchforge claude install`, namespaced as `/researchforge:overnight` and so on.
- **Local models through Ollama.** `RESEARCHFORGE_LLM=ollama/<model>` (server at `OLLAMA_HOST`,
  default `http://localhost:11434`) uses Ollama's native API: no key and no SDK. Calls are recorded
  as `ollama/<model>`, and their tokens are priced at zero. Kaggle Overnight runs headless on it
  (`ai_provider_mode: ollama`), and background services now pass on `RESEARCHFORGE_LLM` and
  `OLLAMA_HOST`. They also pass on `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`; before this,
  Telegram alerts stopped under the scheduler. The previously documented `http://localhost:11434/api` URL now reaches Ollama;
  it used to be sent to an OpenAI client that cannot talk to that endpoint.
- **`researchforge --version`.**
- Fixed: setup claimed API-key mode lets "the laptop be off". The loop always runs on the
  machine where it starts. A key or a local model removes the need for the editor, not for the
  machine.

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
