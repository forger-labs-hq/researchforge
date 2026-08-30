"""AI-powered experiment plan generation.

Given an experiment context (hypothesis + contract + baseline results + repo files),
generates:
  plan.yaml            — the experiment plan artifact
  patches/<name>.patch — one unified diff per variant  (for code changes)

For simple config-only experiments the AI may use env_overrides instead of
patch files, in which case patches is an empty dict.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

from researchforge.ai.file_diff import PatchSynthesisError, synthesize_patch
from researchforge.ai.providers import AiProvider
from researchforge.experiments.context_export import PATCHES_DIR_NAME, ExperimentContext
from researchforge.experiments.repo_context import RepoSnapshot

_SYSTEM = """\
You are generating a ResearchForge experiment plan for a machine learning project.

Given a hypothesis and project context, produce:
1. plan.yaml — specifying experiment variants
2. Zero or more patch files — unified diffs for code changes

PLAN.YAML FORMAT — these three top-level keys and nothing else. Any other
top-level key is rejected by the importer and the whole plan is thrown away.
```yaml
hypothesis_id: <hyp-id>
approach_summary: <one sentence>
experiments:
  - key: <slug>             # a-z0-9 and hyphens, e.g. "nano-backbone"
    title: <short title>
    change_summary: <what changes and why>
    # Use env_overrides ONLY when the code already reads that variable
    # (os.environ / getenv). Otherwise change the file itself — see <files>.
    env_overrides:
      CONFIG_KEY: "new_value"
    expected_effect:
      metric: <metric_name>
      direction: increase   # or decrease
    notes: <optional>
```

Do NOT write `patch_file` yourself and do NOT write diffs. Give the changed
file in <files> instead and the engine writes the patch and links it.

WHEN TO USE env_overrides vs a file change:
- The code reads the value from the environment already → env_overrides
- The value is a constant in the source, or the change is code logic → <files>
- NEVER use both on the same experiment entry

RULES:
- Change ONLY files under editable_paths, and only files listed in REPOSITORY
  below. Do not invent a file that is not there; if what you need does not
  exist, choose a different experiment.
- Never touch protected_paths
- 2-5 experiments max
- Each experiment tests ONE change
- Keep every variant runnable: the benchmark command must still work and still
  write the result file

OUTPUT FORMAT — output EXACTLY this structure:
<plan>
(plan.yaml content here)
</plan>
<files>
(JSON: {"<experiment key>": {"<repo path>": "<COMPLETE new content of that file>"}}
 Use {} when every experiment uses env_overrides.
 Give the WHOLE file, not a fragment and not a diff: it replaces the original
 byte for byte. Copy the parts you are not changing exactly as they appear.
 Example: {"nano-backbone": {"src/config.py": "MODEL = \\"nano\\"\\nEPOCHS = 10\\n"}})
</files>
"""


def _extract_tag(text: str, tag: str) -> str | None:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    if m:
        content = m.group(1).strip()
        content = re.sub(r"^```(?:yaml|json)?\s*\n?", "", content, flags=re.MULTILINE)
        content = re.sub(r"\n?```\s*$", "", content, flags=re.MULTILINE)
        return content.strip()
    return None


def _repository_section(ctx: ExperimentContext) -> str:
    """The editable source, verbatim. Everything the plan may change is here."""
    repo = ctx.repository
    if not repo.files and not repo.omitted:
        return (
            "## REPOSITORY\n"
            "No editable files could be read. Plan env_overrides-only "
            "experiments, or none at all.\n"
        )

    where = (
        f"as {repo.applied[-1]} leaves them (baseline + {', '.join(repo.applied)})"
        if repo.applied
        else "at the baseline commit"
    )
    parts = [
        f"## REPOSITORY — the editable files {where}\n"
        "This is the complete, current content of every file you may change. A "
        "patch is applied to exactly this state, so change only what is here.\n"
    ]
    for file in repo.files:
        parts.append(f"### {file.path}\n```\n{file.content}\n```\n")
    if repo.omitted:
        listed = "\n".join(f"- {o.path} ({o.reason})" for o in repo.omitted)
        parts.append(
            "### Present but not shown\n"
            "These exist and may be read by the code, but their contents are "
            "not included — do not rewrite them.\n" + listed + "\n"
        )
    return "\n".join(parts)


def _build_prompt(ctx: ExperimentContext) -> str:
    lines: list[str] = []

    h = ctx.hypothesis
    lines.append(f"## Hypothesis\n**{h.title}**\n{h.claim}\n{h.rationale}\n")
    lines.append(f"**Proposed experiment:** {h.proposed_experiment}\n")
    if h.supporting_paper_ids:
        lines.append(f"**Supporting papers:** {', '.join(h.supporting_paper_ids)}\n")

    c = ctx.contract
    constraints = [f"{hc.name} {hc.operator} {hc.value}" for hc in c.hard_constraints]
    lines.append(
        f"## Contract\n"
        f"- Objective: {c.objective_description}\n"
        f"- Primary metric: {c.primary_metric.name} ({c.primary_metric.direction.value})\n"
        f"- Hard constraints: {constraints}\n"
        f"- Editable paths: {c.editable_paths}\n"
        f"- Protected paths: {c.protected_paths}\n"
        f"- Full command: {c.full_command}\n"
        f"- Result file: {c.result_file}\n"
        f"- Max experiments: {c.max_experiments}\n"
    )

    b = ctx.baseline
    lines.append(
        f"## Baseline result\n"
        f"- {b.primary_metric.name} = {b.primary_metric.value}\n"
        f"- Secondary: {b.secondary_metrics}\n"
    )

    lines.append(_repository_section(ctx))

    lines.append(
        f"## Expected plan artifact paths\n"
        f"- plan.yaml: {ctx.expected_artifacts.plan_path}\n"
        f"- patches dir: {ctx.expected_artifacts.patches_dir}\n"
    )

    lines.append("## Authoring instructions\n" + "\n".join(f"- {i}" for i in ctx.instructions))

    lines.append(
        "\n## Task\n"
        "Generate a plan.yaml with 2-4 experiment variants for this hypothesis.\n"
        "Prefer env_overrides for config-only changes (MODEL_VARIANT, EPOCHS, LR, etc.).\n"
        "Use patch_file only for actual code logic changes.\n"
        f"Hypothesis ID to use: {h.hypothesis_id}\n"
    )

    return "\n".join(lines)


_PLAN_TOP_LEVEL_KEYS = ("hypothesis_id", "approach_summary", "experiments")
"""In the order plan.yaml reads best; also the only keys the importer accepts."""


def _drop_unknown_top_level_keys(parsed: object, plan_raw: str) -> str:
    """Remove top-level keys the plan artifact forbids, returning the YAML to write.

    The importer rejects unknown keys outright, which throws away an entire
    round's planning over a stray `plan_id_hint` or `version`. None of those
    can affect what runs, so dropping them costs nothing.

    This deliberately stops at the top level. An unknown key *inside* an
    experiment entry — `env` where `env_overrides` was meant — would change
    what the experiment tests, and must keep failing loudly rather than
    silently running an unmodified variant.
    """
    if not isinstance(parsed, dict):
        return plan_raw
    unknown = set(parsed) - set(_PLAN_TOP_LEVEL_KEYS)
    if not unknown:
        return plan_raw
    kept = {key: value for key, value in parsed.items() if key not in unknown}
    return yaml.safe_dump(kept, sort_keys=False, allow_unicode=True)


def planning_phase_label(ctx: ExperimentContext, provider_name: str) -> str:
    """What the model was handed — so a long wait reads as work, not a hang."""
    shown = len(ctx.repository.files)
    kb = sum(f.size_bytes for f in ctx.repository.files) / 1024
    return (
        f"Asking {provider_name} for variants "
        f"({shown} file(s), {kb:.0f} KB of code) — usually 30–90s…"
    )


def describe_plan(plan_yaml: str, patches: dict[str, str]) -> str:
    """One line saying what came back, before it is validated."""
    try:
        parsed = yaml.safe_load(plan_yaml)
    except yaml.YAMLError:
        return "the response was not valid YAML"
    entries = parsed.get("experiments") if isinstance(parsed, dict) else None
    count = len(entries) if isinstance(entries, list) else 0
    detail = f"{len(patches)} with a code patch" if patches else "all config-only"
    return f"{count} variant(s) proposed, {detail}"


def _parse_json_object(raw: str, what: str) -> dict[str, object]:
    if raw.strip() in {"", "{}"}:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{what} JSON parse error: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{what} must be a JSON object.")
    return parsed


def _patches_from_files(
    files_by_key: dict[str, object], repo: RepoSnapshot
) -> tuple[dict[str, str], dict[str, str]]:
    """Diff each experiment's rewritten files against the baseline content.

    Returns the patches to write, keyed by file name, and the `patch_file`
    value each experiment key should carry in plan.yaml.
    """
    patches: dict[str, str] = {}
    patch_file_by_key: dict[str, str] = {}

    for key, rewritten in files_by_key.items():
        if not isinstance(rewritten, dict) or not rewritten:
            continue
        after: dict[str, str] = {}
        for path, content in rewritten.items():
            if not isinstance(content, str):
                raise ValueError(f"experiments.{key}: {path} must be the file's text.")
            omitted = repo.was_omitted(str(path))
            if omitted is not None:
                raise ValueError(
                    f"experiments.{key}: {path} was not shown in full "
                    f"({omitted.reason}), so it cannot be rewritten."
                )
            after[str(path)] = content

        # Only paths that exist at the baseline: a path absent from `before` is
        # a new file, and diffing it as an empty existing one would not apply.
        before = {path: content for path in after if (content := repo.content_of(path)) is not None}
        try:
            diff = synthesize_patch(before, after)
        except PatchSynthesisError as exc:
            raise ValueError(f"experiments.{key}: {exc}") from exc
        if not diff.strip():  # the "change" was the file as it already is
            continue
        name = f"{key}.patch"
        patches[name] = diff
        patch_file_by_key[key] = f"{PATCHES_DIR_NAME}/{name}"

    return patches, patch_file_by_key


def _link_patches(parsed: object, patch_file_by_key: dict[str, str]) -> object:
    """Point each experiment at the patch written for it, and drop stale links.

    The model is told not to write `patch_file`; this makes that true either
    way, so a remembered file name from its training data cannot send the
    importer looking for a patch nobody wrote.
    """
    if not isinstance(parsed, dict):
        return parsed
    entries = parsed.get("experiments")
    if not isinstance(entries, list):
        return parsed
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        linked = patch_file_by_key.get(str(entry.get("key", "")))
        if linked is not None:
            entry["patch_file"] = linked
            entry.pop("env_overrides", None)  # a patch and env vars are exclusive
        elif "patch_file" in entry:
            entry.pop("patch_file")
    return parsed


def generate_experiment_plan(
    ctx: ExperimentContext,
    provider: AiProvider,
) -> tuple[str, dict[str, str]]:
    """Call AI and return (plan_yaml_str, {patch_filename: patch_content}).

    The model returns changed files; the diffs are computed here against the
    baseline content, so a patch is never the model's arithmetic.

    Raises ValueError with a descriptive message if parsing fails.
    """
    prompt = _build_prompt(ctx)
    raw = provider.generate(_SYSTEM, prompt, max_tokens=8192)

    plan_raw = _extract_tag(raw, "plan")
    files_raw = _extract_tag(raw, "files")
    patches_raw = _extract_tag(raw, "patches")

    if plan_raw is None:
        raise ValueError("AI response missing <plan>…</plan>. Try again.")

    try:
        parsed = yaml.safe_load(plan_raw)
    except yaml.YAMLError as exc:
        raise ValueError(f"plan.yaml YAML parse error: {exc}") from exc

    if patches_raw is None:
        # No <files> at all is a plan that changes no files — every variant is
        # an env_overrides one. That is a normal answer, not a broken response.
        files_raw = files_raw or "{}"

    if files_raw is not None:
        files_by_key = _parse_json_object(files_raw, "files")
        patches, patch_file_by_key = _patches_from_files(files_by_key, ctx.repository)
        linked = _link_patches(parsed, patch_file_by_key)
        if not isinstance(linked, dict):
            raise ValueError("plan.yaml must be a mapping.")
        kept = {key: linked[key] for key in _PLAN_TOP_LEVEL_KEYS if key in linked}
        return yaml.safe_dump(kept, sort_keys=False, allow_unicode=True), patches

    # A model that still answers with diffs: take them as written.
    patches = {
        name: text
        for name, text in _parse_json_object(patches_raw or "{}", "patches").items()
        if isinstance(text, str)
    }
    return _drop_unknown_top_level_keys(parsed, plan_raw), patches


def write_patch_files(patches_dir: Path, patches: dict[str, str]) -> list[Path]:
    """Write generated patches into `patches_dir`, returning the paths written.

    The model is asked for keys like ``patches/nano.patch`` — the same spelling
    `patch_file` uses in plan.yaml, where it is relative to the experiments
    directory. Joining that onto `patches_dir` would nest a second `patches/`
    that does not exist, so only the file name is used. This also keeps a key
    with `..` in it from escaping the directory.
    """
    patches_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, content in patches.items():
        target = patches_dir / Path(name).name
        target.write_text(content, encoding="utf-8")
        written.append(target)
    return written
