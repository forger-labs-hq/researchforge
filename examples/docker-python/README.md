# docker-python — the launch demo, in Docker mode

The same classifier and benchmark as
[simple-python](../simple-python/README.md), plus a `Dockerfile`, with the
contract's `execution.mode: docker`. ResearchForge builds the image from
each experiment worktree and runs the benchmark in a locked-down container
(`--network=none`, CPU/memory/pids limits, `--cap-drop=ALL`,
`no-new-privileges`, non-root, nothing mounted but the worktree and the
artifact directory).

Use this variant when you want to see the stronger isolation mode; the
numbers and the demo script are identical to simple-python. Requirements:
a running Docker daemon (`researchforge doctor` checks). Everything else in
[docs/demo.md](../../docs/demo.md) applies unchanged.

Docker improves process and dependency isolation but is **not** a hardened
sandbox for hostile code — run only repositories you trust (see
[Security & Isolation](../../README.md#security--isolation--what-runs-where)).
