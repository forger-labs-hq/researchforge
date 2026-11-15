"""Entry point of a detached auto-run launch: ``python -m researchforge.kaggle.auto_run_child``.

A separate module so ``-m`` does not re-execute ``auto_run``, which the package has
already imported (runpy warns about that, and the module would load twice).
"""

from __future__ import annotations

import sys
from pathlib import Path

from researchforge.kaggle.auto_run import run_child

if __name__ == "__main__":  # python -m researchforge.kaggle.auto_run_child <artifacts_dir> <slug>
    raise SystemExit(run_child(Path(sys.argv[1]), sys.argv[2]))
