"""Benchmark: YOLOv5 detection accuracy and inference cost on COCO128.

Writes the machine-readable result ResearchForge's contract expects:

    artifacts/results.json
    {"schema_version": 1,
     "primary_metric": {"name": "map50", "value": ...},
     "secondary_metrics": {"inference_ms": ..., "map50_95": ..., "recall": ...}}

`--quick` evaluates at a smaller resolution, which is the screening pass: it
correlates with the full result and finishes in a fraction of the time, so
obviously bad variants are dropped before they cost a full evaluation.

Unlike the simple-python demo this is a real model on real images, so the
numbers depend on the machine. That is the point of freezing a baseline: every
experiment is compared against a measurement taken on the same hardware, in the
same isolated environment, rather than against a number copied from a paper.
"""

import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402

SCREENING_IMGSZ = 320


def measure(imgsz: int) -> dict[str, float]:
    """Validate the configured model and return the metrics that decide things.

    The import is here rather than at module scope so that `--help`-style
    inspection and any import error surface separately from the model load,
    which is slow and downloads weights on a cold cache.
    """
    from ultralytics import YOLO

    model = YOLO(config.MODEL)

    started = time.perf_counter()
    metrics = model.val(
        data="coco128.yaml",
        imgsz=imgsz,
        conf=config.CONF,
        iou=config.IOU,
        verbose=False,
        plots=False,
    )
    wall_seconds = time.perf_counter() - started

    # `speed` is per-image milliseconds, reported by ultralytics for the stages
    # it times. Inference is the part a model change actually moves; the wall
    # clock also carries dataset loading and metric computation, so it is kept
    # separately rather than passed off as inference time.
    speed = getattr(metrics, "speed", {}) or {}
    return {
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "recall": float(metrics.box.mr),
        "precision": float(metrics.box.mp),
        "inference_ms": float(speed.get("inference", 0.0)),
        "preprocess_ms": float(speed.get("preprocess", 0.0)),
        "postprocess_ms": float(speed.get("postprocess", 0.0)),
        "wall_seconds": wall_seconds,
    }


def main() -> None:
    quick = "--quick" in sys.argv
    imgsz = SCREENING_IMGSZ if quick else config.IMGSZ
    measured = measure(imgsz)

    result = {
        "schema_version": 1,
        "primary_metric": {"name": "map50", "value": round(measured["map50"], 4)},
        "secondary_metrics": {
            "inference_ms": round(measured["inference_ms"], 2),
            "map50_95": round(measured["map50_95"], 4),
            "recall": round(measured["recall"], 4),
            "precision": round(measured["precision"], 4),
        },
        "metadata": {
            "model": config.MODEL,
            "imgsz": str(imgsz),
            "conf": str(config.CONF),
            "iou": str(config.IOU),
            "stage": "screening" if quick else "full",
        },
    }
    pathlib.Path("artifacts").mkdir(exist_ok=True)
    pathlib.Path("artifacts/results.json").write_text(json.dumps(result, indent=2), "utf-8")

    print(f"{'screening' if quick else 'full'} evaluation at imgsz={imgsz}")
    print(f"  map50       = {result['primary_metric']['value']}")
    print(f"  inference   = {result['secondary_metrics']['inference_ms']} ms/image")
    print(f"  map50_95    = {result['secondary_metrics']['map50_95']}")
    print(f"  wall clock  = {measured['wall_seconds']:.1f}s")


if __name__ == "__main__":
    main()
