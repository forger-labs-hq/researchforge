# Docker isolation for the YOLOv5 example. venv mode is the default and needs
# nothing extra; use this when the host Python cannot build torch, or when you
# want the benchmark environment to be identical across machines.
#
#   researchforge.yaml → execution.mode: docker
#
# CPU inference only: Apple's MPS backend and NVIDIA CUDA are not reachable from
# inside a portable container, so Docker numbers are slower than venv numbers on
# the same machine. That is fine — the baseline is frozen in whichever mode you
# choose, and every experiment is compared against it in that same mode. The
# contract records the mode, and ResearchForge refuses to compare across modes.
FROM python:3.12-slim

# libgl/libglib: opencv, an ultralytics dependency, links against them.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work

# torch is large; installing before the source copy keeps it in a cached layer
# so a one-line change to src/config.py does not reinstall it.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Weights and COCO128 are fetched on first run, which is why the contract sets
# network.mode: enabled. To run fully offline, bake them in here instead:
#   RUN python -c "from ultralytics import YOLO; YOLO('yolov5su.pt')"
CMD ["python", "benchmarks/evaluate.py"]
