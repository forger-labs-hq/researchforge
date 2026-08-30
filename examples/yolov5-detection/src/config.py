"""Tunable detection settings — experiments patch this file.

Every value here changes what the benchmark measures. They are deliberately
plain module constants rather than a config object, so an experiment's patch is
a one-line unified diff that is obvious to review before it runs.
"""

# The pretrained checkpoint to evaluate. A larger model usually detects more and
# always costs more time, which is the trade-off the latency constraint forces
# an experiment to confront.
MODEL = "yolov5su.pt"

# Confidence floor for a detection to count. Lower keeps more low-confidence
# boxes: recall rises, precision falls, and mAP can move either way.
CONF = 0.001

# IoU threshold for non-maximum suppression, i.e. how much two boxes may
# overlap before one is discarded as a duplicate.
IOU = 0.6

# Square inference resolution. More pixels finds smaller objects and costs
# roughly quadratic time.
IMGSZ = 640
