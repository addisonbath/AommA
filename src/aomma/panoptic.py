"""Mask2Former's panoptic segments of a run's photo, for the missed-listings look (`stages.relook`).

Runs in its own pinned environment, not aomma's (torch and transformers are not aomma dependencies), so it imports
nothing from aomma; `stages.relook.predict` starts it as

    uv run --no-project --with torch==2.15.0 --with transformers==5.24.0 --with timm==1.0.31 --with pillow \
        --with numpy python src/aomma/panoptic.py <run>...

For each run it segments input.jpg (the 1024 px photo the masks and labels.png are made on) and saves
`<run>/panoptic.npz`: `segments` (int32, one id per pixel, 0 where no segment) and `info` (JSON: each segment's id,
ADE20K label and score, and the model, versions and seconds). The labels are kept to read by; nothing decides on them.
A run that already has panoptic.npz is skipped. Transformers reports Swin's final layer norm as missing from the
checkpoint (left at its initial values); the segments were checked by eye and look right regardless.
"""

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # before torch loads

import numpy as np
import timm
import torch
import transformers
from PIL import Image
from transformers import AutoModelForUniversalSegmentation, AutoProcessor

MODEL = "facebook/mask2former-swin-large-ade-panoptic"
OUT = "panoptic.npz"


def main(runs: list[Path]) -> None:
    todo = [run for run in runs if not (run / OUT).exists()]
    if not todo:
        return
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(MODEL)
    model = AutoModelForUniversalSegmentation.from_pretrained(MODEL).to(device).eval()
    for run in todo:
        image = Image.open(run / "input.jpg").convert("RGB")
        start = time.time()
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model(**inputs)
        result = processor.post_process_panoptic_segmentation(output, target_sizes=[image.size[::-1]])[0]
        seconds = time.time() - start
        segments = result["segmentation"].cpu().numpy().astype(np.int32)
        segments[segments < 0] = 0
        info = {
            "model": MODEL, "device": device, "seconds": round(seconds, 2),
            "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "timm": timm.__version__},
            "segments": [{"id": int(s["id"]), "label": model.config.id2label[s["label_id"]],
                          "score": round(float(s["score"]), 3)} for s in result["segments_info"]],
        }
        np.savez_compressed(run / OUT, segments=segments, info=json.dumps(info))
        print(f"{run.name}: {len(info['segments'])} panoptic segments, {seconds:.1f} s", flush=True)


if __name__ == "__main__":
    main([Path(p) for p in sys.argv[1:]])
