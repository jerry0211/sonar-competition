# =====================================================================
# Kaggle bootstrap for Phase 2  (paste into a notebook cell)
# =====================================================================
# BEFORE running, set in the right-hand panel:
#   - Accelerator : GPU T4 x2   (or P100)
#   - Internet    : ON          (to pip-install the toolkit + fetch pretrained weights)
#
# --- get the toolkit onto the box ---
# Option A (Internet ON): install straight from GitHub
#   !pip -q install "git+https://github.com/<your-user>/sonar-competition.git"
#
# Option B (Internet OFF): upload the repo as a Kaggle Dataset, then:
#   import sys; sys.path.append("/kaggle/input/sonar-competition")
#
# --- persistent storage ---
# Write cache + checkpoints under /kaggle/working (saved when you click
# "Save Version"). Point experiments/02_audio_cnn/config.yaml at:
#   paths.cache_dir: /kaggle/working/cache
#   paths.ckpt_dir:  /kaggle/working/checkpoints
# Free sessions drop without warning; the Trainer checkpoints the best model per
# fold, so a re-run resumes from saved weights instead of starting over.

import torch
from sonar_toolkit.training.device import get_device, describe_device

print("CUDA available:", torch.cuda.is_available())
print("device:", describe_device(get_device()))

# Then run the experiment:
# !python experiments/02_audio_cnn/run.py
