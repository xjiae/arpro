import os
import sys
from pathlib import Path
import wandb
import torch
import torch.nn as nn

""" Some directories """
PROJ_ROOT = str(Path(__file__).parent.parent.resolve())
sys.path.insert(0, PROJ_ROOT)

DUMP_DIR = str(Path(PROJ_ROOT, "_dump"))
Path(DUMP_DIR).mkdir(parents=True, exist_ok=True)
WANDB_PROJECT = "anomaly_project"

