from dataclasses import dataclass
from typing import Optional
import torch

@dataclass
class FixerModelOutput:
    x_fix: torch.FloatTensor
    others: Optional[dict] = None


