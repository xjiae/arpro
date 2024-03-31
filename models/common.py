from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class ADModelOutput:
    score: torch.FloatTensor
    alpha: Optional[torch.FloatTensor] = None
    others: Optional[dict] = None


@dataclass
class FixerModelOutput:
    x_fix: torch.FloatTensor
    others: Optional[dict] = None

