from dataclasses import dataclass
from typing import Optional
import torch

@dataclass
class ADModelOutput:
    score: torch.FloatTensor
    alpha: Optional[torch.FloatTensor] = None
    other: Optional[dict] = None

