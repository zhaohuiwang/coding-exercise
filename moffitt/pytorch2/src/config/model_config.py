

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Any, Optional

from torch.utils.data import Dataset

# --- DATASET & MODEL ---
class InputDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cats: list[str], nums: list[str], targets: list[str]) -> None:
        self.cats: torch.Tensor = torch.tensor(df[cats].values, dtype=torch.long)
        self.nums: torch.Tensor = torch.tensor(df[nums].values, dtype=torch.float32)
        self.y: torch.Tensor = torch.tensor(df[targets].values, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.cats[idx], self.nums[idx], self.y[idx]

class DynamicModel(nn.Module):
    def __init__(self, emb_sizes: list[tuple[int, int]], n_numeric: int, n_targets: int, 
                 hidden_dims: list[int], dropout: float) -> None:
        super().__init__()
        self.embeddings: nn.ModuleList = nn.ModuleList([nn.Embedding(c, s) for c, s in emb_sizes])
        n_emb: int = sum(s for c, s in emb_sizes)
        
        layers: list[nn.Module] = []
        in_dim: int = n_emb + n_numeric
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim
            
        self.network: nn.Sequential = nn.Sequential(*layers)
        self.output_layer: nn.Linear = nn.Linear(in_dim, n_targets)

    def forward(self, x_cat: torch.Tensor, x_num: torch.Tensor) -> torch.Tensor:
        x_emb: list[torch.Tensor] = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x: torch.Tensor = torch.cat(x_emb + [x_num], dim=1)
        return self.output_layer(self.network(x))
