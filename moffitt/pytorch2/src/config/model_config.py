

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
    """ 
    This model is a dynamic feedforward neural network for tabular data. It can take any number of numeric or categorical (input)features dynamically, process categorical features via embeddings, combine them with numeric features, pass them through a configurable MLP, and predict multiple targets simultaneously.
    Highly configurable: number of layers, hidden dimensions, dropout, batch normalization.
    This supports multi-target regression or multi-label classification, depending on the loss function used.
    """
    def __init__(self, emb_sizes: list[tuple[int, int]], n_numeric: int, n_targets: int, hidden_dims: list[int], dropout: float) -> None:
        """
        emb_sizes: List of (num_categories, embedding_size) for each categorical feature.
        Example: [(5, 3), (10, 4)] → 2 categorical features, first with 5 categories embedded into 3-dim vector, second 10 categories into 4-dim.
        n_numeric: Number of numeric features.
        n_targets: Number of outputs the model predicts (multi-target support).
        hidden_dims: List of hidden layer sizes for the MLP. Example: [128, 64].
        dropout: Dropout probability.
        """
        super().__init__()
        self.embeddings: nn.ModuleList = nn.ModuleList([nn.Embedding(c, s) for c, s in emb_sizes])
        # Total dimension after concatenating all embeddings.
        n_emb: int = sum(s for c, s in emb_sizes)
        
        layers: list[nn.Module] = []
        in_dim: int = n_emb + n_numeric
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim)) # Fully connect layer
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        
        # The hidden layers are combined in nn.Sequential. output_layer maps the final hidden layer to n_targets outputs.
        self.network: nn.Sequential = nn.Sequential(*layers) 
        self.output_layer: nn.Linear = nn.Linear(in_dim, n_targets)
        

    def forward(self, x_cat: torch.Tensor, x_num: torch.Tensor) -> torch.Tensor:
        x_emb: list[torch.Tensor] = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x: torch.Tensor = torch.cat(x_emb + [x_num], dim=1)
        return self.output_layer(self.network(x))

# Heuristic Embedding Dimensions
# The older / simpler rule: min(50, (cardinality + 1) // 2)
# The newer / more aggressive rule: min(600, round(1.6 * cardinality ** 0.56))