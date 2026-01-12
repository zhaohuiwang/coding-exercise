



# Flexible Variable MLP Model
import torch
import torch.nn as nn

# Simple input X
class FlexibleMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, dropout=0.0, activation="relu"):
        super().__init__()
        # Map string to activation function
        activations = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(0.1),
            "gelu": nn.GELU()
        }
        act_fn = activations.get(activation, nn.ReLU())
        
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(act_fn)
            layers.append(nn.Dropout(dropout))
            # layers += [nn.Linear(in_dim, h), act_fn, nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)
    

    
# When input X has mixed categorical and numerical types 
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
    







# General Optuna Objective
import torch.optim as optim
import optuna
from torch.utils.data import DataLoader

def flexible_mlp_objective(
    trial: optuna.Trial,
    train_dataset,
    val_dataset,
    input_dim: int,
    output_dim: int,
    device: torch.device,
    task: str = "regression",
    n_epochs: int = 20,
    max_layers: int = 5,
    units_range: tuple = (32, 512),
    dropout_range: tuple = (0.0, 0.5),
    emb_sizes = [(2, 1), (3, 2), (51, 26)],
):
    # 1️⃣ Sample batch size
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])

    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    train_loader = train_dataset
    val_loader = val_dataset
    
    # 2️⃣ Sample architecture
    n_layers = trial.suggest_int("n_layers", 1, max_layers)
    hidden_dims = [
        trial.suggest_int(f"n_units_l{i}", *units_range, log=True) for i in range(n_layers)
    ]
    dropout = trial.suggest_float("dropout", *dropout_range)
    
    # 3️⃣ Sample activation function
    activation = trial.suggest_categorical("activation", ["relu", "leaky_relu", "gelu"])
    
    # 4️⃣ Build model
    model = FlexibleMLP(input_dim, output_dim, hidden_dims, dropout, activation).to(device)

    # model: nn.Module = DynamicModel(
    #         emb_sizes, 13, 3, hidden_dims, dropout,
    #     ).to(device)


    
    # 5️⃣ Sample optimizer
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD", "RMSprop"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_name == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    
    # 6️⃣ Loss function
    if task == "regression":
        criterion = nn.MSELoss()
    elif task == "classification":
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown task: {task}")
    
    # 7️⃣ Training loop
    for epoch in range(n_epochs):
        model.train()
        for xc, xn, y in train_loader:
            x = torch.cat([xc, xn], dim=1).to(device)
            y = xc.to(device), xn.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
        
        # 8️⃣ Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xc, xn, y in val_loader:
                x = torch.cat([xc, xn], dim=1).to(device)
                y = xc.to(device), xn.to(device), y.to(device)
                output = model(x)
                val_loss += criterion(output, y).item()
        val_loss /= len(val_loader)
        
        # 9️⃣ Pruning
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    
    return val_loss




import optuna

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Example: assume input_dim=10, output_dim=1 (regression)
study = optuna.create_study(direction="minimize")

study.optimize(
    lambda t: flexible_mlp_objective(
        trial=t,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        input_dim=10,
        output_dim=1,
        device=device,
        task="regression",
        n_epochs=30
    ),
    n_trials=50
)

print("Best trial hyperparameters:", study.best_trial.params)

"""

Features of this version

Automatic variable-depth MLP
    n_layers and hidden_dims per trial.
Dropout tuning
    Sampled per trial.
Activation function tuning
    ReLU, LeakyReLU, GELU.
Optimizer selection
    Adam, SGD, RMSprop.
Learning rate tuning
    Log-uniform sampled.
Batch size tuning
    Dynamically creates DataLoaders per trial.
Supports regression & classification
    Just switch task="classification".
Pruning
    Stops bad trials early.

"""


# 1️⃣ Extract the best hyperparameters
best_params = study.best_trial.params
print(best_params)

# 2️⃣ Rebuild the model
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Build hidden_dims list from best_params
hidden_dims = [
    best_params[f"n_units_l{i}"] for i in range(best_params["n_layers"])
]

# Rebuild model
model = FlexibleMLP(
    input_dim=10,  # adjust to your dataset
    output_dim=1,  # adjust to your dataset
    hidden_dims=hidden_dims,
    dropout=best_params["dropout"],
    activation=best_params["activation"]
).to(device)


3️⃣ Prepare optimizer

import torch.optim as optim

lr = best_params["lr"]
optimizer_name = best_params["optimizer"]

if optimizer_name == "Adam":
    optimizer = optim.Adam(model.parameters(), lr=lr)
elif optimizer_name == "SGD":
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
elif optimizer_name == "RMSprop":
    optimizer = optim.RMSprop(model.parameters(), lr=lr)

# 4️⃣ Prepare DataLoader
from torch.utils.data import DataLoader, ConcatDataset

batch_size = best_params["batch_size"]

# Combine train + val for final training (optional)
full_dataset = ConcatDataset([train_dataset, val_dataset])
train_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)

# 5️⃣ Train the model

criterion = nn.MSELoss()  # or CrossEntropyLoss for classification
n_epochs = 30  # or more

model.train()
for epoch in range(n_epochs):
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
# ✅ After this, model is fully trained with the best hyperparameters.

# 6️⃣ Inference
model.eval()
with torch.no_grad():
    x_test = torch.tensor(X_test.values, dtype=torch.float32).to(device)
    predictions = model(x_test)

# 7️⃣ Save the trained model 
torch.save(model.state_dict(), "best_model.pth")

# Reload for inference
model = FlexibleMLP(input_dim, output_dim, hidden_dims, dropout, activation).to(device)
model.load_state_dict(torch.load("best_model.pth"))
model.eval()






# --- CORE LOGIC ---
def prepare_data(df: pd.DataFrame, cfg: ConfigSchema) -> tuple[pd.DataFrame, OrdinalEncoder, StandardScaler, StandardScaler]:
    """ A data preparation function:
    - Encodes categorical features
    - Scales numeric features and targets

    Parameters
    ----------
    df: pd.DataFrame
    cfg : a Pydantic Model

    Returns
    -------
    Tuple[pd.DataFrame, OrdinalEncoder, StandardScaler, StandardScaler]
        Processed dataframe, fitted categorical encoder, fitted feature scaler,
        fitted target scaler
    """

    data_cat_cols: list[str] = cfg.data.cat_cols
    data_num_cols: list[str] = cfg.data.num_cols
    data_target_cols: list[str] = cfg.data.target_cols


    cat_encoder: OrdinalEncoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    df[data_cat_cols] = cat_encoder.fit_transform(df[data_cat_cols].astype(str))

    in_scaler: StandardScaler = StandardScaler()
    df[data_num_cols] = in_scaler.fit_transform(df[data_num_cols])

    tar_scaler: StandardScaler = StandardScaler()
    df[data_target_cols] = tar_scaler.fit_transform(df[data_target_cols])

    return df, cat_encoder, in_scaler, tar_scaler


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import joblib
import json
import logging
import optuna
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Any, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from torch.utils.data import Dataset, DataLoader

from config.validation_config import ConfigSchema
from config.model_config import DynamicModel, InputDataset

from utils import find_project_root



PROJECT_ROOT: Path = find_project_root(
        # start=Path(__file__).parent, 
        dirname="moffitt"
        )
# Load and Validate Config
with open(PROJECT_ROOT / "pytorch2/config/config.yaml", "r") as f:
    cfg_dict: dict[str, Any] = yaml.safe_load(f)

cfg: ConfigSchema = ConfigSchema(**cfg_dict)
# update to get the absolute date path
cfg.data.path = PROJECT_ROOT / cfg.data.path


export_path: Path = PROJECT_ROOT / 'pytorch2' / cfg.export.dir
export_path.mkdir(parents=True, exist_ok=True)


# 1. Prepare Data

data_path: Path = cfg.data.path
data_drop_columns: list[str] = cfg.data.drop_columns

df: pd.DataFrame = pd.read_parquet(data_path).drop(columns=data_drop_columns)

    
df['population'] = df['population'].fillna(df['population'].median()).astype('int')
df = df.fillna(df.median(numeric_only=True))

df_proc, cat_encoder, in_scaler, tar_scaler = prepare_data(df, cfg)

# 2. Define Embeddings
emb_sizes: list[tuple[int, int]] = [
        (int(df_proc[col].nunique()), min(50, (int(df_proc[col].nunique()) + 1) // 2)) 
        for col in cfg.data.cat_cols
]

embeddings: nn.ModuleList = nn.ModuleList([nn.Embedding(c, s) for c, s in emb_sizes])
# Total dimension after concatenating all embeddings.
n_emb: int = sum(s for c, s in emb_sizes)


# 3. Setup DataLoaders
train_df, val_df = train_test_split(df_proc, test_size=cfg.training.test_size, random_state=cfg.training.random_state)

train_loader = DataLoader(
        InputDataset(train_df, cfg.data.cat_cols, cfg.data.num_cols, cfg.data.target_cols),
        batch_size=cfg.training.batch_size, shuffle=True
    )

val_loader: DataLoader = DataLoader(
        InputDataset(val_df, cfg.data.cat_cols, cfg.data.num_cols, cfg.data.target_cols), 
        batch_size=cfg.training.batch_size
)

# Input dimension
n_emb: int = sum(s for c, s in emb_sizes)
n_numeric = len(cfg.data.num_cols)
input_dim = n_emb + n_numeric

# Output / target dimension
n_targets = len(cfg.data.target_cols)

num_cols: list[str] = in_scaler.feature_names_in_.tolist()
cat_cols: list[str] = cat_encoder.feature_names_in_.tolist()
target_names: list[str] = tar_scaler.feature_names_in_.tolist()
x_cat: torch.Tensor = torch.tensor(df_proc[cat_cols].values, dtype=torch.long).to(device)
x_num: torch.Tensor = torch.tensor(df_proc[num_cols].values, dtype=torch.float32).to(device)

import optuna

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Example: assume input_dim=10, output_dim=1 (regression)
study = optuna.create_study(direction="minimize")

study.optimize(
    lambda t: flexible_mlp_objective(
        trial=t,
        train_dataset=train_loader,
        val_dataset=val_loader,
        input_dim=input_dim,
        output_dim=n_targets,
        device=device,
        task="regression",
        n_epochs=30,
        max_layers = 5,
        units_range = (32, 512),
        dropout_range = (0.0, 0.5),
        emb_sizes = [(2, 1), (3, 2), (51, 26)],
    ),
    n_trials=50
)

print("Best trial hyperparameters:", study.best_trial.params)
