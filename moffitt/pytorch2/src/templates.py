



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
        #  ModuleList ensures PyTorch tracks these layers
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
    
"""
For categories with less unique values like gender, type, One-Hot Encoding is preferred. When you have a high-cardinality feature like "State" (50+ values), you have moved beyond the scope of simple One-Hot Encoding. You need embeddings.

The "Rule of Thumb" for Embedding Dimensions
$$\text{Embedding Size} = \min(50, \pi(\text{Number of Unique Categories})^{0.25} \text{ or } \frac{\text{Unique Categories}}{2})$$

EmbeddingSize = min(50, pi*(Number of Unique Categories)**0.25
or 
Unique Categories / 2
"""        
from pathlib import Path

class EarlyStopping:
    def __init__(
            self,
            patience: int=7,
            verbose: bool=False,
            delta: float=0,
            save_model: bool=False,
            path: str | Path ='best_model.pth'
            ):
        """
        EarlyStopping if placed to the end of each epoch cycle, monitors val_loss (evaluation loss) in each epoch (or set of epochs through DataLoader). When the creteria defined by patience and delta is met, early_stop is set to True to signal the break in the epoch iteration. save_checkpoint() is optional and maybe removed.   
        Args:
        patience (int): How many epochs with no improvement the model should wait before stopping.
            verbose (bool): If True, prints a message for each validation loss improvement. 
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            save_model (bool): Whether to save the model.
            path (str | Path): Path for the checkpoint to be saved to.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.save_model = save_model
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            if self.save_model:
                self.save_checkpoint(model, val_loss)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.save_model:
                self.save_checkpoint(model, val_loss)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


### Model Training with Early Stopping
# Initialize a model
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DynamicModel().to(device)

# Get data loaders 
train_loader: DataLoader = DataLoader(., batch_size=., shuffle=True)
val_loader: DataLoader = DataLoader(., batch_size=., shuffle=True)


# Setup Optimizer with optimized Learning Rate
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Training with Early Stopping
early_stopping = EarlyStopping(patience=3)
history = {'train_loss': [], 'val_loss': []}

for epoch in range(100):
    model.train()
    train_loss = 0
    # When X has two tensors xc and xn in the same order as in the model class
    for xc, xn, y in train_loader:
        xc, xn, y = xc.to(device), xn.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xc, xn), y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    # Average loss for the entire training epoch
    avg_train_loss = np.mean(train_loss)
    history['train_loss'].append(avg_train_loss)
    
    # Validation Phase
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for xc, xn, y in val_loader:
            xc, xn, y = xc.to(device), xn.to(device), y.to(device)
            val_loss += criterion(model(xc, xn), y).item()
    
    avg_val_loss = np.mean(val_loss)
    history['val_loss'].append(avg_val_loss)

    # Check early stopping (using training loss as proxy for this demo)
    early_stopping(loss.item(), model)
    if early_stopping.early_stop:
        print(f"Early stopped at epoch {epoch}")
        break

import matplotlib.pyplot as plt

def plot_losses(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.title('Training and Validation Loss Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_history.png")
    plt.show()

# plot_losses(history)

# Save after the loop ends
joblib.dump(history, 'training_history.joblib')


# 1️⃣ 2️⃣ 3️⃣ 4️⃣ 5️⃣ 6️⃣ 7️⃣ 8️⃣ 9️⃣

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

    # emb_sizes = [(2, 1), (3, 2), (51, 26)],
):
    # 1. Sample batch size
    # batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])

    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    train_loader = train_dataset
    train_loader = val_dataset
    
    # 2. Sample architecture
    n_layers = trial.suggest_int("n_layers", 1, max_layers) # Number of hidden layers [1, max_layers]
    hidden_dims = [
        trial.suggest_int(f"n_units_l{i}", *units_range, log=True) for i in range(n_layers)
    ] # log=True biases toward smaller values - good practice

    dropout = trial.suggest_float("dropout", *dropout_range)
    
    activation = trial.suggest_categorical("activation", ["relu", "leaky_relu", "gelu"])
    
    # 3. Build model
    # model = FlexibleMLP(
    #     input_dim=input_dim,
    #     output_dim=output_dim,
    #     hidden_dims=hidden_dims,
    #     dropout=dropout,
    #     activation=activation_name,
    # ).to(device)
    model: nn.Module = DynamicModel(
            emb_sizes, len(cfg.data.num_cols), len(cfg.data.target_cols), hidden_dims, dropout
        ).to(device)
    

    # 4. Optimizer
    optimizer_name = trial.suggest_categorical(
        "optimizer", ["Adam", "SGD", "RMSprop"]
    )

    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)

    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    
    # 5. Loss
    criterion = nn.MSELoss() if task == "regression" else nn.CrossEntropyLoss()
    
    # 6. Training loop
    # # Early stopping logic
    # patience = 5
    # best_val = float("inf")
    # epochs_no_improve = 0

    for epoch in range(n_epochs):
        model.train()
        # Matching to the model and dataloader architecture
        for xc, xn, y in train_loader:
            xc, xn, y = xc.to(device), xn.to(device), y.to(device)
            x = torch.cat([xc, xn], dim=1)

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
        
        # 7. Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xc, xn, y in val_loader:
                xc, xn, y = xc.to(device), xn.to(device), y.to(device)
                x = torch.cat([xc, xn], dim=1)

                output = model(x)
                val_loss += criterion(output, y).item()

        val_loss /= len(val_loader)

        # # Early stopping logic
        # if val_loss < best_val:
        #     best_val = val_loss
        #     epochs_no_improve = 0
        # else:
        #     epochs_no_improve += 1
        # if epochs_no_improve >= patience:
        #     break

        
        # 8. Optuna pruning
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_loss



import optuna
# model-agnostic automatic hyperparameter optimization

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Example
study = optuna.create_study(direction="minimize")

# study runs many trials each corresponds to one model configuration.
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
    # Number of different hyperparameter configurations to run
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


# # 1️⃣ Extract the best hyperparameters
# best_params = study.best_trial.params
# print(best_params)

# # 2️⃣ Rebuild the model
# import torch

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Build hidden_dims list from best_params
# hidden_dims = [
#     best_params[f"n_units_l{i}"] for i in range(best_params["n_layers"])
# ]

# # Rebuild model
# model = FlexibleMLP(
#     input_dim=10,  # adjust to your dataset
#     output_dim=1,  # adjust to your dataset
#     hidden_dims=hidden_dims,
#     dropout=best_params["dropout"],
#     activation=best_params["activation"]
# ).to(device)


# 3️⃣ Prepare optimizer

# import torch.optim as optim

# lr = best_params["lr"]
# optimizer_name = best_params["optimizer"]

# if optimizer_name == "Adam":
#     optimizer = optim.Adam(model.parameters(), lr=lr)
# elif optimizer_name == "SGD":
#     optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
# elif optimizer_name == "RMSprop":
#     optimizer = optim.RMSprop(model.parameters(), lr=lr)

# # 4️⃣ Prepare DataLoader
# from torch.utils.data import DataLoader, ConcatDataset

# batch_size = best_params["batch_size"]

# # Combine train + val for final training (optional)
# full_dataset = ConcatDataset([train_dataset, val_dataset])
# train_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)

# # 5️⃣ Train the model

# criterion = nn.MSELoss()  # or CrossEntropyLoss for classification
# n_epochs = 30  # or more

# model.train()
# for epoch in range(n_epochs):
#     for x, y in train_loader:
#         x, y = x.to(device), y.to(device)
#         optimizer.zero_grad()
#         output = model(x)
#         loss = criterion(output, y)
#         loss.backward()
#         optimizer.step()
# # ✅ After this, model is fully trained with the best hyperparameters.

# # 6️⃣ Inference
# model.eval()
# with torch.no_grad():
#     x_test = torch.tensor(X_test.values, dtype=torch.float32).to(device)
#     predictions = model(x_test)

# # 7️⃣ Save the trained model 
# torch.save(model.state_dict(), "best_model.pth")

# # Reload for inference
# model = FlexibleMLP(input_dim, output_dim, hidden_dims, dropout, activation).to(device)
# model.load_state_dict(torch.load("best_model.pth"))
# model.eval()






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

import sys
sys.path.append("pytorch2/src/")
from config.model_config import  InputDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        #emb_sizes = [(2, 1), (3, 2), (51, 26)],
    ),
    n_trials=50
)

print("Best trial hyperparameters:", study.best_trial.params)











###  An Example
# Start>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

# --- 1. THE EARLY STOPPING CLASS ---
class EarlyStopping:
    def __init__(self, patience=5, path='best_model.pt'):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)

# --- 2. THE MODEL ARCHITECTURE ---
class MultiInputModel(nn.Module):
    def __init__(self, encoder_dict, num_numerical_cols, embedding_dim=4):
        super(MultiInputModel, self).__init__()
        # ModuleList ensures PyTorch tracks these layers
        self.embeddings = nn.ModuleList([
            nn.Embedding(len(encoder.classes_), embedding_dim) 
            for encoder in encoder_dict.values()
        ])
        total_input_size = (len(encoder_dict) * embedding_dim) + num_numerical_cols
        self.network = nn.Sequential(
            nn.Linear(total_input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
    def forward(self, cat_tensor, num_tensor):
        emb_outputs = [emb_layer(cat_tensor[:, i]) for i, emb_layer in enumerate(self.embeddings)]
        x = torch.cat(emb_outputs + [num_tensor], dim=1)
        return self.network(x)

# --- 3. DATA GENERATION & PREPROCESSING ---
# Create dummy data
data = {
    'City': ['NYC', 'LA', 'Chicago', 'NYC', 'LA'] * 20,
    'Type': ['A', 'B', 'A', 'C', 'B'] * 20,
    'Temp': np.random.rand(100),
    'Humidity': np.random.rand(100),
    'Target': np.random.rand(100) * 100
}
df = pd.DataFrame(data)

# Process Categorical
cat_cols = ['City', 'Type']
encoder_dict = {col: LabelEncoder().fit(df[col]) for col in cat_cols}
cat_data = np.stack([encoder_dict[col].transform(df[col]) for col in cat_cols], axis=1)

# Process Numerical
in_scaler = MinMaxScaler().fit(df[['Temp', 'Humidity']])
num_data = in_scaler.transform(df[['Temp', 'Humidity']])

# Process Target
tar_scaler = MinMaxScaler().fit(df[['Target']])
target_data = tar_scaler.transform(df[['Target']])

# Save Assets
joblib.dump({'in_scaler': in_scaler, 'tar_scaler': tar_scaler, 'encoder_dict': encoder_dict}, 'assets.joblib')

# Create DataLoader
dataset = TensorDataset(torch.long(cat_data), torch.float32(num_data), torch.float32(target_data))
train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(dataset, batch_size=8, shuffle=True)

# --- 4. TRAINING LOOP ---
model = MultiInputModel(encoder_dict, num_numerical_cols=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()
early_stopping = EarlyStopping(patience=3)
history = {'train_loss': [], 'val_loss': []}


train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(dataset, batch_size=8, shuffle=True)


for epoch in range(100):
    model.train()
    train_loss = 0
    # When X has two tensors xc and xn in the same order as in the model class
    for xc, xn, y in train_loader:
        xc, xn, y = xc.to(device), xn.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xc, xn), y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    # Average loss for the entire training epoch
    avg_train_loss = np.mean(train_loss)
    history['train_loss'].append(avg_train_loss)
    
    # Validation Phase
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for xc, xn, y in val_loader:
            xc, xn, y = xc.to(device), xn.to(device), y.to(device)
            val_loss += criterion(model(xc, xn), y).item()
    
    avg_val_loss = np.mean(val_loss)
    history['val_loss'].append(avg_val_loss)

    # Check early stopping (using training loss as proxy for this demo)
    early_stopping(loss.item(), model)
    if early_stopping.early_stop:
        print(f"Early stopped at epoch {epoch}")
        break

import matplotlib.pyplot as plt

def plot_losses(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.title('Training and Validation Loss Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_history.png")
    plt.show()

plot_losses(history)



# --- 5. THE UNIFIED PREDICTION FUNCTION ---
def unified_prediction(cat_dict, num_list):
    # Load Assets
    assets = joblib.load('assets.joblib')
    # Encode Categories
    cat_idxs = [assets['encoder_dict'][k].transform([v])[0] for k, v in cat_dict.items()]
    cat_t = torch.tensor([cat_idxs], dtype=torch.long)
    # Scale Numerics
    num_s = assets['in_scaler'].transform([num_list])
    num_t = torch.tensor(num_s, dtype=torch.float32)
    # Predict
    model.load_state_dict(torch.load('best_model.pt'))
    model.eval()
    with torch.no_grad():
        out = model(cat_t, num_t)
    # Inverse Target
    return assets['tar_scaler'].inverse_transform(out.numpy())[0][0]

# --- TEST IT ---
test_cat = {'City': 'NYC', 'Type': 'B'}
test_num = [0.5, 0.8]
result = unified_prediction(test_cat, test_num)
print(f"\nFinal Prediction (Real Scale): {result:.2f}")


# Training with Early Stopping
# End <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<