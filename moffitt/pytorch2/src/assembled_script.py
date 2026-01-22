import argparse
import sys

import pandas as pd
import torch
import torch.nn as nn
import yaml
import joblib
import json
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Any
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

import shap



from pathlib import Path

def find_project_root(
    start: Path | None = None,
    *, # everything after this must be passed as a keyword argument
    markers: tuple[str, ...] = (),
    dirname: str | None = None,
) -> Path:
    """
    Find the project root directory by searching upward in the filesystem.

    Starting from `start` (or the current working directory if not provided),
    this function walks up through parent directories until it finds a
    directory that matches one of the following conditions:

    - Contains at least one file or directory listed in `markers`
      (e.g. "pyproject.toml", ".git")
    - Has a directory name equal to `dirname`

    At least one of `markers` or `dirname` must be provided.

    Parameters
    ----------
    start : Path | None, optional
        The directory to start searching from. If None, the current working
        directory is used. The path is resolved to an absolute path before
        searching.

    markers : tuple[str, ...], keyword-only, optional
        A tuple of file or directory names that indicate the project root.
        If any marker exists in a directory, that directory is considered
        the project root.

    dirname : str | None, keyword-only, optional
        The expected name of the project root directory. If a directory with
        this name is encountered while searching upward, it is returned as
        the project root.

    Returns
    -------
    Path
        The resolved absolute path of the detected project root directory.

    Raises
    ------
    ValueError
        If neither `markers` nor `dirname` is provided.

    FileNotFoundError
        If no matching project root is found before reaching the filesystem
        root.

    Examples
    --------
    Find project root using marker files:

    >>> find_project_root(markers=("pyproject.toml", ".git"))

    Find project root using directory name:

    >>> find_project_root(dirname="my_project")

    Hybrid search (recommended):

    >>> find_project_root(
    ...     start=Path(__file__).parent,
    ...     dirname="my_project",
    ...     markers=("pyproject.toml", ".git")
    ... )
    """
    if not markers and not dirname:
        raise ValueError("Provide at least one of `markers` or `dirname`")

    if start is None:
        start = Path.cwd()

    start = start.resolve()

    for path in [start, *start.parents]:
        if dirname and path.name == dirname:
            return path

        if markers and any((path / m).exists() for m in markers):
            return path

    raise FileNotFoundError(
        f"Project root not found starting from {start} "
        f"(dirname={dirname}, markers={markers})"
    )




import sys 
sys.path.append(str(PROJECT_ROOT / "pytorch2/src"))

from config.model_config import DynamicModel
from config.validation_config import ConfigSchema
from train_suite import prepare_data

# Load and Validate Config
with open(PROJECT_ROOT / "pytorch2/config/config.yaml", "r") as f:
    cfg_dict: dict[str, Any] = yaml.safe_load(f)

cfg: ConfigSchema = ConfigSchema(**cfg_dict)


# --- ASSET LOADING ---

model_path = PROJECT_ROOT / cfg.export.dir

# 1. Load Metadata
with open(model_path / 'metadata.json', 'r') as f:
    meta: dict[str, Any] = json.load(f)
        
# 2. Reconstruct Model Architecture
best_params: dict[str, Any] = meta['best_params']
hidden_dims: list[int] = [best_params[f'n_units_l{i}'] for i in range(best_params['n_layers'])]
        
model: DynamicModel = DynamicModel(
    emb_sizes=meta['emb_sizes'],
    n_numeric=len(meta['num_cols']),
    n_targets=len(meta['target_names']),
    hidden_dims=hidden_dims,
    dropout=best_params['dropout']
)
        
# 3. Load Weights
model.load_state_dict(torch.load(model_path / 'champion_weights.pth', weights_only=True))
model.eval()
        
# 4. Load Scalers and Encoders
in_scaler: StandardScaler = joblib.load(model_path / 'input_scaler.pkl')
tar_scaler: StandardScaler = joblib.load(model_path / 'target_scaler.pkl')
cat_encoder: OrdinalEncoder = joblib.load(model_path / 'label_encoders.pkl')


num_cols: list[str] = in_scaler.feature_names_in_.tolist()
cat_cols: list[str] = cat_encoder.feature_names_in_.tolist()
target_names: list[str] = tar_scaler.feature_names_in_.tolist()



new_df: pd.DataFrame = pd.read_parquet(
    PROJECT_ROOT / "data/model_df.parquet"

)

# preparation
df_proc: pd.DataFrame = new_df.copy()

df_proc[cat_cols] = cat_encoder.transform(df_proc[cat_cols].astype(str))
df_proc[num_cols] = in_scaler.transform(df_proc[num_cols])
    
device: torch.device = next(model.parameters()).device
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x_cat: torch.Tensor = torch.tensor(df_proc[cat_cols].values, dtype=torch.long).to(device)
x_num: torch.Tensor = torch.tensor(df_proc[num_cols].values, dtype=torch.float32).to(device)


# prediction
results: pd.DataFrame = new_df.copy()

model.eval()
model.to(device)
with torch.no_grad():
    preds: np.ndarray = model(x_cat, x_num).cpu().numpy()
    real_preds: np.ndarray = tar_scaler.inverse_transform(preds)
for i, name in enumerate(target_names):
    results[f'{name}_pred'] = real_preds[:, i]

results




# SHAP

df_proc: pd.DataFrame = new_df.copy()

df_proc: pd.DataFrame = new_df.copy().sample(
    n=100,
    random_state=42
)


df_proc[cat_cols] = cat_encoder.transform(df_proc[cat_cols].astype(str))
df_proc[num_cols] = in_scaler.transform(df_proc[num_cols])

device: torch.device = next(model.parameters()).device

x_cat: torch.Tensor = torch.tensor(df_proc[cat_cols].values, dtype=torch.long).to(device)
x_num: torch.Tensor = torch.tensor(df_proc[num_cols].values, dtype=torch.float32).to(device)


# Convert DataFrame to numpy array
X = np.concatenate([x_cat.numpy(), x_num.numpy()], axis=1)

# Wrapper for KernelExplainer - wrap two input tensors into the model so it matches the SHAP syntax 
def model_wrapper(X_numpy):
    x_cat_t = torch.tensor(X_numpy[:, :3], dtype=torch.long)
    x_num_t = torch.tensor(X_numpy[:, 3:], dtype=torch.float)
    model.eval()
    with torch.no_grad():
        preds = model(x_cat_t, x_num_t)
        return preds.cpu().numpy()

# Background dataset (small sample)
X_background = X

# Background must come from the training distribution
# Random sampling (default, usually good enough)
# KMeans centroids (best practice for tabular data)
# from sklearn.cluster import KMeans
# k = 50
# kmeans = KMeans(n_clusters=k, random_state=42)
# kmeans.fit(df_proc)

# background_df = pd.DataFrame(
#     kmeans.cluster_centers_,
#     columns=df_proc.columns #feature_cols
# )



import shap
# Create KernelExplainer
explainer = shap.KernelExplainer(model_wrapper, X_background)

# Compute SHAP values
shap_values = explainer.shap_values(X)


# ==========================================
# 4. AUTOMATED PLOTTING AND SAVING
# ==========================================

all_feature_names = cat_cols + num_cols
# os.makedirs('plots', exist_ok=True)
Path("plots").mkdir(parents=True, exist_ok=True)

for i, name in enumerate(target_names):
    print(f"Generating high-density plot for: {name}...")
    
    # Slice target from 3D array: (samples, features, target)
    # Check if shap_values is a list (typical for KernelExplainer) or array
    if isinstance(shap_values, list):
        target_shap_matrix = shap_values[i]
    else:
        target_shap_matrix = shap_values[:, :, i]

    plt.figure(figsize=(14, 10))
    
    # Generate the Beeswarm dot plot
    shap.summary_plot(
        target_shap_matrix, 
        X, 
        feature_names=all_feature_names,
        plot_type="dot",
        show=False,
        alpha=0.7
    )
    
    plt.title(f"SHAP Analysis: {name.replace('_', ' ').title()}", fontsize=16)
    plt.xlabel("Impact on Model Prediction (SHAP Value)", fontsize=12)
    
    save_path = f'plots/{name}_shap_plot.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()



# Correlation 
data_drop_columns: list[str] = cfg.data.drop_columns
df_proc, cat_encoder, in_scaler, tar_scaler = prepare_data(new_df.drop(columns=data_drop_columns), cfg)

corr_matrix = df_proc.corr()

# 2.1 Plot the Heatmap - all features vs all features
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', center=0)
plt.title("Feature Correlation Matrix")    
save_path = f'plots/all_feature_correlation_plot.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()

# 2.2 Plot the Heatmap - targets vs all features
corr_matrix = corr_matrix.loc[cfg.data.target_cols, ]

plt.figure(figsize=(12, 3))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', center=0)
plt.title("Feature Correlation Matrix")    
save_path = f'plots/rget_feature_correlation_plot.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()




import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor

def compute_vif(df):
    """
    That only works for pure numeric NumPy dtypes.
    WHen you run new_df[numeric_cols].dtypes, If you see: Int64, object or boolean. Those will break VIF.
    """
    X = df.assign(const=1)
    vifs = pd.Series(
        [variance_inflation_factor(X.values, i)
         for i in range(X.shape[1]-1)],
        index=df.columns
    )
    return vifs.sort_values(ascending=False)

new_df: pd.DataFrame = pd.read_parquet(PROJECT_ROOT / "data/model_df.parquet")


new_df["population"] = new_df["population"].astype("int64")
"""
Before	            After
dtype = Int64	    dtype = int64
Supports pd.NA	    Does not support NA
NumPy sees object	NumPy sees int64
"""

numeric_cols = new_df.select_dtypes(include=[np.number]).columns
vifs = compute_vif(new_df[numeric_cols].dropna())
print(vifs)

"""
Rules of thumb:
| VIF      | Interpretation                       |
| ------   | ------------------------------------ |
| 1 - 5    | OK                                   |
| 5 - 10   | Concerning                           |
| >10      | Severe multicollinearity             |
| >100     | Essentially redundant                |
| >1,000   | Near-deterministic linear dependence |


Model	            Drop redundancy?
XGBoost	            ❌ Not required
PyTorch MLP	        ✅ Yes
Linear regression	✅ Absolutely
Elastic Net	        🟡 Less critical

XGBoost does not care about multicollinearity
KEEP redundancy if your goal is pure prediction
DROP redundancy if you care about interpretability
Do not use VIF to eliminate features for XGBoost
Do not standardize tree inputs

Minimum required steps for PyTorch

1. Standardize (embedding or encoding) inputs and targets
2. Regularize - weight_decay (L2), Dropout, Early stopping
3. Reduce feature redundancy - Not as aggressively as linear regression but do not keep multiple versions.
Prediction-focused PyTorch - Keep mild redundancy
Stability-focused PyTorch - Drop macro duplicates entirely


Suggested workflow to choose the better model
1. Evaluate RMSE / MAE / R-squared on hold-out test set
2. Confirm stability across CV folds
3. Check generalization gap (train vs val)
4. Consider efficiency / deployment (size, inference speed)
5. Optionally use ensemble if performance differences are small
6. Optionally check interpretability with SHAP

"""

# Target relevance (statistical signal)
# Check mutual information with each target.

from sklearn.feature_selection import mutual_info_regression

X = new_df.drop(columns=[
    "breast", "lung_and_bronchus", "melanoma_of_the_skin"
])

y = new_df[["breast", "lung_and_bronchus", "melanoma_of_the_skin"]]

mi_scores = {}
for target in y.columns:
    mi_scores[target] = mutual_info_regression(
        pd.get_dummies(X, drop_first=True),
        y[target]
    )

mi_df = pd.DataFrame(
    mi_scores,
    index=pd.get_dummies(X, drop_first=True).columns
)

mi_df.mean(axis=1).sort_values(ascending=False)
# Features with near-zero MI (mutual information) across all targets are safe to drop.









"""
Optuna example that optimizes multi-layer perceptrons using PyTorch with checkpoint.
In this example, we optimize the validation accuracy of fastion product recognition using
PyTorch and FashionMNIST. We optimize the neural network architecture as well as the optimizer configuration. As it is too time consuming to use the whole FashionMNIST dataset, we here use a small subset of it.
Even if the process where the trial is running is killed for some reason, you can restart from previous saved checkpoint using heartbeat.

    $ timeout 20 python pytorch/pytorch_checkpoint.py
    $ python pytorch/pytorch_checkpoint.py


https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_checkpoint.py

"""

import os

import tempfile

import optuna
from optuna.artifacts import download_artifact
from optuna.artifacts import FileSystemArtifactStore
from optuna.artifacts import upload_artifact
from optuna.storages import RetryFailedTrialCallback
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import datasets
from torchvision import transforms


DEVICE = torch.device("cpu")
BATCHSIZE = 128
CLASSES = 10
DIR = os.getcwd()
EPOCHS = 10
LOG_INTERVAL = 10
N_TRAIN_EXAMPLES = BATCHSIZE * 30
N_VALID_EXAMPLES = BATCHSIZE * 10
CHECKPOINT_DIR = "pytorch_checkpoint"

base_path = "./artifacts"
os.makedirs(base_path, exist_ok=True)
artifact_store = FileSystemArtifactStore(base_path=base_path)


def define_model(trial):
    """
    Define a fully connected neural network whose depth, width, and dropout rates are learned by a hyperparameter search.
    """
    # Hidden layers is etween 1-3.
    n_layers = trial.suggest_int("n_layers", 1, 3)
    layers = []
    # Input is flattened (28 × 28 = 784)
    in_features = 28 * 28

    for i in range(n_layers):
        # layer width is optimized per layer, each can have 4-128 neurons
        out_features = trial.suggest_int(f"n_units_l{i}", 4, 128)
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())

        p = trial.suggest_float(f"dropout_l{i}", 0.2, 0.5)
        layers.append(nn.Dropout(p))
        in_features = out_features
    # Final layer maps to CLASSES using softmax
    layers.append(nn.Linear(in_features, CLASSES))
    layers.append(nn.LogSoftmax(dim=1))

    return nn.Sequential(*layers)


def get_mnist():
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(DIR, train=True, download=True, transform=transforms.ToTensor()),
        batch_size=BATCHSIZE,
        shuffle=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(DIR, train=False, transform=transforms.ToTensor()),
        batch_size=BATCHSIZE,
        shuffle=True,
    )
    return train_loader, valid_loader


def objective(trial):
    model = define_model(trial).to(DEVICE)

    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    artifact_id = None
    retry_history = RetryFailedTrialCallback.retry_history(trial)
    for trial_number in reversed(retry_history):
        artifact_id = trial.study.trials[trial_number].user_attrs.get("artifact_id")
        if artifact_id is not None:
            retry_trial_number = trial_number
            break

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, "checkpoint.pt")

        if artifact_id is not None:
            download_artifact(
                artifact_store=artifact_store,
                file_path=checkpoint_path,
                artifact_id=artifact_id,
            )
            checkpoint = torch.load(checkpoint_path)
            epoch_begin = checkpoint["epoch"] + 1

            print(
                f"Loading checkpoint from trial {retry_trial_number}, epoch {checkpoint['epoch']}."
            )

            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            accuracy = checkpoint["accuracy"]
        else:
            epoch_begin = 0

        train_loader, valid_loader = get_mnist()

        for epoch in range(epoch_begin, EPOCHS):
            model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                if batch_idx * BATCHSIZE >= N_TRAIN_EXAMPLES:
                    break
                data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)
                optimizer.zero_grad()
                output = model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()

            model.eval()
            correct = 0
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(valid_loader):
                    if batch_idx * BATCHSIZE >= N_VALID_EXAMPLES:
                        break
                    data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)
                    output = model(data)
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()

            accuracy = correct / min(len(valid_loader.dataset), N_VALID_EXAMPLES)
            trial.report(accuracy, epoch)

            print(f"Saving a checkpoint in epoch {epoch}.")

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "accuracy": accuracy,
                },
                checkpoint_path,
            )

            artifact_id = upload_artifact(
                artifact_store=artifact_store,
                file_path=checkpoint_path,
                study_or_trial=trial,
            )
            trial.set_user_attr("artifact_id", artifact_id)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    return accuracy


if __name__ == "__main__":
    storage = optuna.storages.RDBStorage(
        "sqlite:///example.db",
        heartbeat_interval=1,
        failed_trial_callback=RetryFailedTrialCallback(),
    )
    study = optuna.create_study(
        storage=storage,
        study_name="pytorch_checkpoint",
        direction="maximize",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=10, timeout=600)

    pruned_trials = study.get_trials(states=(optuna.trial.TrialState.PRUNED,))
    complete_trials = study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")


    optuna.visualization.plot_intermediate_values(study).show()