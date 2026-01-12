
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

PROJECT_ROOT: Path = find_project_root(
        # start=Path(__file__).parent, 
        dirname="moffitt"
        )


import yaml

yaml_path = PROJECT_ROOT / "pytorch2" / "config/config.yaml"
with open(yaml_path, 'r') as file:
    try:
        data_dict = yaml.safe_load(file)
        print(data_dict)
    except yaml.YAMLError as exc:
        print(exc)



import sys
sys.path.append(str(PROJECT_ROOT / "pytorch2" / "src"))


from config.validation_config import DataConfig, TrainConfig, ConfigSchema

from config.model_config import InputDataset, DynamicModel

from train_suite import objective, prepare_data

from predict import load_inference_assets, generate_model_prediction



# Load and Validate Config
with open(PROJECT_ROOT / "pytorch2/config/config.yaml", "r") as f:
        cfg_dict: dict[str, Any] = yaml.safe_load(f)

cfg: ConfigSchema = ConfigSchema(**cfg_dict)
# update to get the absolute date path
cfg.data.path = PROJECT_ROOT / cfg.data.path


export_path: Path = PROJECT_ROOT / 'pytorch2' / cfg.export['dir']
export_path.mkdir(parents=True, exist_ok=True)

# 1. Prepare Data
data_path: Path = cfg.data.path
data_drop_columns: list[str] = cfg.data.drop_columns

df: pd.DataFrame = pd.read_parquet(data_path).drop(columns=data_drop_columns)

    
df['population'] = df['population'].fillna(df['population'].median()).astype('int')
df = df.fillna(df.median(numeric_only=True))
df_proc, cat_enc, in_scaler, tar_scaler = prepare_data(df, cfg)

# 2. Define Embeddings
emb_sizes: list[tuple[int, int]] = [
        (int(df_proc[col].nunique()), min(50, (int(df_proc[col].nunique()) + 1) // 2)) 
        for col in cfg.data.cat_cols
]

# 3. Setup DataLoaders
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

train_df, val_df = train_test_split(df_proc, test_size=cfg.training.test_size, random_state=cfg.training.random_state)

train_loader = DataLoader(
        InputDataset(train_df, cfg.data.cat_cols, cfg.data.num_cols, cfg.data.target_cols),
        batch_size=cfg.training.batch_size, shuffle=True
        )

val_loader: DataLoader = DataLoader(
        InputDataset(val_df, cfg.data.cat_cols, cfg.data.num_cols, cfg.data.target_cols), 
        batch_size=cfg.training.batch_size
    )

# 4. Run Optuna Optimization
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
study: optuna.Study = optuna.create_study(direction='minimize')
study.optimize(lambda t: objective(t, cfg, train_loader, val_loader, emb_sizes, device), n_trials=cfg.optuna['n_trials'])


# 5. Initialize Champion Model with Best Params
    
best_params = study.best_params
best_hidden_dims: list[int] = [best_params[f'n_units_l{i}'] for i in range(best_params['n_layers'])]
    
champion_model: nn.Module = DynamicModel(
    emb_sizes=emb_sizes,
    n_numeric=len(cfg.data.num_cols),
    n_targets=len(cfg.data.target_cols),
    hidden_dims=best_hidden_dims,
    dropout=best_params['dropout']
).to(device)

# 6. Final Export
# Note: For a true production run, you might want to retrain for a few epochs here

# 6.1. Save Torch Weights
torch.save(champion_model.state_dict(), export_path / cfg.export['weights'])
    
# 6.2. Save Sklearn Objects (Scalers & Encoder)
joblib.dump(in_scaler, export_path / cfg.export['in_scaler'])
joblib.dump(tar_scaler, export_path / cfg.export['tar_scaler'])
joblib.dump(cat_enc, export_path / cfg.export['cat_encoder'])

# 6.3. Save Metadata (To reconstruct architecture during inference)
metadata: dict[str, Any] = {
    "best_params": best_params,
    "emb_sizes": emb_sizes,
    "num_cols": cfg.data.num_cols,
    "cat_cols": cfg.data.cat_cols,
    "target_names": cfg.data.target_cols
}
    
with open(export_path / cfg.export['metadata'], 'w') as f:
    json.dump(metadata, f, indent=4)



# predict.py

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

# from config.validation_config import ConfigSchema
# from config.model_config import DynamicModel, InputDataset

# from utils import find_project_root

# --- CUSTOM EXCEPTIONS ---
class InferenceError(Exception):
    """Base for inference exceptions."""
class MissingColumnError(InferenceError):
    """Missing required columns."""
class AssetLoadError(InferenceError):
    """Assets (pkl/pth) missing."""






    
model_dir = PROJECT_ROOT / 'pytorch2' / cfg.export['dir']

# parser = argparse.ArgumentParser(
#     description="Run 2026 Cancer Prediction Inference Engine.",
#     formatter_class=argparse.ArgumentDefaultsHelpFormatter
# )
    
# # Path Arguments
# parser.add_argument(
#     "--input", "-i", 
#         type=str, 
#         required=True, 
#         help="Path to the raw input parquet file."
#     )
# parser.add_argument(
#         "--output", "-o", 
#         type=str, 
#         default="forecasts/predictions.csv", 
#         help="Path where the results will be saved."
#     )
# parser.add_argument(
#         "--model_dir", "-m", 
#         type=str, 
#         default="model_export", 
#         help="Directory containing the model artifacts and metadata."
#     )

# parser.add_argument("--iterations", "-it", type=int, default=None)

# args = parser.parse_args()

# Execute the forecast with CLI arguments
try:
    model, in_scaler, tar_scaler, cat_encoder = load_inference_assets(model_dir)
            
    # new_data: pd.DataFrame = pd.read_parquet(args.input)
    new_data: pd.DataFrame = pd.read_parquet(PROJECT_ROOT / "data/model_df.parquet")
            
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    results_df: pd.DataFrame = generate_model_prediction(
            new_df=new_data,
            model=model,
            in_scaler=in_scaler,
            tar_scaler=tar_scaler,
            cat_encoder=cat_encoder,
            # ci_iterations=args.iterations
            ci_iterations=None

    )
            
    # output_path: Path = PROJECT_ROOT / args.output
    # output_path.parent.mkdir(parents=True, exist_ok=True)
    # results_df.to_csv(output_path, index=False)

except Exception as e:
    sys.exit(1)




# Generating the Correlation Heatmap
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Combine numeric and categorical columns for the analysis
# We use df_proc which contains the encoded values

df_proc, cat_enc, in_scaler, tar_scaler = prepare_data(df, cfg)

corr_matrix = df_proc.corr()

# 2.1 Plot the Heatmap - all features vs all features
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', center=0)
plt.title("Feature Correlation Matrix")    
save_path = f'all_feature_correlation_plot.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()

# 2.2 Plot the Heatmap - targets vs all features
corr_matrix = corr_matrix.loc[cfg.data.target_cols, ]

plt.figure(figsize=(12, 3))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', center=0)
plt.title("Feature Correlation Matrix")    
save_path = f'target_feature_correlation_plot.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()



import torch

# Background must come from the training distribution
# Random sampling (default, usually good enough)
from sklearn.cluster import KMeans
background_df = df_proc.sample(
    n=100,
    random_state=42
)

# # KMeans centroids (best practice for tabular data)
# from sklearn.cluster import KMeans
# k = 50
# kmeans = KMeans(n_clusters=k, random_state=42)
# kmeans.fit(df_proc[feature_cols])

# background_df = pd.DataFrame(
#     kmeans.cluster_centers_,
#     columns=feature_cols
# )


# ============================================================
# SHAP for DynamicModel (PyTorch Tabular with Embeddings)
# ============================================================

"""
Even though df.info() shows float64, Pandas will still produce a
numpy.object_ array if any column in num_cols contains:

mixed types (e.g. numbers + strings)
hidden strings like "NA", " "
None
values loaded from CSV as text
integers mixed with floats in certain edge cases

Best practice (recommended for tabular NN)
def prepare_numeric(df, cols):
    df = df.copy()
    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
    df[cols] = df[cols].fillna(df[cols].mean())
    return df

df = prepare_numeric(df, num_cols)
"""


# ============================================================

cat_cols = cfg.data.cat_cols

num_cols = cfg.data.num_cols

target_cols = cfg.data.target_cols



model, in_scaler, tar_scaler, cat_encoder = load_inference_assets(model_dir)
            
# new_data: pd.DataFrame = pd.read_parquet(args.input)
new_df: pd.DataFrame = pd.read_parquet(PROJECT_ROOT / "data/model_df.parquet")
            
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)




num_cols: list[str] = in_scaler.feature_names_in_.tolist()
cat_cols: list[str] = cat_encoder.feature_names_in_.tolist()
target_names: list[str] = tar_scaler.feature_names_in_.tolist()
    
df_proc: pd.DataFrame = new_df.copy().sample(
    n=10,
    random_state=42
)
df_proc[cat_cols] = cat_encoder.transform(df_proc[cat_cols].astype(str))
df_proc[num_cols] = in_scaler.transform(df_proc[num_cols])

device: torch.device = next(model.parameters()).device

x_cat: torch.Tensor = torch.tensor(df_proc[cat_cols].values, dtype=torch.long).to(device)
x_num: torch.Tensor = torch.tensor(df_proc[num_cols].values, dtype=torch.float32).to(device)


# Convert DataFrame to numpy array
X = np.concatenate([x_cat.numpy(), x_num.numpy()], axis=1)

# Wrapper for KernelExplainer
def model_wrapper(X_numpy):
    x_cat_t = torch.tensor(X_numpy[:, :3], dtype=torch.long)
    x_num_t = torch.tensor(X_numpy[:, 3:], dtype=torch.float)
    model.eval()
    with torch.no_grad():
        preds = model(x_cat_t, x_num_t)
        return preds.cpu().numpy()

# Background dataset (small sample)
X_background = X

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



   
########################################################

# ==========================================
# 1. PREPARE THE DATA FOR SHAP
# ==========================================
# Number of samples to explain (Density)
num_samples = 150 

# Create unified background (reference) and test (explanation) sets
background_raw = np.concatenate([
    x_cat[:100].cpu().numpy(), 
    x_num[:100].cpu().numpy()
], axis=1)

# Use kmeans to speed up KernelExplainer (summarizes data into 15 centroids)
background_summary = shap.kmeans(background_raw, 15)

# Extract a larger sample for a "dense" beeswarm plot
test_raw_dense = np.concatenate([
    df_proc.iloc[:num_samples][cat_cols].values, 
    df_proc.iloc[:num_samples][num_cols].values
], axis=1).astype(np.float32)

# ==========================================
# 2. DEFINE THE UNIFIED MODEL WRAPPER
# ==========================================
def model_predict_for_shap(combined_input):
    """Wrapper to split combined numpy array back into Categorical and Numeric tensors."""
    if len(combined_input.shape) == 1:
        combined_input = combined_input.reshape(1, -1)
        
    combined_tensor = torch.tensor(combined_input, dtype=torch.float32).to(device)
    
    # Slice: first 3 columns are categorical, remaining 13 are numeric
    x_cat = combined_tensor[:, :3].long()
    x_num = combined_tensor[:, 3:]
    
    model.eval()
    with torch.no_grad():
        preds = model(x_cat, x_num)
    return preds.cpu().numpy()

# ==========================================
# 3. CALCULATE SHAP VALUES
# ==========================================
print(f"Initializing Explainer and calculating SHAP for {num_samples} points...")
explainer = shap.KernelExplainer(model_predict_for_shap, background_summary)

# This will result in an array of shape (Samples, Features, Targets) -> (150, 16, 3)
shap_values_dense = explainer.shap_values(test_raw_dense)

# ==========================================
# 4. AUTOMATED PLOTTING AND SAVING
# ==========================================
#target_names = ['breast_cancer', 'lung_and_bronchus', 'melanoma_of_the_skin']
all_feature_names = cat_cols + num_cols
# os.makedirs('plots', exist_ok=True)
Path("plots").mkdir(parents=True, exist_ok=True)

for i, name in enumerate(target_names):
    print(f"Generating high-density plot for: {name}...")
    
    # Slice target from 3D array: (samples, features, target)
    # Check if shap_values is a list (typical for KernelExplainer) or array
    if isinstance(shap_values_dense, list):
        target_shap_matrix = shap_values_dense[i]
    else:
        target_shap_matrix = shap_values_dense[:, :, i]

    plt.figure(figsize=(14, 10))
    
    # Generate the Beeswarm dot plot
    shap.summary_plot(
        target_shap_matrix, 
        test_raw_dense, 
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

print("Execution Complete. High-density plots are in the 'plots' folder.")