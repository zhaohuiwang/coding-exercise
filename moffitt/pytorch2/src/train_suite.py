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

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("training.log"), logging.StreamHandler()]
)
logger: logging.Logger = logging.getLogger(__name__)


# --- CORE LOGIC ---
def prepare_data(cfg: ConfigSchema) -> tuple[pd.DataFrame, OrdinalEncoder, StandardScaler, StandardScaler]:
    """ A data preparation function:
    - Loads a parquet dataset
    - Drops specified columns
    - Fills missing values
    - Encodes categorical features
    - Scales numeric features and targets

    Parameters
    ----------
    cfg : a Pydantic Model

    Returns
    -------
    Tuple[pd.DataFrame, OrdinalEncoder, StandardScaler, StandardScaler]
        Processed dataframe, fitted categorical encoder, fitted feature scaler,
        fitted target scaler
    """

    data_path: Path = cfg.data.path
    data_drop_columns: list[str] = cfg.data.drop_columns
    data_cat_cols: list[str] = cfg.data.cat_cols
    data_num_cols: list[str] = cfg.data.num_cols
    data_target_cols: list[str] = cfg.data.target_cols


    df: pd.DataFrame = pd.read_parquet(data_path).drop(columns=data_drop_columns)
    
    df['population'] = df['population'].fillna(df['population'].median()).astype('int')
    df = df.fillna(df.median(numeric_only=True))

    cat_enc: OrdinalEncoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    df[data_cat_cols] = cat_enc.fit_transform(df[data_cat_cols].astype(str))

    in_scaler: StandardScaler = StandardScaler()
    df[data_num_cols] = in_scaler.fit_transform(df[data_num_cols])

    tar_scaler: StandardScaler = StandardScaler()
    df[data_target_cols] = tar_scaler.fit_transform(df[data_target_cols])

    return df, cat_enc, in_scaler, tar_scaler

def objective(trial: optuna.Trial, cfg: ConfigSchema, train_loader: DataLoader, 
              val_loader: DataLoader, emb_sizes: list[tuple[int, int]], device: torch.device) -> float:
    try:
        n_layers: int = trial.suggest_int('n_layers', *cfg.optuna['layer_range'])
        hidden_dims: list[int] = [
            trial.suggest_int(f'n_units_l{i}', *cfg.optuna['units_range'], log=True) 
            for i in range(n_layers)
        ]
        dropout: float = trial.suggest_float('dropout', *cfg.optuna['dropout_range'])
        lr: float = trial.suggest_float('lr', *cfg.optuna['lr_range'], log=True)

        model: nn.Module = DynamicModel(
            emb_sizes, len(cfg.data.num_cols), len(cfg.data.target_cols), hidden_dims, dropout
        ).to(device)
        
        optimizer: optim.Optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion: nn.MSELoss = nn.MSELoss()

        for _ in range(cfg.optuna['n_epochs_per_trial']):
            model.train()
            for xc, xn, y in train_loader:
                xc, xn, y = xc.to(device), xn.to(device), y.to(device)
                optimizer.zero_grad()
                criterion(model(xc, xn), y).backward()
                optimizer.step()

        model.eval()
        v_loss: float = 0.0
        with torch.no_grad():
            for xc, xn, y in val_loader:
                v_loss += criterion(model(xc.to(device), xn.to(device)), y.to(device)).item()
        return v_loss / len(val_loader)
    except Exception as e:
        raise optuna.exceptions.TrialPruned()

def main() -> None:

    PROJECT_ROOT: Path = find_project_root(
        # start=Path(__file__).parent, 
        dirname="moffitt"
        )

    logger.info(f"Project root: {PROJECT_ROOT}")

    # Load and Validate Config
    with open(PROJECT_ROOT / "pytorch2/config/config.yaml", "r") as f:
        cfg_dict: dict[str, Any] = yaml.safe_load(f)

    cfg: ConfigSchema = ConfigSchema(**cfg_dict)
    # update to get the absolute date path
    cfg.data.path = PROJECT_ROOT / cfg.data.path


    export_path: Path = PROJECT_ROOT / 'pytorch2' / cfg.export['dir']
    export_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Export path is set as: {export_path}/")

    # 1. Prepare Data
    df_proc, cat_enc, in_scaler, tar_scaler = prepare_data(cfg)

    # 2. Define Embeddings
    emb_sizes: list[tuple[int, int]] = [
        (int(df_proc[col].nunique()), min(50, (int(df_proc[col].nunique()) + 1) // 2)) 
        for col in cfg.data.cat_cols
    ]

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

    # 4. Run Optuna Optimization
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    study: optuna.Study = optuna.create_study(direction='minimize')
    study.optimize(lambda t: objective(t, cfg, train_loader, val_loader, emb_sizes, device), n_trials=cfg.optuna['n_trials'])

    logger.info(f"Optimization finished. Best Params: {study.best_params}")

    # 5. Initialize Champion Model with Best Params
    logger.info("Training final Champion Model with best hyperparameters...")
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
        
    logger.info(f"Successfully exported all artifacts to {export_path}/")



if __name__ == "__main__":
    main()