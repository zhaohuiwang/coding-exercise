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

from config.model_config import DynamicModel
from config.validation_config import ConfigSchema

from utils import find_project_root

# --- CUSTOM EXCEPTIONS ---
class InferenceError(Exception):
    """Base for inference exceptions."""
class MissingColumnError(InferenceError):
    """Missing required columns."""
class AssetLoadError(InferenceError):
    """Assets (pkl/pth) missing."""


# --- UNIFIED PREDICTION FUNCTION ---    
def generate_model_prediction(
    new_df: pd.DataFrame, 
    model: torch.nn.Module, 
    in_scaler: StandardScaler, 
    tar_scaler: StandardScaler, 
    cat_encoder: OrdinalEncoder, 
    ci_iterations: int | None = None
) -> pd.DataFrame:
    """
    Unified prediction function [Updated 2026-01-09].
    Incorporates internal scaling and encoding logic for deployment.
    """
    num_cols: list[str] = in_scaler.feature_names_in_.tolist()
    cat_cols: list[str] = cat_encoder.feature_names_in_.tolist()
    target_names: list[str] = tar_scaler.feature_names_in_.tolist()
    
    df_proc: pd.DataFrame = new_df.copy()
    df_proc[cat_cols] = cat_encoder.transform(df_proc[cat_cols].astype(str))
    df_proc[num_cols] = in_scaler.transform(df_proc[num_cols])
    
    device: torch.device = next(model.parameters()).device
    x_cat: torch.Tensor = torch.tensor(df_proc[cat_cols].values, dtype=torch.long).to(device)
    x_num: torch.Tensor = torch.tensor(df_proc[num_cols].values, dtype=torch.float32).to(device)

    results: pd.DataFrame = new_df.copy()

    if not ci_iterations:
        model.eval()
        with torch.no_grad():
            preds: np.ndarray = model(x_cat, x_num).cpu().numpy()
            real_preds: np.ndarray = tar_scaler.inverse_transform(preds)
        for i, name in enumerate(target_names):
            results[f'{name}_pred'] = real_preds[:, i]
    else:
        model.train() # Enable Dropout for MC sampling
        all_samples: list[np.ndarray] = []
        for _ in range(ci_iterations):
            with torch.no_grad():
                sample: np.ndarray = model(x_cat, x_num).cpu().numpy()
                all_samples.append(tar_scaler.inverse_transform(sample))
        
        arr_samples: np.ndarray = np.array(all_samples)
        mean_preds: np.ndarray = np.mean(arr_samples, axis=0)
        lower_bound: np.ndarray = np.percentile(arr_samples, 2.5, axis=0)
        upper_bound: np.ndarray = np.percentile(arr_samples, 97.5, axis=0)
        
        for i, name in enumerate(target_names):
            results[f'{name}_pred'] = mean_preds[:, i]
            results[f'{name}_lower_95'] = lower_bound[:, i]
            results[f'{name}_upper_95'] = upper_bound[:, i]
            
    return results

# --- ASSET LOADING ---
def load_inference_assets(model_dir: str = "model_export") -> tuple[nn.Module, StandardScaler, StandardScaler, OrdinalEncoder]:

    model_path: Path = Path(model_dir)
    
    try:
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
        
        return model, in_scaler, tar_scaler, cat_encoder
    
    except FileNotFoundError as e:
        raise AssetLoadError(f"Required file missing in {model_dir}")
    
    except Exception as e:
        raise AssetLoadError(str(e))

   


def main() -> None:
    """
    Command Line Interface for the Inference Engine.
    """

    PROJECT_ROOT: Path = find_project_root(
        # start=Path(__file__).parent, 
        dirname="moffitt"
        )
    
    # --- LOGGING SETUP ---
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(PROJECT_ROOT / "pytorch2/logging"/"training.log"), logging.StreamHandler()]
    )
    logger: logging.Logger = logging.getLogger(__name__)

    # Load and Validate Config
    with open(PROJECT_ROOT / "pytorch2/config/config.yaml", "r") as f:
        cfg_dict: dict[str, Any] = yaml.safe_load(f)

    cfg: ConfigSchema = ConfigSchema(**cfg_dict)
    
    model_dir = PROJECT_ROOT / 'pytorch2' / cfg.export['dir']

    parser = argparse.ArgumentParser(
        description="Run 2026 Cancer Prediction Inference Engine.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Path Arguments
    parser.add_argument(
        "--input", "-i", 
        type=str, 
        required=True, 
        help="Path to the raw input parquet file."
    )
    parser.add_argument(
        "--output", "-o", 
        type=str, 
        default="forecasts/predictions.csv", 
        help="Path where the results will be saved."
    )
    parser.add_argument(
        "--model_dir", "-m", 
        type=str, 
        default="model_export", 
        help="Directory containing the model artifacts and metadata."
    )

    parser.add_argument("--iterations", "-it", type=int, default=None)

    args = parser.parse_args()

    # Execute the forecast with CLI arguments
    try:
        model, in_scaler, tar_scaler, cat_encoder = load_inference_assets(model_dir)
            
        new_data: pd.DataFrame = pd.read_parquet(args.input)
            
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        results_df: pd.DataFrame = generate_model_prediction(
            new_df=new_data,
            model=model,
            in_scaler=in_scaler,
            tar_scaler=tar_scaler,
            cat_encoder=cat_encoder,
            ci_iterations=args.iterations
        )
            
        output_path: Path = PROJECT_ROOT / args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False)

        logger.critical(f"Prediction saved to: {output_path}")

    except Exception as e:
        logger.critical(f"CLI Execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
