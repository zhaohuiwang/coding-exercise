# Multi-Target  Prediction Suite - Using Cancer Data as Example

This project implements a deep learning pipeline for multi-target regression, validation and configuration management.


```
source /mnt/e/zhaohuiwang/dev/venvs/uv-venvs/pytorch/.venv/bin/activate
```

## 📁 Project Structure
- `config.yaml`: Central configuration for hyperparameters and column mappings.
- `train_suite.py`: Data validation, hyperparameter tuning (Optuna), and model training.
- `predict.py`: Inference script using the unified prediction function.
- `model_export/`: Directory containing trained weights and scalers.

## 🚀 Getting Started

### 1. Configuration
Modify `config.yaml` to define your input features and target columns. This file also controls the Optuna search ranges.

### 2. Training & Optimization
Run the training suite to find the best hyperparameters and export the model:
```bash
...coding-exercise/moffitt/$ python3 pytorch2/src/train_suite.py
```
### 3. Start the Training/Optimization
```bash
docker-compose run trainer
docker-compose run predictor
```

### 4. Prediction
Run the predict to regenrate model outputs. If you like to get a confident interval, specify the number of iterations in the cummand:
```bash
python3 pytorch2/src/predict.py -i data/model_df.parquet -it 200
python3 pytorch2/src/predict.py -i data/model_df.parquet
```

pytorch2/src/predict.py