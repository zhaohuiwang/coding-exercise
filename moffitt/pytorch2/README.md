# Multi-Target  Prediction Suite - Using Cancer Data as Example

This project implements a deep learning pipeline for multi-target regression, validation and configuration management.


```
source /mnt/e/zhaohuiwang/dev/venvs/uv-venvs/pytorch/.venv/bin/activate

# or Symlink .venv
rm -rf .venv
ln -s /mnt/e/zhaohuiwang/dev/venvs/uv-venvs/pytorch/.venv

source .venv/bin/activate

```

## 📁 Project Structure
- `pytorch/`: Sandbox
- `pytorch2`: verified Working scripts
- `sklearn`: Sklearn pipeline   
- `pytorch2/config/config.yaml`: Central configuration for hyperparameters and column mappings.
- `pytorch2/src/train_suite.py`: Data validation, hyperparameter tuning (Optuna), and model training.
- `pytorch2/src/predict.py`: Inference script using the unified prediction function.
- `model_export/`: Directory containing trained weights and scalers.
```
   .
├── README.md
├── data
│   ├── cancer_count_by_state_year_agegte25lt45.txt
│   ├── cancer_count_by_state_year_agegte45lt65.txt
│   ├── cancer_count_by_state_year_agegte65.txt
│   ├── cancer_rate_pop_econ_nons_df.parquet
│   ├── cdc_smoking_data.json
│   ├── model_df.csv
│   ├── model_df.parquet
│   └── population_data.db
├── forecasts
├── model_export
├── plots
├── pytorch
├── pytorch2
│   ├── Dockerfile
│   ├── DynamicModel.png
│   ├── README.md
│   ├── docker-compose.yml
│   ├── config
│   │   └── config.yaml
│   ├── forecasts
│   │   └── predictions.csv
│   ├── logging
│   ├── model_export
│   ├── sample_files
│   └── src
│       ├── assembled_script.py
│       ├── predict.py
│       ├── templates.py
│       ├── train_suite.py
│       ├── unified_prediction_function_examples.py
│       ├── utils.py
│       ├── config
│       │   ├── model_config.py
│       │   └── validation_config.py
│       ├── model_export
│       └── plots
└── sklearn
```    

## 🚀 Getting Started

### 1. Configuration
Modify `config.yaml` to define your input features and target columns. This file also controls the Optuna search ranges.

### 2. Training & Optimization
Run the training suite to find the best hyperparameters and export the model:
```bash
.../moffitt/$ python3 -m pytorch2.src.train_suite
```
### 3. Start the Training/Optimization
```bash
docker-compose run trainer
docker-compose run predictor
```

### 4. Prediction
Run the predict to regenrate model outputs. If you like to get a confident interval, specify the number of iterations in the cummand:
```bash
.../moffitt/$ python3 -m pytorch2.src.predict -i data/model_df.parquet -it 200
.../moffitt/$ python3 -m pytorch2.src.predict -i data/model_df.parquet
```