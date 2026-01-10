from predict import load_inference_assets, generate_model_prediction
import pandas as pd

# 1. Load the "Unified" assets
model, in_scaler, tar_scaler, cat_encoder = load_inference_assets("model_export")

# 2. Load your rawnew data
raw_data = pd.read_parquet("../../data/model_df.parquet")

# 3. Predict with 95% Confidence Intervals (MC Dropout)
# This internally uses your scalers and encoder
final_forecast = generate_model_prediction(
    new_df=raw_data,
    model=model,
    in_scaler=in_scaler,
    tar_scaler=tar_scaler,
    cat_encoder=cat_encoder,
    ci_iterations=100
)

print(final_forecast[['breast_pred', 'breast_lower_95', 'breast_upper_95']].head())
