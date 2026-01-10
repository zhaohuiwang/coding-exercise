
from pydantic import BaseModel, Field
from typing import Any, Optional


# --- CONFIG VALIDATION SCHEMA ---
class DataConfig(BaseModel):
    path: str
    drop_columns: list[str]
    cat_cols: list[str]
    date_cols: list[str]
    num_cols: list[str]
    target_cols: list[str]

class TrainConfig(BaseModel):
    test_size: float = Field(gt=0, lt=1)
    random_state: int
    batch_size: int
    max_epochs: int
    patience: int

class ConfigSchema(BaseModel):
    data: DataConfig
    training: TrainConfig
    optuna: dict[str, Any]
    export: dict[str, str]
