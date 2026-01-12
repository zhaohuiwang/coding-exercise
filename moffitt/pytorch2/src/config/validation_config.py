
from pydantic import BaseModel, Field, field_validator
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

class OptunaConfig(BaseModel):
    n_trials: int = Field(..., ge=1, description="Number of Optuna trials")
    n_epochs_per_trial: int = Field(..., ge=1, description="Epochs per trial")

    layer_range: tuple[int, int] = Field(
        ..., description="Min/max number of layers"
    )
    units_list: list[int] = Field(
        ..., description="List hidden units per layer"
    )
    dropout_range: tuple[float, float] = Field(
        ..., description="Min/max dropout rate"
    )
    lr_range: tuple[float, float] = Field(
        ..., description="Min/max learning rate"
    )

    @field_validator("layer_range")
    @classmethod
    def validate_int_ranges(cls, v):
        lo, hi = v
        if lo >= hi:
            raise ValueError("Lower bound must be < upper bound")
        return v

    @field_validator("dropout_range")
    @classmethod
    def validate_dropout(cls, v):
        lo, hi = v
        if not (0.0 <= lo < hi <= 1.0):
            raise ValueError("Dropout range must be within [0, 1]")
        return v

    @field_validator("lr_range")
    @classmethod
    def validate_lr(cls, v):
        lo, hi = v
        if lo <= 0 or hi <= 0 or lo >= hi:
            raise ValueError("Learning rate bounds must be positive and lo < hi")
        return v


class ExportConfig(BaseModel):
    dir: str = Field(..., min_length=1)
    weights: str
    in_scaler: str
    tar_scaler: str
    cat_encoder: str
    metadata: str

class ConfigSchema(BaseModel):
    data: DataConfig
    training: TrainConfig
    optuna: OptunaConfig
    export: ExportConfig

