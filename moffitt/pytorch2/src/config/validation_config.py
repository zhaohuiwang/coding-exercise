
from pydantic import BaseModel, Field, field_validator
from typing import Annotated



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
    # n_trials: int = Field(..., ge=1, description="Number of Optuna trials")
    # n_epochs_per_trial: int = Field(..., ge=1, description="Epochs per trial")

    # layer_range: tuple[int, int] = Field(
    #     ..., description="Min/max number of layers"
    # )
    # units_list: list[int] = Field(
    #     ..., description="List hidden units per layer"
    # )
    # dropout_range: tuple[float, float] = Field(
    #     ..., description="Min/max dropout rate"
    # )
    # lr_range: tuple[float, float] = Field(
    #     ..., description="Min/max learning rate"
    # )

    # Alternative using Annotated from typing - Behavior identical, Preferred! 
    n_trials: Annotated[int, Field(..., ge=1, description="Number of Optuna trials")]
    n_epochs_per_trial: Annotated[int, Field(..., ge=1, description="Epochs per trial")]
    layer_range: Annotated[tuple[int, int], Field(..., description="Min/max number of layers")]

    units_list: Annotated[list[int], Field(..., description="List of hidden units per layer")]

    dropout_range: Annotated[tuple[float, float], Field(..., description="Min/max dropout rate")]
    lr_range: Annotated[tuple[float, float], Field(..., description="Min/max learning rate")]

   
    


    # Note: Field is appropriate for simple type & bounds checks (ge, le), metadata (description), and required values (...). For custom logic, cross-element validation, and complex rules (lo < hi, positive ranges, etc.), we need @field_validator().

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


"""
# Pydantic: Takes untrusted data (JSON, dicts, env vars, API input); Reads your type hints; Validates, coerces, or rejects data at runtime.

Pydantic is 100% runtime.
Typing works only if you run a type checker (or your IDE runs one for you). Python itself does not enforce types.
Typing = promises; Pydantic = enforcement

# Options: organize the constraints using Annotated from typing
class OptunaConfig(BaseModel):

Field() is a function that lets you attach extra metadata and validation rules to a field — things that you cannot express just by writing the type annotation.
Field(...) or Field(default=...) the three dots ... (Ellipsis) indicates that the field is required and has no default value. The value must be provided when creating an instance of the model.

Annotated[BaseType, metadata, more_metadata, ...], e.g. Annotated[str, Field(...)] is used to combine type hints with Field(...) to provide rich metadata and validation rules for Pydantic model fields. where type checkers will just see the base type (str in this case) and ignore the metadata. Pydantic, however, will process the metadata to enforce validation.

# General Project Structure


├── config
│   └── config.yaml
├── ConfigSchema
    └── validation_config.py


# Load and Validate Config
with open("config/config.yaml", "r") as file:
    try:
        cfg_dict: dict[str, Any] = yaml.safe_load(f)
        cfg: ConfigSchema = ConfigSchema(**cfg_dict) # Manual instantiation
        print(cfg.data.host)
    except ValidationError as e:
        print("Invalid configuration file", e.json())

# Use validated config anywhere in the script, e.g.
data_path = cfg.data.path

"""