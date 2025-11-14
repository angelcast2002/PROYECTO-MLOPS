"""Wrapper: re-exporta API desde paquete independiente proyecto_modeling."""
from proyecto_modeling import (
    make_classification_pipeline,
    train_model,
    cross_validate_model,
    save_model,
    load_model,
    log_experiment,
    hyperparameter_sweep,
)

__all__ = [
    "make_classification_pipeline",
    "train_model",
    "cross_validate_model",
    "save_model",
    "load_model",
    "log_experiment",
    "hyperparameter_sweep",
]
