"""Wrapper: re-exporta API desde paquete independiente proyecto_du."""
from proyecto_du import (
    load_raw_dataset,
    explore_data,
    save_data_exploration,
    make_data_schema,
    save_data_schema,
    validate_schema,
)

__all__ = [
    "load_raw_dataset",
    "explore_data",
    "save_data_exploration",
    "make_data_schema",
    "save_data_schema",
    "validate_schema",
]
