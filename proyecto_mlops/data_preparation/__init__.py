"""Wrapper: re-exporta API desde paquete independiente proyecto_dp."""
from proyecto_dp import (
    normalize_text,
    tokenize_simple,
    clean_tokens,
    stem_tokens,
    preprocess_dataframe,
    save_preprocessed_data,
    load_preprocessed_data,
    prepare_data_pipeline,
    get_preprocessing_config,
)

__all__ = [
    "normalize_text",
    "tokenize_simple",
    "clean_tokens",
    "stem_tokens",
    "preprocess_dataframe",
    "save_preprocessed_data",
    "load_preprocessed_data",
    "prepare_data_pipeline",
    "get_preprocessing_config",
]
