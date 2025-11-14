"""Wrapper: re-exporta API desde paquete independiente proyecto_bu."""
from proyecto_bu import (
    define_business_objectives,
    save_business_document,
    load_business_document,
)

__all__ = [
    "define_business_objectives",
    "save_business_document",
    "load_business_document",
]
