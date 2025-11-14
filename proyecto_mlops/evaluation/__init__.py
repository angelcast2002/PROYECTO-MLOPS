"""Wrapper: re-exporta API desde paquete independiente proyecto_eval."""
from proyecto_eval import (
    evaluate_model,
    measure_latency,
    check_fairness,
    full_evaluation,
    save_evaluation_report,
)

__all__ = [
    "evaluate_model",
    "measure_latency",
    "check_fairness",
    "full_evaluation",
    "save_evaluation_report",
]
