"""Wrapper: re-exporta API desde paquete independiente proyecto_deploy."""
from proyecto_deploy import (
    register_model_in_registry,
    promote_to_production,
    get_production_model,
    create_deployment_package,
    generate_deployment_guide,
    save_deployment_guide,
)

__all__ = [
    "register_model_in_registry",
    "promote_to_production",
    "get_production_model",
    "create_deployment_package",
    "generate_deployment_guide",
    "save_deployment_guide",
]
