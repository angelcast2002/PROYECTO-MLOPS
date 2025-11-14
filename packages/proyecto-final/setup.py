# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

setup(
    name="proyecto-final",
    version="0.0.1",
    description="Full CRISP-DM pipeline aggregator (depends on all phase packages)",
    author="Angel Castellanos, Alejandro Azurdia, Diego Morales",
    packages=find_packages(),
    python_requires=">=3.8",
        install_requires=[
            "proyecto-core==0.0.1",
            "proyecto-bu==0.0.1",
            "proyecto-du==0.0.1",
            "proyecto-dp==0.0.1",
            "proyecto-modeling==0.0.1",
            "proyecto-eval==0.0.1",
            "proyecto-deploy==0.0.1",
            "typer[all]>=0.9.0",
        ],
    entry_points={
        "console_scripts": [
            "proyecto-final=proyecto_final.cli:main",
        ]
    }
)
