# -*- coding: utf-8 -*-
"""
Setup configuration for proyecto_mlops package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="proyecto-mlops",
    version="0.1.0",
    author="Angel Castellanos",
    author_email="angelcast2002@gmail.com",
    description="MLOps Pipeline para Clasificación de Documentos en Español",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/angelcast2002/PROYECTO-MLOPS",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.20.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "nltk>=3.8",
        "gensim>=4.0.0",
        "spacy>=3.0.0",
        "typer[all]>=0.9.0",
        "pyyaml>=6.0",
        "joblib>=1.2.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "docker": [
            "docker>=5.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "proyecto-mlops=proyecto_mlops.cli:app",
        ],
    },
    include_package_data=True,
    keywords="mlops machine-learning pipeline classification nlp",
    project_urls={
        "Documentation": "https://github.com/angelcast2002/PROYECTO-MLOPS",
        "Bug Reports": "https://github.com/angelcast2002/PROYECTO-MLOPS/issues",
        "Source Code": "https://github.com/angelcast2002/PROYECTO-MLOPS",
    },
)
