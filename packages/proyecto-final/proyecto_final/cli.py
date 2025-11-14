#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CLI para el pipeline completo basado en los paquetes de fases independientes.
"""

from typing import Optional
import os
import typer
import pandas as pd

from proyecto_core import (
    DATA_PROCESSED_DIR,
    DATA_RAW_CSV,
    MODELS_DIR,
    PROCESSED_PARQUET,
    load_json,
)
from proyecto_bu import save_business_document
from proyecto_du import load_raw_dataset, explore_data, save_data_schema
from proyecto_dp import prepare_data_pipeline
from proyecto_modeling import train_model, cross_validate_model, save_model
from proyecto_eval import full_evaluation, save_evaluation_report
from proyecto_deploy import (
    register_model_in_registry,
    promote_to_production,
    save_deployment_guide,
    get_production_model,
)


app = typer.Typer(
    name="proyecto-final",
    help="🚀 Pipeline CRISP-DM multi-paquete",
    rich_markup_mode="rich",
)


@app.command()
def business():
    typer.echo("[bold blue]INICIANDO BUSINESS UNDERSTANDING[/bold blue]")
    doc = save_business_document()
    typer.echo("[bold green][OK] Documento guardado[/bold green]")
    typer.echo(f"   [INFO] Objetivo: {doc['business_objective']['titulo']}")
    typer.echo(f"   [INFO] Target F1-Macro: {doc['ml_objective']['métricas_éxito']['f1_macro_min']}")


@app.command()
def understand():
    typer.echo("[bold blue]INICIANDO DATA UNDERSTANDING[/bold blue]")
    df = load_raw_dataset()
    typer.echo(f"   [OK] Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
    exploration = explore_data(df)
    typer.echo(f"   [INFO] Etiquetas: {exploration.get('label_distribution', {})}")
    save_data_schema()
    typer.echo("[bold green][OK] Data Understanding completado[/bold green]")


@app.command()
def prepare():
    typer.echo("[bold blue]INICIANDO DATA PREPARATION[/bold blue]")
    prepare_data_pipeline()
    typer.echo("[bold green][OK] Datos preparados guardados en data/processed/[/bold green]")


@app.command()
def train(
    do_cv: bool = typer.Option(True, help="Realizar validación cruzada"),
    do_sweep: bool = typer.Option(False, help="Realizar grid search"),
):
    typer.echo("[bold blue]INICIANDO MODELING[/bold blue]")
    # Cargar datos preparados o crudos
    csv_path = os.path.join(os.path.dirname(PROCESSED_PARQUET), "preprocesado.csv")
    try:
        df_prep = pd.read_csv(csv_path)
        typer.echo("   [INFO] Datos cargados desde CSV procesado")
    except Exception:
        df_prep = pd.read_csv(DATA_RAW_CSV)
        typer.echo("   [INFO] Datos cargados desde CSV (sin procesamiento)")
    texts = df_prep['text' if 'text' in df_prep.columns else 'text_prep'].tolist()
    labels = df_prep['label'].tolist()
    typer.echo(f"   [OK] {len(texts)} muestras cargadas")
    model, metrics = train_model(texts, labels)
    typer.echo(f"   [OK] Modelo entrenado - F1-Macro: {metrics['f1_macro']:.4f}")
    if do_cv:
        typer.echo("   [INFO] Realizando validacion cruzada...")
        cv_metrics = cross_validate_model(texts, labels)
        f1_value = cv_metrics.get('mean_f1_macro') or cv_metrics.get('f1_macro_mean') or cv_metrics.get('f1_macro') or 0.0
        if f1_value:
            typer.echo(f"   [OK] CV F1: {f1_value:.4f}")
    save_model(model)
    typer.echo("[bold green][OK] Modelo guardado en models/[/bold green]")


@app.command()
def evaluate():
    typer.echo("[bold blue]INICIANDO EVALUATION[/bold blue]")
    import joblib
    from sklearn.model_selection import train_test_split
    csv_path = os.path.join(os.path.dirname(PROCESSED_PARQUET), "preprocesado.csv")
    try:
        df_prep = pd.read_csv(csv_path)
    except Exception:
        df_prep = pd.read_csv(DATA_RAW_CSV)
    texts = df_prep['text' if 'text' in df_prep.columns else 'text_prep'].tolist()
    labels = df_prep['label'].tolist()
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42, stratify=labels)
    model_files = [f for f in os.listdir(MODELS_DIR) if f.startswith('svm_tfidf_v') and f.endswith('.joblib')]
    if not model_files:
        typer.echo("   [WARN] No hay modelo entrenado")
        raise typer.Exit(1)
    latest_model = sorted(model_files, key=lambda x: int(x.replace('svm_tfidf_v', '').replace('.joblib', '')))[-1]
    pipe = joblib.load(os.path.join(MODELS_DIR, latest_model))
    typer.echo(f"   [INFO] Modelo cargado: {latest_model}")
    status_report = full_evaluation(pipe, X_test, y_test)
    save_evaluation_report(status_report)
    typer.echo(f"   [OK] Status: {status_report.get('overall_status', 'UNKNOWN')}")
    typer.echo("[bold green][OK] Evaluacion completada[/bold green]")


@app.command()
def deploy(
    promote: bool = typer.Option(False, help="Promover a produccion"),
    version: Optional[int] = typer.Option(None, help="Version a promover"),
):
    typer.echo("[bold blue]INICIANDO DEPLOYMENT[/bold blue]")
    import joblib
    model_files = [f for f in os.listdir(MODELS_DIR) if f.startswith('svm_tfidf_v') and f.endswith('.joblib')]
    if not model_files:
        typer.echo("   [WARN] No hay modelo entrenado para desplegar")
        raise typer.Exit(1)
    latest_model = sorted(model_files, key=lambda x: int(x.replace('svm_tfidf_v', '').replace('.joblib', '')))[-1]
    model_path = os.path.join(MODELS_DIR, latest_model)
    model_version = int(latest_model.replace('svm_tfidf_v', '').replace('.joblib', ''))
    evaluation_report_path = os.path.join(DATA_PROCESSED_DIR, "evaluation_report.json")
    if os.path.exists(evaluation_report_path):
        eval_report = load_json(evaluation_report_path)
        metrics = eval_report.get("metrics", {})
    else:
        metrics = {"status": "not_evaluated"}
    register_model_in_registry(model_path=model_path, model_name="svm_tfidf", version=model_version, metrics=metrics, status="candidate")
    typer.echo("   [OK] Modelo registrado")
    if promote is True and version is not None:
        promote_to_production(version)
        typer.echo(f"   [OK] Modelo v{version} promovido a produccion")
    save_deployment_guide()
    typer.echo("[bold green][OK] Deployment completado[/bold green]")


@app.command()
def all(cv: bool = typer.Option(True, help="Incluir validacion cruzada")):
    typer.echo("[bold cyan]INICIANDO PIPELINE COMPLETO CRISP-DM[/bold cyan]\n")
    business(); typer.echo()
    understand(); typer.echo()
    prepare(); typer.echo()
    train(do_cv=cv); typer.echo()
    evaluate(); typer.echo()
    deploy(); typer.echo()
    typer.echo("[bold green]*** PIPELINE COMPLETO FINALIZADO EXITOSAMENTE ***[/bold green]")


@app.command()
def status():
    typer.echo("[bold blue]Estado Actual[/bold blue]")
    prod_model = get_production_model()
    if prod_model:
        typer.echo(f"   [OK] Modelo en Produccion: v{prod_model['version']}")
        typer.echo(f"   [INFO] Fecha: {prod_model.get('created_at', 'N/A')}")
        typer.echo(f"   [INFO] F1-Macro: {prod_model.get('metrics', {}).get('f1_macro', 'N/A')}")
    else:
        typer.echo("   [WARN] No hay modelo en produccion")


def main():
    app()


if __name__ == "__main__":
    main()
