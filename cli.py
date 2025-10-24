# cli.py
# -*- coding: utf-8 -*-

import os
import json
from datetime import datetime
from typing import Optional, List

import typer

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from pipeline import (
    # rutas/helpers
    ensure_dirs, set_seeds, save_json, load_json,
    DATA_RAW_CSV, PROCESSED_PARQUET, PROCESSED_DIR, DOCS_DIR, MODELS_DIR,
    MODEL_REGISTRY, REGISTRY_DATA_DIR,

    # ingesta + esquema + versionado
    load_dataset, make_data_schema_from_df, validate_dataset_against_schema,
    register_dataset, ingest_new_csv,

    # preprocesamiento
    load_or_preprocess,

    # representación + ngramas (expuestos por si los quieres invocar)
    ngram_sweep,

    # modelado
    make_pipe, train_and_eval_holdout, cross_val_summary, measure_latency,

    # registro modelos/experimentos
    log_experiment, register_model,

    # predicción / catálogo / sweep HP
    predict_csv, write_features_catalog, hp_sweep,

    # metas/SLA
    write_goals, assert_goals,

    # monitoreo
    drift_report_and_flag,
)

app = typer.Typer(help="CLI del proyecto NLP + MLOps (español)")

# ---------------------------------------------------------------------
# Utilidades locales
# ---------------------------------------------------------------------
def _latest_model_path() -> str:
    if not os.path.exists(MODELS_DIR):
        raise typer.Exit("No existe la carpeta 'models/'. Entrena primero.")
    cands = [os.path.join(MODELS_DIR, p) for p in os.listdir(MODELS_DIR) if p.endswith(".joblib")]
    if not cands:
        raise typer.Exit("No hay modelos .joblib en 'models/'. Entrena primero.")
    return max(cands, key=os.path.getmtime)

def _save_confusion(cm: np.ndarray, labels: List[str], path: str):
    from sklearn.metrics import ConfusionMatrixDisplay
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(6,6))
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
    plt.title("Matriz de confusión (holdout)")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout(); plt.savefig(path, dpi=160); plt.close(fig)

# ---------------------------------------------------------------------
# Comandos
# ---------------------------------------------------------------------

@app.command("init-schema")
def init_schema(csv_path: str = typer.Option(DATA_RAW_CSV, help="CSV base con columnas text,label"),
                out_path: str = typer.Option(os.path.join(DOCS_DIR, "data_schema.json"),
                                             help="Ruta donde guardar el esquema")):
    """
    Genera un **contrato de datos** (esquema) a partir del dataset actual.
    """
    ensure_dirs()
    df = load_dataset(csv_path)
    schema = make_data_schema_from_df(df)
    save_json(schema, out_path)
    typer.echo(f"OK → esquema guardado en {out_path}")

@app.command("validate-data")
def validate_data(csv_path: str = typer.Option(DATA_RAW_CSV, help="CSV a validar"),
                  schema_path: str = typer.Option(os.path.join(DOCS_DIR, "data_schema.json")),
                  report_path: str = typer.Option(os.path.join(PROCESSED_DIR, "data_validation_report.json"))):
    """
    Valida el CSV contra el esquema y guarda un reporte JSON.
    """
    ensure_dirs()
    df = load_dataset(csv_path)
    schema = load_json(schema_path)
    report = validate_dataset_against_schema(df, schema)
    save_json(report, report_path)
    if report["errors"]:
        typer.echo(f"⚠️  Validación con ERRORES → {report_path}")
        for e in report["errors"]:
            typer.echo(f" - {e}")
        raise typer.Exit(code=1)
    else:
        typer.echo(f"OK: sin errores → {report_path}")

@app.command("register-dataset")
def register_dataset_cmd(note: str = typer.Option("base", help="Nota/etiqueta de esta versión"),
                         csv_path: str = typer.Option(DATA_RAW_CSV)):
    """
    Crea un snapshot versionado del dataset y actualiza el registro.
    """
    ensure_dirs()
    meta = register_dataset(src_csv=csv_path, note=note)
    typer.echo(f"OK → dataset registrado v{meta['version']} ({meta['rows']} filas)")

@app.command("ingest")
def ingest_cmd(new_csv: str = typer.Argument(..., help="Ruta al nuevo CSV con text,label"),
               note: str = typer.Option("nuevo", help="Nota para el registro")):
    """
    Valida un **nuevo** CSV con el esquema y, si pasa, lo instala como dataset activo
    y lo registra como nueva versión.
    """
    ensure_dirs()
    meta = ingest_new_csv(new_csv, note=note)
    typer.echo(f"OK → dataset activo actualizado y registrado v{meta['version']}")

@app.command("preprocess")
def preprocess_cmd(use_spacy: bool = typer.Option(False, help="Usar spaCy para lematización (lento)")):
    """
    Carga/Preprocesa y guarda parquet en data/processed/preprocesado.parquet.
    """
    ensure_dirs()
    df = load_or_preprocess(use_spacy=use_spacy)
    typer.echo(f"OK → preprocesado: {df.shape} guardado en {PROCESSED_PARQUET}")

@app.command("train-cv")
def train_cv(cv: int = typer.Option(5, help="Número de folds"),
             min_df: int = typer.Option(2),
             C: float = typer.Option(1.0),
             out_path: str = typer.Option(os.path.join(PROCESSED_DIR, "cv_summary.json"))):
    """
    Entrena con validación cruzada y guarda métricas agregadas.
    """
    ensure_dirs()
    df = load_or_preprocess()
    texts = df["text_norm"].astype(str).tolist()
    labels = df["label"].astype(str).tolist()
    summary = cross_val_summary(texts, labels, cv=cv, min_df=min_df, C=C)
    save_json(summary, out_path)
    typer.echo(f"OK → CV guardado en {out_path}")
    typer.echo(f"F1-macro(mean)={summary['f1_macro_mean']:.4f} ± {summary['f1_macro_std']:.4f}")

@app.command("train-holdout")
def train_holdout(min_df: int = typer.Option(2),
                  C: float = typer.Option(1.0),
                  fig_path: str = typer.Option("figures/confusion_holdout.png"),
                  out_path: str = typer.Option(os.path.join(PROCESSED_DIR, "holdout_summary.json"))):
    """
    Split 80/20, entrena, evalúa y guarda matriz de confusión (PNG).
    """
    ensure_dirs()
    df = load_or_preprocess()
    texts = df["text_norm"].astype(str).tolist()
    labels = df["label"].astype(str).tolist()
    pipe, res = train_and_eval_holdout(texts, labels, min_df=min_df, C=C)
    save_json({"accuracy": res["accuracy"], "f1_macro": res["f1_macro"]}, out_path)
    _save_confusion(np.array(res["cm"]), res["labels"], fig_path)
    typer.echo(f"OK → holdout: acc={res['accuracy']:.4f}  f1_macro={res['f1_macro']:.4f}")
    typer.echo(f"PNG: {fig_path}")

@app.command("train-final")
def train_final(min_df: int = typer.Option(2),
                C: float = typer.Option(1.0),
                save_as: str = typer.Option(os.path.join(MODELS_DIR, "svm_tfidf.joblib"),
                                            help="Ruta del modelo 'de trabajo'"),
                also_register: bool = typer.Option(True, help="Guardar además versión en el registry")):
    """
    Entrena con TODO el dataset y guarda el modelo. (Opcional) Registra versión.
    """
    ensure_dirs()
    df = load_or_preprocess()
    texts = df["text_norm"].astype(str).tolist()
    labels = df["label"].astype(str).tolist()
    pipe = make_pipe(min_df=min_df, C=C).fit(texts, labels)
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(pipe, save_as)
    typer.echo(f"OK → modelo guardado en {save_as}")
    if also_register:
        vpath = register_model(pipe, base_name="svm_tfidf")
        typer.echo(f"OK → registrado en {vpath}")

@app.command("latency")
def latency_cmd():
    """
    Mide latencia p50/p95 con textos cortos (SLA demo).
    """
    ensure_dirs()
    # Carga modelo reciente o entrena rápido
    try:
        model_path = _latest_model_path()
        pipe = joblib.load(model_path)
    except Exception:
        df = load_or_preprocess()
        texts = df["text_norm"].astype(str).tolist()
        labels = df["label"].astype(str).tolist()
        pipe = make_pipe().fit(texts, labels)
    ejemplos = [
        "El gobierno anunció nuevas medidas económicas para el próximo trimestre.",
        "Resultados financieros positivos impulsan a la compañía en los mercados.",
        "El telescopio registró una nueva explosión solar visible desde Europa.",
    ]
    lat = measure_latency(pipe, ejemplos, repeat_each=70)
    save_json(lat, os.path.join(PROCESSED_DIR, "latency_summary.json"))
    typer.echo(lat)

@app.command("goals-write")
def goals_write(f1_macro_min: float = typer.Option(0.94),
                f1_min_class: float = typer.Option(0.80),
                latencia_p95_ms: float = typer.Option(100),
                path: str = typer.Option(os.path.join(DOCS_DIR, "objetivo_y_slas.json"))):
    """
    Escribe metas y SLA por defecto (ajustables por flags).
    """
    goals = {
        "optimizamos": "F1-macro",
        "metas": {"f1_macro_min": f1_macro_min, "f1_por_clase_min": f1_min_class},
        "sla_demo": {"latencia_p95_ms": latencia_p95_ms}
    }
    write_goals(goals, path)
    typer.echo(f"OK → {path}")

@app.command("goals-check")
def goals_check(path: str = typer.Option(os.path.join(DOCS_DIR, "objetivo_y_slas.json")),
                cv_path: str = typer.Option(os.path.join(PROCESSED_DIR, "cv_summary.json")),
                lat_path: str = typer.Option(os.path.join(PROCESSED_DIR, "latency_summary.json"))):
    """
    Verifica auto: F1-macro, fairness simple y SLA de latencia.
    """
    assert_goals(path, cv_summary_path=cv_path, latency_summary_path=lat_path)
    typer.echo("OK: metas y SLA cumplidos ✅")

@app.command("predict-csv")
def predict_csv_cmd(in_csv: str = typer.Argument(..., help="CSV con columna 'text'"),
                    out_csv: str = typer.Argument(..., help="Ruta de salida con 'pred'"),
                    model_path: Optional[str] = typer.Option(None, help="Modelo .joblib a usar (opcional)")):
    """
    Hace inferencia por lotes sobre un CSV.
    """
    ensure_dirs()
    out, used = predict_csv(in_csv, out_csv, model_path=model_path)
    typer.echo(f"OK → {out}  (modelo: {used})")

@app.command("features-catalog")
def features_catalog_cmd():
    """
    Escribe/actualiza docs/features_catalog.json
    """
    ensure_dirs()
    cat = write_features_catalog()
    typer.echo("OK → docs/features_catalog.json")

@app.command("hp-sweep")
def hp_sweep_cmd():
    """
    Hace un mini-barrido de hiperparámetros (min_df, C) y loguea en exp_log.jsonl
    """
    ensure_dirs()
    df = load_or_preprocess()
    texts = df["text_norm"].astype(str).tolist()
    labels = df["label"].astype(str).tolist()
    grid = [
        {"min_df": 1, "C": 1.0},
        {"min_df": 2, "C": 1.0},
        {"min_df": 2, "C": 0.5},
        {"min_df": 3, "C": 1.0},
    ]
    res = hp_sweep(texts, labels, grid=grid, cv=3)
    typer.echo(res)

@app.command("drift-check")
def drift_check(prod_csv: Optional[str] = typer.Option(None, help="CSV de prod con 'text'"),
                psi_threshold: float = typer.Option(0.2)):
    """
    Calcula drift (PSI) entre distribución de entrenamiento y predicciones recientes.
    Si no se pasa prod_csv, usa un muestreo del dataset.
    """
    ensure_dirs()

    # 1) base: distribución de labels del dataset
    base = pd.read_csv(DATA_RAW_CSV)["label"].astype(str)

    # 2) predicciones recientes
    model_path = _latest_model_path()
    clf = joblib.load(model_path)

    if prod_csv and os.path.exists(prod_csv):
        prod = pd.read_csv(prod_csv)
        preds = pd.Series(clf.predict(prod["text"].astype(str).tolist()))
    else:
        df = pd.read_csv(DATA_RAW_CSV).sample(frac=0.2, random_state=42)
        preds = pd.Series(clf.predict(df["text"].astype(str).tolist()))

    flag = drift_report_and_flag(base_labels=base, pred_labels=preds,
                                 psi_threshold=psi_threshold)
    typer.echo(flag)

@app.command("list-models")
def list_models():
    """
    Muestra el registro de modelos versionados.
    """
    if not os.path.exists(MODEL_REGISTRY):
        typer.echo("No existe registry de modelos.")
        raise typer.Exit()
    reg = load_json(MODEL_REGISTRY)
    for v in reg["versions"]:
        typer.echo(f"v{v['version']} → {v['path']} ({v['created_at']})")

@app.command("list-datasets")
def list_datasets():
    """
    Muestra el registro de datasets versionados.
    """
    reg_path = os.path.join(REGISTRY_DATA_DIR, "registry.json")
    if not os.path.exists(reg_path):
        typer.echo("No existe registry de datasets.")
        raise typer.Exit()
    reg = load_json(reg_path)
    for d in reg["datasets"]:
        typer.echo(f"v{d['version']} → {d['path']} (sha256={d['sha256'][:10]}...)")

# Entrada
if __name__ == "__main__":
    set_seeds(42)
    ensure_dirs()
    app()
