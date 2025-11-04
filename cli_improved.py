"""
ENTRYPOINT CLI MEJORADO
Sistema MLOps para clasificación de documentos
"""

import typer
import os
from pathlib import Path
from typing import Optional

# Import fases CRISP-DM
from proyecto_mlops.business_understanding import save_business_document
from proyecto_mlops.data_understanding import (
    save_data_exploration,
    save_data_schema,
    load_raw_dataset,
    validate_schema
)
from proyecto_mlops.data_preparation import prepare_data_pipeline
from proyecto_mlops.modeling import (
    train_model,
    cross_validate_model,
    save_model,
    log_experiment,
    hyperparameter_sweep
)
from proyecto_mlops.evaluation import (
    full_evaluation,
    save_evaluation_report
)
from proyecto_mlops.deployment import (
    register_model_in_registry,
    promote_to_production,
    save_deployment_guide
)

app = typer.Typer(help="🚀 PROYECTO-MLOPS: Pipeline completo CRISP-DM")


@app.command()
def business(
    output_dir: str = typer.Option("docs", help="Directorio de salida")
):
    """📊 Fase 1: Business Understanding"""
    typer.echo("🎯 Iniciando Business Understanding...")
    doc = save_business_document()
    typer.echo(f"✅ Documento guardado: {output_dir}/business_understanding.json")
    typer.echo(f"   Objetivo: {doc['business_objective']['titulo']}")
    typer.echo(f"   Target F1-Macro: {doc['ml_objective']['métricas_éxito']['f1_macro_min']}")


@app.command()
def understand(
    input_csv: str = typer.Option("data/raw/dataset.csv", help="CSV de entrada"),
    output_dir: str = typer.Option("data/processed", help="Directorio de salida")
):
    """📈 Fase 2: Data Understanding"""
    typer.echo("🔍 Iniciando Data Understanding...")
    
    # Cargar y explorar
    df = load_raw_dataset(input_csv)
    typer.echo(f"✅ Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
    
    # Exploración
    exploration = save_data_exploration()
    typer.echo(f"✅ Exploración completada")
    typer.echo(f"   Clases: {list(exploration['label_distribution'].keys())}")
    
    # Esquema
    schema = save_data_schema()
    typer.echo(f"✅ Esquema de datos creado")
    
    # Validación
    validation = validate_schema(df, schema)
    typer.echo(f"✅ Validación: {'PASS' if validation['valid'] else 'FAIL'}")


@app.command()
def prepare(
    input_csv: str = typer.Option("data/raw/dataset.csv", help="CSV de entrada"),
    output_parquet: str = typer.Option("data/processed/preprocesado.parquet", help="Parquet de salida"),
    use_lemmatization: bool = typer.Option(False, help="Usar lematización (spaCy)")
):
    """🧹 Fase 3: Data Preparation"""
    typer.echo("🔧 Iniciando Data Preparation...")
    
    df_clean = prepare_data_pipeline(
        csv_path=input_csv,
        output_path=output_parquet,
        use_lemmatization=use_lemmatization
    )
    
    typer.echo(f"✅ Datos preprocesados: {df_clean.shape[0]} filas")
    typer.echo(f"   Guardado: {output_parquet}")
    typer.echo(f"   Columnas: {list(df_clean.columns)}")


@app.command()
def train(
    input_parquet: str = typer.Option("data/processed/preprocesado.parquet", help="Parquet preprocesado"),
    min_df: int = typer.Option(2, help="Min document frequency"),
    C: float = typer.Option(1.0, help="Parámetro de regularización SVC"),
    cv_folds: int = typer.Option(5, help="Número de folds para cross-validation")
):
    """🤖 Fase 4: Modeling"""
    typer.echo("⚙️ Iniciando Modeling...")
    
    from proyecto_mlops.data_preparation import load_preprocessed_data
    
    # Cargar datos preprocesados
    df = load_preprocessed_data(input_parquet)
    texts = df["text_norm"].tolist()
    labels = df["label"].tolist()
    
    typer.echo(f"✅ Datos cargados: {len(texts)} documentos")
    
    # Cross-validation
    cv_result = cross_validate_model(
        texts=texts,
        labels=labels,
        cv_folds=cv_folds,
        min_df=min_df,
        C=C
    )
    
    typer.echo(f"✅ Cross-Validation completada")
    typer.echo(f"   Accuracy: {cv_result['accuracy_mean']:.4f} ± {cv_result['accuracy_std']:.4f}")
    typer.echo(f"   F1-Macro: {cv_result['f1_macro_mean']:.4f} ± {cv_result['f1_macro_std']:.4f}")
    
    # Entrenar modelo final
    pipe, metrics = train_model(
        texts=texts,
        labels=labels,
        min_df=min_df,
        C=C
    )
    
    model_path = save_model(pipe)
    typer.echo(f"   Modelo guardado: {model_path}")
    
    # Registrar experimento
    log_experiment({
        "stage": "final_training",
        "min_df": min_df,
        "C": C,
        "metrics": metrics,
        "cv_results": cv_result
    })


@app.command()
def evaluate(
    model_path: str = typer.Option("models/svm_tfidf_v1.joblib", help="Ruta del modelo"),
    input_parquet: str = typer.Option("data/processed/preprocesado.parquet", help="Parquet preprocesado"),
    min_f1_per_class: float = typer.Option(0.70, help="F1 mínimo por clase")
):
    """📊 Fase 5: Evaluation"""
    typer.echo("✅ Iniciando Evaluation...")
    
    from proyecto_mlops.modeling import load_model
    from proyecto_mlops.data_preparation import load_preprocessed_data
    from sklearn.model_selection import train_test_split
    
    # Cargar modelo y datos
    pipe = load_model(model_path)
    df = load_preprocessed_data(input_parquet)
    
    texts = df["text_norm"].tolist()
    labels = df["label"].tolist()
    
    # Split test
    _, X_test, _, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Evaluación completa
    eval_result = full_evaluation(
        pipe=pipe,
        X_test=X_test,
        y_test=y_test,
        min_f1_per_class=min_f1_per_class
    )
    
    # Guardar reporte
    report_path = save_evaluation_report(eval_result)
    
    typer.echo(f"✅ Evaluación completada")
    typer.echo(f"   Status: {eval_result['overall_status']}")
    typer.echo(f"   Accuracy: {eval_result['model_metrics']['accuracy']:.4f}")
    typer.echo(f"   F1-Macro: {eval_result['model_metrics']['f1_macro']:.4f}")
    typer.echo(f"   Latencia P95: {eval_result['latency_metrics']['latency_p95_ms']:.1f}ms")
    typer.echo(f"   Reporte: {report_path}")


@app.command()
def deploy(
    model_version: int = typer.Option(1, help="Versión del modelo a desplegar"),
    promote: bool = typer.Option(False, help="Promover a producción")
):
    """🚀 Fase 6: Deployment"""
    typer.echo("📦 Iniciando Deployment...")
    
    model_path = f"models/svm_tfidf_v{model_version}.joblib"
    
    # Registrar modelo
    register_model_in_registry(
        model_path=model_path,
        model_name="svm_tfidf",
        version=model_version,
        metrics={"status": "ready"},
        status="candidate"
    )
    
    typer.echo(f"✅ Modelo registrado: v{model_version}")
    
    # Promover si se especifica
    if promote:
        promote_to_production(version=model_version)
        typer.echo(f"✅ Modelo promovido a PRODUCCIÓN: v{model_version}")
    
    # Generar guía de deployment
    guide_path = save_deployment_guide()
    typer.echo(f"   Guía de deployment: {guide_path}")


@app.command()
def all(
    fast: Optional[int] = typer.Option(None, help="Modo rápido: procesa N filas")
):
    """🔄 Pipeline Completo CRISP-DM"""
    typer.echo("🚀 Ejecutando pipeline COMPLETO CRISP-DM...\n")
    
    # Business Understanding
    typer.echo("━" * 60)
    typer.echo("FASE 1: BUSINESS UNDERSTANDING")
    typer.echo("━" * 60)
    business()
    
    # Data Understanding
    typer.echo("\n" + "━" * 60)
    typer.echo("FASE 2: DATA UNDERSTANDING")
    typer.echo("━" * 60)
    understand()
    
    # Data Preparation
    typer.echo("\n" + "━" * 60)
    typer.echo("FASE 3: DATA PREPARATION")
    typer.echo("━" * 60)
    prepare()
    
    # Modeling
    typer.echo("\n" + "━" * 60)
    typer.echo("FASE 4: MODELING")
    typer.echo("━" * 60)
    train()
    
    # Evaluation
    typer.echo("\n" + "━" * 60)
    typer.echo("FASE 5: EVALUATION")
    typer.echo("━" * 60)
    evaluate()
    
    # Deployment
    typer.echo("\n" + "━" * 60)
    typer.echo("FASE 6: DEPLOYMENT")
    typer.echo("━" * 60)
    deploy(model_version=1, promote=False)
    
    typer.echo("\n" + "🎉" * 30)
    typer.echo("\n✅ PIPELINE COMPLETO EJECUTADO EXITOSAMENTE\n")
    typer.echo("📊 Documentos generados:")
    typer.echo("   - docs/business_understanding.json")
    typer.echo("   - docs/data_schema.json")
    typer.echo("   - data/processed/data_exploration_report.json")
    typer.echo("   - data/processed/preprocesado.parquet")
    typer.echo("   - models/svm_tfidf_v1.joblib")
    typer.echo("   - models/registry.json")
    typer.echo("   - data/processed/evaluation_report.json")
    typer.echo("")


@app.command()
def version():
    """Muestra la versión del paquete"""
    from proyecto_mlops import __version__
    typer.echo(f"🔖 proyecto-mlops versión: {__version__}")


@app.command()
def info():
    """Información del proyecto"""
    typer.echo("""
╔════════════════════════════════════════════════════════════════╗
║         PROYECTO-MLOPS: Clasificación de Documentos            ║
║                  Ciclo Completo CRISP-DM                       ║
╚════════════════════════════════════════════════════════════════╝

📚 Fases CRISP-DM:
   1. business          - Comprensión del negocio
   2. understand        - Exploración de datos
   3. prepare           - Preprocesamiento
   4. train             - Entrenamiento de modelo
   5. evaluate          - Evaluación
   6. deploy            - Despliegue

🚀 Comandos especiales:
   all                  - Ejecutar pipeline completo
   version              - Mostrar versión
   info                 - Esta información

📖 Documentación:
   - CRISP_DM_REPORT.md (Reporte técnico)
   - BUSINESS_PRESENTATION.md (Presentación de negocio)

🔗 Enlaces:
   PyPI:     https://pypi.org/project/proyecto-mlops/
   GitHub:   https://github.com/angelcast2002/PROYECTO-MLOPS
   Docker:   docker pull angelcast2002/proyecto-mlops:latest

✨ Ejemplo de uso:
   proyecto-mlops all               # Pipeline completo
   proyecto-mlops business          # Solo business understanding
   proyecto-mlops train --help      # Ver opciones de entrenamiento
""")


if __name__ == "__main__":
    app()
