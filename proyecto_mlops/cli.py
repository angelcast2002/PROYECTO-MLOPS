#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PROYECTO-MLOPS: CLI Principal
Herramienta de línea de comandos para el pipeline MLOps completo
"""

import typer
from typing import Optional

# Import fases CRISP-DM
from proyecto_mlops.business_understanding import save_business_document
from proyecto_mlops.data_understanding import (
    load_raw_dataset,
    explore_data,
    make_data_schema,
    validate_schema,
    save_data_schema
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
    save_deployment_guide,
    get_production_model
)

app = typer.Typer(
    name="proyecto-mlops",
    help="🚀 MLOps Pipeline para Clasificación de Documentos en Español",
    rich_markup_mode="rich"
)


@app.command()
def business():
    """📊 Fase 1: Business Understanding - Define objetivos de negocio"""
    typer.echo("[bold blue]🎯 Iniciando Business Understanding...[/bold blue]")
    try:
        doc = save_business_document()
        typer.echo("[bold green]✅ Documento guardado[/bold green]")
        typer.echo(f"   📌 Objetivo: {doc['business_objective']['titulo']}")
        typer.echo(f"   🎯 Target F1-Macro: {doc['ml_objective']['métricas_éxito']['f1_macro_min']}")
    except Exception as e:
        typer.echo(f"[bold red]❌ Error: {e}[/bold red]", err=True)
        raise typer.Exit(1)


@app.command()
def understand():
    """📈 Fase 2: Data Understanding - Explora y valida datos"""
    typer.echo("[bold blue]📊 Iniciando Data Understanding...[/bold blue]")
    try:
        # Load and explore
        df = load_raw_dataset()
        typer.echo(f"   ✅ Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
        
        # Explore
        exploration = explore_data()
        typer.echo(f"   📌 Etiquetas: {exploration.get('label_distribution', {})}")
        
        # Schema
        schema = make_data_schema()
        save_data_schema(schema)
        typer.echo("[bold green]✅ Data Understanding completado[/bold green]")
    except Exception as e:
        typer.echo(f"[bold red]❌ Error: {e}[/bold red]", err=True)
        raise typer.Exit(1)


@app.command()
def prepare():
    """🔄 Fase 3: Data Preparation - Preprocesa textos"""
    typer.echo("[bold blue]🔄 Iniciando Data Preparation...[/bold blue]")
    try:
        prepare_data_pipeline()
        typer.echo("[bold green]✅ Datos preparados guardados en data/processed/[/bold green]")
    except Exception as e:
        typer.echo(f"[bold red]❌ Error: {e}[/bold red]", err=True)
        raise typer.Exit(1)


@app.command()
def train(
    do_cv: bool = typer.Option(True, help="Realizar validación cruzada"),
    do_sweep: bool = typer.Option(False, help="Realizar grid search")
):
    """🧠 Fase 4: Modeling - Entrena modelos de clasificación"""
    typer.echo("[bold blue]🧠 Iniciando Modeling...[/bold blue]")
    try:
        # Train
        model, metrics = train_model()
        typer.echo(f"   ✅ Modelo entrenado - F1-Macro: {metrics['f1_macro']:.4f}")
        
        # CV
        if do_cv:
            typer.echo("   📊 Realizando validación cruzada...")
            cv_metrics = cross_validate_model()
            typer.echo(f"   ✅ CV F1-Macro: {cv_metrics['mean_f1_macro']:.4f}")
        
        # Save
        save_model(model)
        typer.echo("[bold green]✅ Modelo guardado en models/[/bold green]")
    except Exception as e:
        typer.echo(f"[bold red]❌ Error: {e}[/bold red]", err=True)
        raise typer.Exit(1)


@app.command()
def evaluate():
    """📋 Fase 5: Evaluation - Evalúa desempeño del modelo"""
    typer.echo("[bold blue]📋 Iniciando Evaluation...[/bold blue]")
    try:
        status, report = full_evaluation()
        save_evaluation_report(report)
        typer.echo(f"   ✅ Status: {status}")
        typer.echo(f"   📊 Metrics: {report.get('metrics', {})}")
        typer.echo("[bold green]✅ Evaluación completada[/bold green]")
    except Exception as e:
        typer.echo(f"[bold red]❌ Error: {e}[/bold red]", err=True)
        raise typer.Exit(1)


@app.command()
def deploy(
    promote: bool = typer.Option(False, help="Promover a producción"),
    version: Optional[int] = typer.Option(None, help="Versión a promover")
):
    """🚀 Fase 6: Deployment - Registra y despliega modelos"""
    typer.echo("[bold blue]🚀 Iniciando Deployment...[/bold blue]")
    try:
        register_model_in_registry()
        typer.echo("   ✅ Modelo registrado")
        
        if promote and version:
            promote_to_production(version)
            typer.echo(f"   ✅ Modelo v{version} promovido a producción")
        
        save_deployment_guide()
        typer.echo("[bold green]✅ Deployment completado[/bold green]")
    except Exception as e:
        typer.echo(f"[bold red]❌ Error: {e}[/bold red]", err=True)
        raise typer.Exit(1)


@app.command()
def all(
    cv: bool = typer.Option(True, help="Incluir validación cruzada")
):
    """🎯 Pipeline Completo - Ejecuta todas las fases CRISP-DM"""
    typer.echo("[bold cyan]🚀 INICIANDO PIPELINE COMPLETO CRISP-DM[/bold cyan]\n")
    
    try:
        # Phase 1
        typer.echo("[bold]1️⃣ Business Understanding[/bold]")
        business()
        typer.echo()
        
        # Phase 2
        typer.echo("[bold]2️⃣ Data Understanding[/bold]")
        understand()
        typer.echo()
        
        # Phase 3
        typer.echo("[bold]3️⃣ Data Preparation[/bold]")
        prepare()
        typer.echo()
        
        # Phase 4
        typer.echo("[bold]4️⃣ Modeling[/bold]")
        train(do_cv=cv)
        typer.echo()
        
        # Phase 5
        typer.echo("[bold]5️⃣ Evaluation[/bold]")
        evaluate()
        typer.echo()
        
        # Phase 6
        typer.echo("[bold]6️⃣ Deployment[/bold]")
        deploy()
        typer.echo()
        
        typer.echo("[bold green]✨ PIPELINE COMPLETO FINALIZADO EXITOSAMENTE ✨[/bold green]")
        
    except Exception as e:
        typer.echo(f"\n[bold red]❌ PIPELINE FALLIDO: {e}[/bold red]", err=True)
        raise typer.Exit(1)


@app.command()
def status():
    """📊 Ver estado del modelo actual en producción"""
    typer.echo("[bold blue]📊 Estado Actual[/bold blue]")
    try:
        prod_model = get_production_model()
        if prod_model:
            typer.echo(f"   ✅ Modelo en Producción: v{prod_model['version']}")
            typer.echo(f"   📅 Fecha: {prod_model.get('created_at', 'N/A')}")
            typer.echo(f"   🎯 F1-Macro: {prod_model.get('metrics', {}).get('f1_macro', 'N/A')}")
        else:
            typer.echo("   ⚠️ No hay modelo en producción")
    except Exception as e:
        typer.echo(f"[bold red]❌ Error: {e}[/bold red]", err=True)


@app.command()
def version():
    """📌 Ver versión del paquete"""
    typer.echo("proyecto-mlops v0.1.0")
    typer.echo("MLOps Pipeline para Clasificación de Documentos en Español")


def main():
    """Punto de entrada principal"""
    app()


if __name__ == "__main__":
    main()
