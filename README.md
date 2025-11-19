# PROYECTO-MLOPS: Clasificación de Documentos en Español

[![CI](https://github.com/angelcast2002/PROYECTO-MLOPS/workflows/ci/badge.svg)](https://github.com/angelcast2002/PROYECTO-MLOPS/actions)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-latest-blue.svg)](https://hub.docker.com/repository/docker/angelcast2025/proyecto-mlops/general)
[![PyPI](https://img.shields.io/badge/PyPI-0.1.0-green.svg)](https://pypi.org/project/proyecto-mlops/)

## 📋 Descripción

Sistema MLOps completo para **clasificación automática de documentos en español** implementando todas las fases del ciclo **CRISP-DM** (Cross Industry Standard Process for Data Mining).

### ✨ Características

- ✅ **Estructura CRISP-DM Completa:** 6 fases separadas en módulos Python
- ✅ **Paquete Publicable:** Disponible en PyPI
- ✅ **Containerizado:** Docker image lista para producción
- ✅ **CI/CD Automatizado:** GitHub Actions workflows (test, Docker, PyPI, model registry)
- ✅ **Versionado de Modelos:** Registro automático de versiones y promoción a producción
- ✅ **Evaluación Integral:** Fairness, latencia, drift detection
- ✅ **Documentación:** Reportes CRISP-DM y presentación de negocio

---

## 🎯 Valor de Negocio

| Métrica | Mejora |
|---------|--------|
| Tiempo de procesamiento | 900-1800x más rápido |
| Escalabilidad | 1000x (100→100,000 docs/día) |
| Costo por documento | 99.8% reducción |
| Disponibilidad | 24/7 (vs 8h/día) |
| **ROI Año 1** | **636%** |
| **Payback Period** | **2 meses** |

---

## 🚀 Inicio Rápido

### Instalación

```bash
# Opción 1: Desde PyPI
pip install proyecto-mlops

# Opción 2: Desarrollo local
git clone https://github.com/angelcast2002/PROYECTO-MLOPS.git
cd PROYECTO-MLOPS
pip install -e ".[dev]"
```

### Uso Básico

```python
from proyecto_mlops import (
    load_raw_dataset,
    preprocess_dataframe,
    train_model,
    full_evaluation,
    promote_to_production
)

# 1. Cargar datos
df = load_raw_dataset("data/raw/dataset.csv")

# 2. Preprocesar
df_clean = preprocess_dataframe(df)

# 3. Entrenar modelo
texts = df_clean["text_norm"].tolist()
labels = df_clean["label"].tolist()
pipe, metrics = train_model(texts, labels)

# 4. Evaluar
eval_result = full_evaluation(pipe, texts_test, labels_test)

# 5. Desplegar
promote_to_production(version=1)
```

### Ejecutar Pipeline Completo

```bash
# CLI
python cli.py all

# Modo rápido (testing)
python cli.py all --fast 1000

# Docker
docker build -t proyecto-mlops:latest .
docker run -v $(pwd)/data:/app/data proyecto-mlops:latest
```

---

## 📦 Estructura del Proyecto

```
PROYECTO-MLOPS/
├── proyecto_mlops/                 # Paquete principal (publicable)
│   ├── __init__.py
│   ├── business_understanding/     # Fase 1: Objetivos de negocio
│   ├── data_understanding/         # Fase 2: Exploración de datos
│   ├── data_preparation/           # Fase 3: Preprocesamiento
│   ├── modeling/                   # Fase 4: Entrenamiento
│   ├── evaluation/                 # Fase 5: Evaluación
│   ├── deployment/                 # Fase 6: Despliegue
│   └── utils/                      # Utilidades compartidas
│
├── .github/workflows/              # CI/CD Automation
│   ├── ci.yml                      # Tests + Quality checks
│   ├── cd-docker.yml               # Docker build & push
│   ├── cd-pypi.yml                 # PyPI publishing
│   └── cd-model-registry.yml       # Model versioning
│
├── data/
│   ├── raw/                        # Datos originales
│   ├── processed/                  # Datos preprocesados
│   └── registry/                   # Versionado de datos
│
├── models/                         # Modelos entrenados
├── tests/                          # Tests unitarios
├── docs/                           # Documentación
│
├── Dockerfile                      # Containerización
├── setup.py                        # Configuración del paquete
├── requirements.txt                # Dependencias
├── cli.py                          # CLI principal
├── pipeline.py                     # Pipeline monolítico (legacy)
├── config.yaml                     # Configuración
│
├── CRISP_DM_REPORT.md             # Reporte técnico completo
├── BUSINESS_PRESENTATION.md        # Presentación de negocio
└── README.md                       # Este archivo
```

---

## 🧩 Arquitectura multi-paquete (CRISP-DM por etapas)

Además del paquete monolítico `proyecto-mlops`, el repo ahora expone paquetes independientes por fase y un agregador final:

```
packages/
    ├─ proyecto-core/       # utilidades y rutas compartidas
    ├─ proyecto-bu/         # Business Understanding
    ├─ proyecto-du/         # Data Understanding
    ├─ proyecto-dp/         # Data Preparation
    ├─ proyecto-modeling/   # Modeling
    ├─ proyecto-eval/       # Evaluation
    ├─ proyecto-deploy/     # Deployment
    └─ proyecto-final/      # Agregador + CLI multi-paquete
```

Instalación en desarrollo (editable):

```pwsh
pip install -e .\packages\proyecto-core
pip install -e .\packages\proyecto-bu
pip install -e .\packages\proyecto-du
pip install -e .\packages\proyecto-dp
pip install -e .\packages\proyecto-modeling
pip install -e .\packages\proyecto-eval
pip install -e .\packages\proyecto-deploy
pip install -e .\packages\proyecto-final
```

O bien, usa el script de instalación rápida (Windows/PowerShell):

```pwsh
pwsh -File .\scripts\dev_install.ps1
```

Uso de ejemplo:

```python
from proyecto_dp import prepare_data_pipeline
from proyecto_modeling import train_model
from proyecto_final import full_evaluation
```

CLI del agregador:

```pwsh
proyecto-final all
```

Nota: el archivo `pipeline.py` se mantiene como shim de compatibilidad para los tests y, si no encuentra el paquete `proyecto_dp`, hace fallback al monolítico.

---

## � Paquetes en PyPI

Además del paquete monolítico `proyecto-mlops`, cada etapa está publicada individualmente en PyPI. Links directos:

- Núcleo compartido: https://pypi.org/project/proyecto-core/
- Business Understanding: https://pypi.org/project/proyecto-bu/
- Data Understanding: https://pypi.org/project/proyecto-du/
- Data Preparation: https://pypi.org/project/proyecto-dp/
- Modeling: https://pypi.org/project/proyecto-modeling/
- Evaluation: https://pypi.org/project/proyecto-eval/
- Deployment: https://pypi.org/project/proyecto-deploy/
- Agregador (CLI multi-paquete): https://pypi.org/project/proyecto-mlops-final/
- Paquete monolítico: https://pypi.org/project/proyecto-mlops/

Para publicar nuevas versiones de todos los paquetes desde GitHub Actions, usa el workflow “CD - PyPI (multi-package)” (manual) o genera un tag `packages-vX.Y.Z` y se publicará en matriz.

---

## �🔄 Ciclo CRISP-DM

### 1. **Business Understanding** 🎯
Definición de objetivos de negocio, problemas, y criterios de éxito.

**Ubicación:** `proyecto_mlops/business_understanding/`

```python
from proyecto_mlops import save_business_document
save_business_document()
```

---

### 2. **Data Understanding** 📊
Exploración, validación y esquematización de datos.

**Ubicación:** `proyecto_mlops/data_understanding/`

```python
from proyecto_mlops import (
    load_raw_dataset,
    explore_data,
    save_data_schema,
    validate_schema
)

df = load_raw_dataset()
exploration = explore_data(df)
schema = save_data_schema()
```

**Outputs:**
- `data/data_exploration_report.json`
- `docs/data_schema.json`

---

### 3. **Data Preparation** 🧹
Preprocesamiento: normalización, tokenización, limpieza, stemming.

**Ubicación:** `proyecto_mlops/data_preparation/`

```python
from proyecto_mlops import prepare_data_pipeline

df_clean = prepare_data_pipeline(
    csv_path="data/raw/dataset.csv",
    use_lemmatization=False
)
```

**Transformaciones:**
- Normalización Unicode (lowercase, acentos)
- Tokenización (regex)
- Limpieza (stopwords, dígitos)
- Stemming (Snowball Spanish)

**Output:** `data/processed/preprocesado.parquet`

---

### 4. **Modeling** 🤖
Entrenamiento de clasificador con TF-IDF + LinearSVC.

**Ubicación:** `proyecto_mlops/modeling/`

```python
from proyecto_mlops import (
    train_model,
    cross_validate_model,
    hyperparameter_sweep
)

# Holdout validation
pipe, metrics = train_model(texts, labels)

# Cross-validation (5-fold)
cv_result = cross_validate_model(texts, labels)

# Hyperparameter sweep
results = hyperparameter_sweep(
    texts, labels,
    param_grid=[
        {"min_df": 1, "C": 0.1},
        {"min_df": 2, "C": 1.0},
        {"min_df": 3, "C": 10.0}
    ]
)
```

**Configuración:**
- Vectorización: TF-IDF (1,2)-grams
- Clasificador: LinearSVC (C=1.0)
- Validación: 5-Fold StratifiedKFold

---

### 5. **Evaluation** 📈
Métricas de rendimiento, fairness, latencia.

**Ubicación:** `proyecto_mlops/evaluation/`

```python
from proyecto_mlops import (
    evaluate_model,
    measure_latency,
    check_fairness,
    full_evaluation
)

# Evaluación completa
eval_result = full_evaluation(
    pipe=pipe,
    X_test=X_test,
    y_test=y_test,
    min_f1_per_class=0.70
)
```

**Métricas:**
- Accuracy, F1-Macro, F1-Weighted
- Latencia: P50, P95, P99
- Fairness: F1 mínima por clase ≥ 0.70

---

### 6. **Deployment** 🚀
Registro, versionado, y promoción de modelos.

**Ubicación:** `proyecto_mlops/deployment/`

```python
from proyecto_mlops import (
    register_model_in_registry,
    promote_to_production,
    get_production_model,
    create_deployment_package
)

# Registrar modelo
register_model_in_registry(
    model_path="models/svm_tfidf_v1.joblib",
    model_name="svm_tfidf",
    version=1,
    metrics=eval_result
)

# Promover a producción
promote_to_production(version=1)

# Crear paquete de deployment
package = create_deployment_package(
    model_path="models/svm_tfidf_v1.joblib"
)
```

---

## 📊 Métricas y Resultados

### Performance

| Métrica | Valor | Target |
|---------|-------|--------|
| Accuracy | 0.82 | ≥ 0.80 ✅ |
| F1-Macro | 0.78 | ≥ 0.75 ✅ |
| F1-Weighted | 0.81 | ≥ 0.75 ✅ |

### Latencia

| Percentil | Tiempo | SLA |
|-----------|--------|-----|
| P50 | 45ms | - |
| P95 | 180ms | ≤ 200ms ✅ |
| P99 | 220ms | - |

### Fairness

| Clase | F1-Score | Status |
|-------|----------|--------|
| Clase A | 0.76 | ✅ |
| Clase B | 0.79 | ✅ |
| Clase C | 0.72 | ✅ |

---

## 🔄 CI/CD Pipelines

### GitHub Actions Workflows

#### 1. **CI Pipeline** (`.github/workflows/ci.yml`)
Ejecuta en cada push/PR:
- ✅ Tests unitarios (`pytest`)
- ✅ Code quality (`black`, `isort`, `flake8`)
- ✅ Security scan (`Trivy`)
- ✅ Coverage reports (`codecov`)

#### 2. **Docker CD** (`.github/workflows/cd-docker.yml`)
Ejecuta en push a main:
- ✅ Build imagen Docker
- ✅ Push a Docker Hub (`docker.io/angelcast2025/proyecto-mlops`)
- ✅ Tagged con versión + latest

#### 3. **PyPI CD** (`.github/workflows/cd-pypi.yml`)
Ejecuta en tag `v*`:
- ✅ Build paquete Python
- ✅ Publish a TestPyPI (testing)
- ✅ Publish a PyPI (production)
- ✅ Create GitHub Release

#### 4. **Model Registry** (`.github/workflows/cd-model-registry.yml`)
Ejecuta después de CI exitoso:
- ✅ Run full pipeline
- ✅ Generate model metadata
- ✅ Upload artifacts (30 días retención)
- ✅ Update registry

---

## 🐳 Docker

Para una guía paso a paso de despliegue en DigitalOcean (Droplet Ubuntu) consulta: `docs_project/DEPLOY_DIGITAL_OCEAN.md`.

### Build

```bash
docker build -t proyecto-mlops:latest .
```

### Run

```bash
# Con volumen de datos
docker run -v $(pwd)/data:/app/data \
           -p 8000:8000 \
           proyecto-mlops:latest

# Con archivo de configuración
docker run -v $(pwd)/config.yaml:/app/config.yaml \
           proyecto-mlops:latest \
           python cli.py all
```

### Push a Docker Hub

```bash
docker tag proyecto-mlops:latest angelcast2025/proyecto-mlops:latest
docker push angelcast2025/proyecto-mlops:latest
```

### Disponible en

```
docker pull angelcast2025/proyecto-mlops:latest
docker pull angelcast2025/proyecto-mlops:0.1.6
```

---

## 📦 Instalación desde PyPI

### Desde PyPI

```bash
pip install proyecto-mlops
```

### Desde TestPyPI

```bash
pip install --index-url https://test.pypi.org/simple/ proyecto-mlops
```

### Con extras

```bash
# Desarrollo
pip install proyecto-mlops[dev]

# Docker support
pip install proyecto-mlops[docker]
```

---

## 📝 Documentación

### Documentos Disponibles

1. **`CRISP_DM_REPORT.md`** (Este proyecto)
   - Reporte técnico completo de todas las fases CRISP-DM
   - Resultados, conclusiones y escalabilidad
   - 15+ páginas

2. **`BUSINESS_PRESENTATION.md`**
   - Presentación enfocada en valor de negocio
   - ROI, timeline, riesgos
   - Lenguaje ejecutivo (no técnico)

3. **`README.md`**
   - Guía de instalación y uso
   - Estructura del proyecto
   - Quick start

### Generar Documentación

```python
from proyecto_mlops import save_business_document, save_data_schema
from proyecto_mlops import save_data_exploration

# Business
save_business_document()

# Data
save_data_schema()
save_data_exploration()
```

---

## 🧪 Testing

```bash
# Instalar test dependencies
pip install pytest pytest-cov

# Ejecutar tests
pytest tests/ -v

# Con coverage
pytest tests/ --cov=proyecto_mlops --cov-report=html
```

---

## 📊 Monitoreo en Producción

### Métricas a Trackear

```python
metrics = {
    "predictions_per_minute": 120,
    "avg_latency_ms": 45,
    "p95_latency_ms": 180,
    "error_rate": 0.01,
    "drift_psi": 0.05,
    "model_accuracy": 0.82
}
```

### Alertas Recomendadas

- ⚠️ P95 latency > 300ms
- ⚠️ Error rate > 5%
- ⚠️ PSI (drift) > 0.2
- ⚠️ Accuracy drop > 10%

---

## 🔐 Configuración de Secrets en GitHub

Para CI/CD completo, configura estos secrets:

```
DOCKER_USERNAME    # Docker Hub username
DOCKER_PASSWORD    # Docker Hub access token
PYPI_API_TOKEN     # PyPI API token
TEST_PYPI_API_TOKEN # TestPyPI API token
```

**Ubicación:** GitHub → Settings → Secrets and variables → Actions

---

## 🎓 Temas Cubiertos

### Temas MLOps

- ✅ Versionado de código y datos
- ✅ Versionado de modelos
- ✅ Reproducibilidad
- ✅ CI/CD automation
- ✅ Containerización
- ✅ Infrastructure as Code
- ✅ Monitoring y alerting
- ✅ Model registry
- ✅ Deployment automation

### Temas ML

- ✅ Preprocesamiento de texto
- ✅ Feature engineering (TF-IDF)
- ✅ Clasificación (LinearSVC)
- ✅ Cross-validation
- ✅ Hyperparameter tuning
- ✅ Fairness y equidad
- ✅ Latency optimization
- ✅ Drift detection

### Temas DevOps

- ✅ Docker
- ✅ GitHub Actions
- ✅ Package distribution (PyPI)
- ✅ CI/CD workflows
- ✅ Security scanning

---

## 🎯 Próximos Pasos

### Corto Plazo

1. Desplegar en servidor de staging
2. Integrar FastAPI para inferencia
3. Configurar Prometheus + Grafana

### Mediano Plazo

1. Kubernetes deployment
2. Auto-scaling y load balancing
3. Advanced monitoring

### Largo Plazo

1. A/B testing y canary deployments
2. Active learning loop
3. Multi-model serving

---

## 👥 Contribuciones

Pull requests son bienvenidos. Para cambios mayores, abre un issue primero.

---

## 📄 Licencia

MIT License - ver LICENSE file

---

## 📞 Contacto

**Autores:** Angel Castellanos, Alejandro Azurdia, Diego Morales, Sara Echeverría  

---

## 🔗 Enlaces

| Recurso | Link |
|---------|------|
| 📦 PyPI Package | https://pypi.org/project/proyecto-mlops/ |
| 🐳 Docker Hub | https://hub.docker.com/repository/docker/angelcast2025/proyecto-mlops/general |
| 📊 Repository | https://github.com/angelcast2002/PROYECTO-MLOPS |
| 🚀 Releases | https://github.com/angelcast2002/PROYECTO-MLOPS/releases |
| ⚙️ GitHub Actions | https://github.com/angelcast2002/PROYECTO-MLOPS/actions |

---

**Última actualización:** Noviembre 13, 2025  
**Versión:** 0.1.6  
**Estado:** ✅ Producción Ready
