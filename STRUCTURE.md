# 📁 Estructura del Proyecto - PROYECTO-MLOPS

## Vista General

```
PROYECTO-MLOPS/
├── 📦 proyecto_mlops/           # ⭐ Paquete principal (publicable en PyPI)
│   ├── __init__.py              # Exports públicos
│   ├── cli.py                   # CLI principal (entry point)
│   ├── business_understanding/  # Fase 1 CRISP-DM
│   ├── data_understanding/      # Fase 2 CRISP-DM
│   ├── data_preparation/        # Fase 3 CRISP-DM
│   ├── modeling/                # Fase 4 CRISP-DM
│   ├── evaluation/              # Fase 5 CRISP-DM
│   ├── deployment/              # Fase 6 CRISP-DM
│   └── utils/                   # Utilidades compartidas
│
├── 📚 docs_project/             # Documentación del proyecto
│   ├── CRISP_DM_REPORT.md       # Reporte técnico (15+ páginas)
│   ├── BUSINESS_PRESENTATION.md # Presentación para stakeholders
│   ├── DEPLOYMENT.md            # Guía de deployment
│   ├── CONTRIBUTING.md          # Guía de contribuciones
│   └── README.md                # Documentación principal
│
├── 🐳 infra/                    # Infraestructura y configuración
│   ├── Dockerfile               # Containerización
│   ├── .dockerignore            # Exclusiones Docker
│   ├── setup.py                 # Configuración legacy (en raíz)
│   ├── pyproject.toml           # Configuración moderna (en raíz)
│   └── Makefile                 # Comandos útiles
│
├── ⚙️ config/                   # Archivos de configuración
│   └── config.yaml              # Configuración de la aplicación
│
├── 💾 data/                     # Datos del proyecto
│   ├── raw/                     # Datos originales (sin procesar)
│   ├── processed/               # Datos procesados y métricas
│   └── registry/                # Versionado de datos (v1, v2, v3...)
│
├── 🤖 models/                   # Modelos entrenados
│   ├── *.joblib                 # Modelos serializados
│   ├── registry.json            # Registro de versiones
│   └── svm_*.joblib             # Versiones específicas
│
├── 📊 figures/                  # Figuras y gráficos generados
│   └── .gitkeep                 # Mantiene la carpeta en git
│
├── 🧪 tests/                    # Tests unitarios
│   ├── conftest.py              # Configuración pytest
│   ├── test_*.py                # Archivos de test
│   └── __pycache__/
│
├── 🔧 .github/workflows/        # GitHub Actions (CI/CD)
│   ├── ci.yml                   # Pipeline de integración continua
│   ├── cd-docker.yml            # Pipeline Docker Hub
│   ├── cd-pypi.yml              # Pipeline PyPI
│   └── cd-model-registry.yml    # Pipeline model versioning
│
└── 📄 Archivos en Raíz
    ├── setup.py                 # Instalación del paquete
    ├── pyproject.toml           # Configuración moderna
    ├── requirements.txt         # Dependencias (pip freeze)
    ├── README.md                # Inicio rápido
    ├── LICENSE                  # MIT License
    ├── .gitignore               # Exclusiones git
    ├── .gitattributes           # Atributos git
    ├── STRUCTURE.md             # Este archivo
    └── cli_improved.py          # CLI antiguo (legacy, no usar)
```

---

## 📋 Descripción por Carpeta

### 📦 `proyecto_mlops/` - Paquete Principal

El corazón del proyecto. Contiene toda la lógica MLOps modularizada en fases CRISP-DM.

**Archivos clave:**
- `cli.py` - **Entry point** para ejecutar desde terminal
- `__init__.py` - Exports de funciones públicas
- `business_understanding/__init__.py` - Define objetivos de negocio
- `data_understanding/__init__.py` - Exploración y validación
- `data_preparation/__init__.py` - Preprocesamiento de textos
- `modeling/__init__.py` - Entrenamiento de modelos
- `evaluation/__init__.py` - Evaluación de desempeño
- `deployment/__init__.py` - Registro y deployment
- `utils/__init__.py` - Funciones compartidas (logging, I/O, etc.)

**Uso:**
```bash
# Instalar en modo desarrollo
pip install -e .

# Ejecutar desde CLI
proyecto-mlops all              # Pipeline completo
proyecto-mlops business         # Solo fase 1
proyecto-mlops understand       # Solo fase 2
# ... etc
```

---

### 📚 `docs_project/` - Documentación

Toda la documentación del proyecto en formato Markdown.

**Archivos:**
- `CRISP_DM_REPORT.md` - Reporte técnico detallado (para técnicos)
- `BUSINESS_PRESENTATION.md` - Presentación de negocio (sin jerga técnica)
- `DEPLOYMENT.md` - Guía de instalación y deployment
- `CONTRIBUTING.md` - Guía para colaboradores
- `README.md` - Inicio rápido y referencia

**Lectores principales:**
- `CRISP_DM_REPORT.md` → Data Scientists, Engineers
- `BUSINESS_PRESENTATION.md` → Stakeholders, Gerentes
- `DEPLOYMENT.md` → DevOps, Ops Engineers
- `README.md` → Todos (primera lectura)

---

### 🐳 `infra/` - Infraestructura

Archivos de configuración para deployment y distribución.

**Archivos:**
- `Dockerfile` - Imagen Docker (Python 3.11-slim)
- `.dockerignore` - Exclusiones de build
- `setup.py` - Configuración del paquete (legacy, también en raíz)
- `pyproject.toml` - Configuración moderna (también en raíz)
- `Makefile` - Comandos útiles para desarrollo

**Uso:**
```bash
# Con Makefile
make install
make test
make docker-build
make pipeline

# Manual
docker build -t proyecto-mlops:latest .
pip install -e ".[dev]"
```

---

### ⚙️ `config/` - Configuración

Archivos de configuración de la aplicación.

**Archivos:**
- `config.yaml` - Parámetros de la app (frecuencia de retraining, rutas, etc.)

---

### 💾 `data/` - Datos

Datos del proyecto organizados por estado.

**Estructura:**
```
data/
├── raw/                   # Datos originales (no modificar)
│   ├── data.csv
│   ├── dataset.csv
│   └── nuevo.csv          # (Generado durante pipeline)
├── processed/             # Datos procesados
│   ├── *.parquet          # Datos normalizados
│   ├── exp_log.jsonl      # Experimentos registrados
│   ├── class_distribution.csv
│   ├── cv_summary.json
│   ├── drift_report.json
│   └── ...
└── registry/              # Versionado de datos
    ├── registry.json
    ├── v1/
    │   ├── dataset.csv
    │   ├── meta.json
    │   └── data_schema.json
    ├── v2/
    └── v3/
```

**Importante:** Los datos en `raw/` nunca se modifican. Se procesan y guardan en `processed/`.

---

### 🤖 `models/` - Modelos

Modelos entrenados y registry de versiones.

**Archivos:**
```
models/
├── registry.json              # Metadata de todas las versiones
├── svm_tfidf_v1.joblib       # Versión 1
├── svm_tfidf_v2.joblib       # Versión 2
├── svm_tfidf_v3.joblib       # Versión 3 (actual)
├── tfidf_vectorizer.joblib   # Vectorizador
├── bow_vectorizer.joblib     # Bag of Words
└── word2vec.model            # Word2Vec
```

**registry.json** contiene:
- Versión del modelo
- Status (candidate, production, archived)
- Métricas (F1, accuracy, latencia)
- Timestamp de creación
- Promoción a producción

---

### 📊 `figures/` - Figuras

Gráficos generados durante el análisis (vacío inicialmente).

```
figures/
├── .gitkeep               # Mantiene la carpeta en git
├── data_distribution.png  # (generado)
├── confusion_matrix.png   # (generado)
└── ...
```

---

### 🧪 `tests/` - Tests Unitarios

Tests automáticos del código.

**Estructura:**
```
tests/
├── conftest.py              # Configuración de pytest
├── test_text_utils.py       # Tests de utilidades
├── test_modeling.py         # (pendiente)
├── test_evaluation.py       # (pendiente)
└── __pycache__/
```

**Ejecutar:**
```bash
pytest                        # Todos
pytest -v                     # Verbose
pytest --cov                  # Con coverage
make test                     # Con Makefile
```

---

### 🔧 `.github/workflows/` - CI/CD

Pipelines automáticos de GitHub Actions.

**Workflows:**
1. `ci.yml` - Tests, linting, security (cada push)
2. `cd-docker.yml` - Build Docker (push a main)
3. `cd-pypi.yml` - Publicar PyPI (en tags v*)
4. `cd-model-registry.yml` - Versionar modelos (post-CI)

---

## 🚀 Cómo Usar

### Instalación Local

```bash
# Clonar repo
git clone https://github.com/angelcast2002/PROYECTO-MLOPS.git
cd PROYECTO-MLOPS

# Instalar dependencias
pip install -r requirements.txt

# Instalar en modo desarrollo
pip install -e ".[dev]"

# (Opcional) Instalar extras
pip install -e ".[docker]"
```

### Ejecutar Pipeline Completo

```bash
# Opción 1: Con CLI
proyecto-mlops all

# Opción 2: Con Makefile
make pipeline

# Opción 3: Python directo
from proyecto_mlops import *
save_business_document()
prepare_data_pipeline()
train_model()
full_evaluation()
```

### Docker

```bash
# Build
docker build -t proyecto-mlops:latest .

# Run
docker run -v $(pwd)/data:/app/data proyecto-mlops:latest
```

### Publicar en PyPI

```bash
# 1. Tag versión
git tag v0.1.0
git push origin v0.1.0

# 2. GitHub Actions automáticamente:
#    - Detecta tag
#    - Build paquete
#    - Publish a TestPyPI
#    - Publish a PyPI
#    - Crear release

# 3. Verificar en PyPI
pip install proyecto-mlops
```

---

## 📊 Checklist de Archivos

### ✅ MANTENER (Necesarios)

- ✅ `proyecto_mlops/` - Código del paquete
- ✅ `docs_project/` - Documentación
- ✅ `infra/` - Docker y setup
- ✅ `config/` - Configuraciones
- ✅ `data/` - Datos del proyecto
- ✅ `models/` - Modelos entrenados
- ✅ `tests/` - Tests unitarios
- ✅ `.github/workflows/` - CI/CD
- ✅ `setup.py` - Instalación
- ✅ `pyproject.toml` - Config moderna
- ✅ `requirements.txt` - Dependencias
- ✅ `README.md` - Inicio rápido
- ✅ `LICENSE` - MIT
- ✅ `.gitignore` - Exclusiones git

### ❌ ELIMINADOS (No necesarios)

- ❌ `pipeline.py` - Monolítico (duplicado en módulos)
- ❌ `cli.py` - Viejo (reemplazado por `proyecto_mlops/cli.py`)
- ❌ `proyecto.ipynb` - Notebook antiguo
- ❌ `README.md` antiguo - Reemplazado por README_UPDATED.md
- ❌ `CHECKLIST_COMPLETACION.md` - Referencia histórica

### 📝 RENOMBRADOS

- 📝 `README_UPDATED.md` → `README.md` (principal)
- 📝 `cli_improved.py` → `proyecto_mlops/cli.py` (CLI oficial)
- 📝 Documentación → `docs_project/` (carpeta específica)
- 📝 `setup.py`, `pyproject.toml` → En `infra/` y raíz
- 📝 `Dockerfile`, `.dockerignore` → En `infra/`
- 📝 `Makefile` → En `infra/`

---

## 🎯 Próximos Pasos

1. **Push a GitHub**
   ```bash
   git add .
   git commit -m "Reorganizar proyecto - estructura limpia"
   git push origin main
   ```

2. **Configurar secrets en GitHub** (Settings → Secrets)
   - `DOCKER_USERNAME`
   - `DOCKER_PASSWORD`
   - `PYPI_API_TOKEN`
   - `TEST_PYPI_API_TOKEN`

3. **Crear primer release**
   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   ```

4. **Verificar workflows** en GitHub Actions

---

## 📞 Contacto

- **GitHub**: https://github.com/angelcast2002/PROYECTO-MLOPS
- **PyPI**: https://pypi.org/project/proyecto-mlops/
- **Docker Hub**: https://hub.docker.com/r/angelcast2002/proyecto-mlops
- **Email**: angelcast2002@gmail.com

---

**Última actualización**: Noviembre 3, 2025  
**Versión**: 0.1.0  
**Estado**: ✅ Estructura optimizada y documentada
