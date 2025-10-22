# PROYECTO-01-NLP

## Resumen

Este repositorio contiene una canalización básica de procesamiento de texto y clasificación de documentos en español. Incluye pasos de preprocesamiento, extracción de características (BOW, TF-IDF, Word2Vec), entrenamiento de clasificadores (Naive Bayes y SVM) y visualizaciones (PCA, t-SNE).

El objetivo principal es explorar representaciones vectoriales de documentos y evaluar modelos simples para clasificación.

## Estructura del repositorio

- `pipeline.py` : Script principal con las etapas de preprocesado, vectorización, entrenamiento y evaluación.
- `proyecto.ipynb` : Notebook con experimentos, visualizaciones y código interactivo.
- `config.yaml` : Archivo de configuración con parámetros del pipeline.
- `data/` : Datos usados por el proyecto
  - `raw/` : Datos originales (CSV)
  - `processed/` : Archivos procesados y artefactos (matrices, índices, resúmenes)
- `figures/` : Gráficas generadas (PCA, t-SNE, matrices de confusión)
- `models/` : Modelos y vectores serializados (`.joblib`, `word2vec.model`)
- `tests/` : Pruebas unitarias para utilidades de texto

## Requisitos

Se asume un entorno con Python 3.8+ y las siguientes librerías principales:

- numpy
- pandas
- scikit-learn
- matplotlib
- gensim
- joblib
- spacy
- nltk

Instalación rápida (recomendado usar un entorno virtual):

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # PowerShell (Windows)
pip install -r requirements.txt  # si existe
```

Si no hay `requirements.txt`, instala las dependencias usadas en el notebook:

```bash
pip install numpy pandas scikit-learn matplotlib gensim joblib spacy nltk
python -m spacy download es_core_news_sm
```

## Uso

1. Coloca los datos en `data/raw/` (CSV).
2. Ejecuta `pipeline.py` o abre `proyecto.ipynb` para correr las celdas paso a paso.

Ejemplo mínimo con el script:

```bash
python pipeline.py --config config.yaml
```

El notebook `proyecto.ipynb` contiene celdas para:

- Cargar y preprocesar textos (tokenización, stemming/lemmatización)
- Construir vectores: BOW, TF-IDF, promedios Word2Vec
- Entrenar clasificadores y guardar modelos en `models/`
- Visualizar con PCA (TF-IDF) y t-SNE (Word2Vec)

## Notas sobre rendimiento

- La lemmatización con `spaCy` puede ser lenta si se procesa documento por documento. Para acelerar, procesar en lotes (`nlp.pipe`) y desactivar componentes innecesarios.
- El entrenamiento de `Word2Vec` también puede tardar; reducir `vector_size`, `epochs` o usar menos datos de muestra para experimentos rápidos.
- `t-SNE` es un método costoso. Use muestras pequeñas (p. ej. 200-500 puntos) o `init="pca"` y menos iteraciones para obtener resultados más rápidos.

## Artefactos generados

- `figures/pca_tfidf.png` — PCA de una muestra de TF-IDF
- `figures/tsne_w2v.png` — t-SNE de embeddings promedio Word2Vec
- `models/*.joblib` — vectorizadores y clasificadores serializados
- `models/word2vec.model` — modelo Word2Vec entrenado

## Tests

Hay pruebas en la carpeta `tests/` que pueden ejecutarse con `pytest`:

```bash
pip install pytest
pytest -q
```

## Extensiones y mejoras sugeridas

- Añadir un `requirements.txt` o `pyproject.toml` con versiones fijas.
- Documentar y parametrizar más opciones en `config.yaml`.
- Añadir notebooks de análisis exploratorio y notebooks que reproduzcan figuras para la entrega.
