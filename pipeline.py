# pipeline.py
# -*- coding: utf-8 -*-
"""
Pipeline de NLP + MLOps (español)
---------------------------------
Funciones reutilizables para:
- Preprocesamiento (normalize/tokenize/stemming/lemmatización opcional)
- Representaciones (BoW/TF-IDF, co-ocurrencia+PPMI, Word2Vec doc-avg)
- Modelos de lenguaje n-gramas + entropía/perplejidad
- Entrenamiento/evaluación (CV/holdout), latencia (SLA)
- Esquema/validación/versionado de datos, catálogo de features
- Sweep de hiperparámetros y registro de experimentos/modelos
- Predicción por CSV, monitoreo de drift (PSI) y gatillo de retraining

Diseñado para ser importado desde cli.py y usado también en notebooks.
"""

from __future__ import annotations

import os
import re
import json
import time
import math
import ast
import shutil
import hashlib
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple, Iterable, Optional

import numpy as np
import pandas as pd

from collections import Counter
from nltk.corpus import stopwords
from nltk.stem.snowball import SpanishStemmer

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix
)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.svm import LinearSVC
import joblib

# --- Constantes de rutas por defecto
DATA_RAW_CSV = "data/raw/dataset.csv"
PROCESSED_PARQUET = "data/processed/preprocesado.parquet"
MODELS_DIR = "models"
FIG_DIR = "figures"
DOCS_DIR = "docs"
PROCESSED_DIR = "data/processed"
REGISTRY_DATA_DIR = "data/registry"
MODEL_REGISTRY = os.path.join(MODELS_DIR, "registry.json")
EXP_LOG = os.path.join(PROCESSED_DIR, "exp_log.jsonl")

# --- Utilidades básicas

def ensure_dirs():
    for d in [os.path.dirname(DATA_RAW_CSV), PROCESSED_DIR, MODELS_DIR, FIG_DIR, DOCS_DIR, REGISTRY_DATA_DIR]:
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)

def set_seeds(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def save_json(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# --------------------------------------------------------------------------------------
# 1) Ingesta de datos + esquema + versionado
# --------------------------------------------------------------------------------------

def load_dataset(csv_path: str = DATA_RAW_CSV) -> pd.DataFrame:
    """Lee un CSV y valida columnas mínimas."""
    df = pd.read_csv(csv_path)
    assert {"text", "label"}.issubset(df.columns), f"Columnas esperadas faltantes en {csv_path}"
    return df

def make_data_schema_from_df(df: pd.DataFrame, version: str = "1.0") -> Dict:
    """Crea un contrato de datos simple (usado para validar futuras ingestas)."""
    return {
        "version": version,
        "dataset_name": "news_es",
        "columns": {
            "text":  {"dtype": "string", "required": True, "min_length": 10},
            "label": {"dtype": "string", "required": True,
                      "allowed": sorted(df["label"].astype(str).unique().tolist())}
        },
        "allow_extra_columns": False
    }

def validate_dataset_against_schema(df: pd.DataFrame, schema: Dict) -> Dict:
    """Valida un DataFrame contra el esquema dado. Retorna un reporte con errores."""
    report = {"total_rows": int(len(df)), "errors": []}

    # columnas presentes
    expected_cols = set(schema["columns"].keys())
    present_cols  = set(df.columns)
    missing = expected_cols - present_cols
    extra   = present_cols - expected_cols
    if missing:
        report["errors"].append({"type": "missing_columns", "cols": sorted(missing)})
    if extra and not schema.get("allow_extra_columns", False):
        report["errors"].append({"type": "extra_columns", "cols": sorted(extra)})

    # reglas por columna
    if "text" in df:
        nulls = int(df["text"].isna().sum())
        short = int((df["text"].astype(str).str.len() < schema["columns"]["text"]["min_length"]).sum())
        if nulls: report["errors"].append({"type": "text_nulls", "count": nulls})
        if short: report["errors"].append({"type": "text_too_short", "count": short})

    if "label" in df:
        allowed = set(schema["columns"]["label"]["allowed"])
        invalid = df["label"].astype(str).map(lambda x: x not in allowed).sum()
        if invalid: report["errors"].append({"type": "invalid_labels", "count": int(invalid)})

    return report

def _sha256(path: str, chunk: int = 1<<20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()

def register_dataset(src_csv: str = DATA_RAW_CSV, note: str = "base") -> Dict:
    """Crea un snapshot versionado del dataset con metadatos y actualiza el registro."""
    ensure_dirs()
    df = pd.read_csv(src_csv)

    reg_path = os.path.join(REGISTRY_DATA_DIR, "registry.json")
    registry = {"datasets": []}
    if os.path.exists(reg_path):
        registry = load_json(reg_path)

    version = len(registry["datasets"]) + 1
    vdir = os.path.join(REGISTRY_DATA_DIR, f"v{version}")
    os.makedirs(vdir, exist_ok=True)

    dst_csv = os.path.join(vdir, "dataset.csv")
    shutil.copy2(src_csv, dst_csv)

    schema_src = os.path.join(DOCS_DIR, "data_schema.json")
    if os.path.exists(schema_src):
        shutil.copy2(schema_src, os.path.join(vdir, "data_schema.json"))

    digest = _sha256(dst_csv)
    cls_dist = df["label"].astype(str).value_counts(normalize=True).round(4).to_dict()

    meta = {
        "version": version,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "rows": int(len(df)),
        "sha256": digest,
        "class_dist": cls_dist,
        "note": note
    }
    save_json(meta, os.path.join(vdir, "meta.json"))

    registry["datasets"].append({"version": version, "path": vdir, "sha256": digest})
    save_json(registry, reg_path)

    return meta

def ingest_new_csv(new_csv_path: str, note: str = "nuevo") -> Dict:
    """Valida un nuevo CSV contra el esquema y, si pasa, lo instala como dataset activo y versiona."""
    schema = load_json(os.path.join(DOCS_DIR, "data_schema.json"))
    df_new = pd.read_csv(new_csv_path)

    # Validación mínima
    rep = validate_dataset_against_schema(df_new, schema)
    assert not rep["errors"], f"El nuevo CSV no cumple el esquema: {rep['errors']}"

    shutil.copy2(new_csv_path, DATA_RAW_CSV)
    return register_dataset(DATA_RAW_CSV, note=note)

# --------------------------------------------------------------------------------------
# 2) Preprocesamiento
# --------------------------------------------------------------------------------------

SPANISH_SW = set(stopwords.words("spanish"))
stemmer = SpanishStemmer()

def normalize(txt: str) -> str:
    txt = txt.lower()
    txt = "".join(c for c in unicodedata.normalize("NFD", txt) if unicodedata.category(c) != "Mn")
    txt = re.sub(r"[\r\n\t]+", " ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt

def tokenize_simple(txt: str) -> List[str]:
    return re.findall(r"\b\w+\b", txt, flags=re.UNICODE)

def clean_tokens(tokens: Iterable[str], remove_digits: bool = True, remove_sw: bool = True) -> List[str]:
    out = []
    for t in tokens:
        if remove_digits and str(t).isdigit():
            continue
        if remove_sw and t in SPANISH_SW:
            continue
        out.append(t)
    return out

def lemmatize_tokens(tokens: List[str], nlp=None) -> List[str]:
    """Lematiza con spaCy si se pasa un objeto nlp; si no, retorna los tokens."""
    if nlp is None:
        return tokens
    doc = nlp(" ".join(tokens))
    return [t.lemma_ if t.lemma_ else t.text for t in doc]

def preprocess_dataframe(df: pd.DataFrame, use_spacy: bool = False, spacy_model: str = "es_core_news_sm") -> pd.DataFrame:
    """Agrega columnas text_norm, tokens, stems, lemmas. No guarda a disco a menos que lo expliques fuera."""
    d = df.copy()
    d["text_norm"] = d["text"].astype(str).map(normalize)
    d["tokens"] = d["text_norm"].map(tokenize_simple).map(clean_tokens)
    d["stems"] = d["tokens"].map(lambda toks: [stemmer.stem(t) for t in toks])

    if use_spacy:
        try:
            import es_core_news_sm  # noqa
            import spacy
            nlp = es_core_news_sm.load()
        except Exception:
            nlp = None
    else:
        nlp = None

    d["lemmas"] = d["tokens"].map(lambda t: lemmatize_tokens(t, nlp))
    return d

def save_preprocessed_parquet(df: pd.DataFrame, out_path: str = PROCESSED_PARQUET):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path, index=False)

# --------------------------------------------------------------------------------------
# 3) Representaciones (BoW/TF-IDF, co-ocurrencia+PPMI, Word2Vec)
# --------------------------------------------------------------------------------------

def build_bow_tfidf(texts: List[str],
                    bow_ngrams=(1,1),
                    tfidf_ngrams=(1,2),
                    tfidf_min_df=2):
    bow = CountVectorizer(ngram_range=bow_ngrams)
    tfidf = TfidfVectorizer(ngram_range=tfidf_ngrams, min_df=tfidf_min_df)
    X_bow = bow.fit_transform(texts)
    X_tfidf = tfidf.fit_transform(texts)
    return (bow, X_bow), (tfidf, X_tfidf)

def cooc_matrix(tokenized_docs: List[List[str]], window: int = 4, min_count: int = 5, top_k: Optional[int] = None):
    vocab_counts = Counter(t for doc in tokenized_docs for t in doc)
    items = [(w, c) for w, c in vocab_counts.items() if c >= min_count]
    if top_k is not None and len(items) > top_k:
        items = sorted(items, key=lambda x: -x[1])[:top_k]
    idx = {w: i for i, (w, _) in enumerate(sorted(items, key=lambda x: -x[1]))}
    V = len(idx)
    C = np.zeros((V, V), dtype=np.float32)
    for doc in tokenized_docs:
        L = len(doc)
        for i, t in enumerate(doc):
            if t not in idx:
                continue
            wi = idx[t]
            left = max(0, i - window)
            right = min(L, i + window + 1)
            for u in doc[left:i]:
                if u in idx:
                    C[wi, idx[u]] += 1.0
            for u in doc[i+1:right]:
                if u in idx:
                    C[wi, idx[u]] += 1.0
    return C, idx

def ppmi(C: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    total = C.sum()
    if total == 0:
        return C
    pi = C.sum(axis=1, keepdims=True)
    pj = C.sum(axis=0, keepdims=True)
    pij = C / (total + eps)
    denom = (pi @ pj) / (total**2 + eps)
    with np.errstate(divide='ignore'):
        pmi = np.log((pij + eps) / (denom + eps))
    return np.maximum(0.0, pmi)

def train_word2vec_docavg(tokenized_docs: List[List[str]],
                          vector_size: int = 100,
                          window: int = 5,
                          min_count: int = 5,
                          epochs: int = 5,
                          workers: int = 2,
                          sg: int = 1,
                          seed: int = 42):
    from gensim.models import Word2Vec
    # limpieza defensiva
    docs = []
    for doc in tokenized_docs:
        if isinstance(doc, str):
            seq = doc.split()
        else:
            try:
                seq = list(doc)
            except Exception:
                continue
        seq = [str(x).strip() for x in seq if str(x).strip()]
        if len(seq) >= 2:
            docs.append(seq)

    w2v = Word2Vec(
        vector_size=vector_size, window=window, min_count=min_count,
        workers=workers, sg=sg, epochs=epochs, seed=seed, compute_loss=False
    )
    w2v.build_vocab(docs, progress_per=10000)
    w2v.train(corpus_iterable=docs, total_examples=w2v.corpus_count, epochs=w2v.epochs)

    dim = w2v.wv.vector_size
    out = []
    for doc in docs:
        vecs = [w2v.wv[w] for w in doc if w in w2v.wv]
        out.append(np.mean(vecs, axis=0) if len(vecs) else np.zeros(dim))
    X_doc = np.vstack(out)
    return w2v, X_doc

# --------------------------------------------------------------------------------------
# 4) Modelos probabilísticos (N-gramas)
# --------------------------------------------------------------------------------------

BOS, EOS, UNK = "<s>", "</s>", "<unk>"

def build_vocab_ng(sents: List[List[str]], min_count: int = 3) -> set:
    cnt = Counter(w for s in sents for w in s)
    vocab = {w for w, c in cnt.items() if c >= min_count}
    return vocab | {BOS, EOS, UNK}

def apply_vocab_ng(sents: List[List[str]], vocab: set) -> List[List[str]]:
    out = []
    for s in sents:
        out.append([w if w in vocab else UNK for w in s])
    return out

def add_boundaries(sents: List[List[str]], n: int) -> List[List[str]]:
    return [[BOS]*(n-1) + s + [EOS] for s in sents]

def ngram_counts(sents: List[List[str]], n: int):
    sents_b = add_boundaries(sents, n)
    counts = Counter()
    ctx_counts = Counter()
    for s in sents_b:
        for i in range(n-1, len(s)):
            ngram = tuple(s[i-n+1:i+1])
            ctx = ngram[:-1]
            counts[ngram] += 1
            ctx_counts[ctx] += 1
    return counts, ctx_counts

def sent_log2prob(sent: List[str], n: int, counts, ctx_counts, V: int, k: float = 1.0) -> float:
    s = [BOS]*(n-1) + sent + [EOS]
    log2p = 0.0
    for i in range(n-1, len(s)):
        ngram = tuple(s[i-n+1:i+1])
        ctx = ngram[:-1]
        c_ng = counts.get(ngram, 0)
        c_ctx = ctx_counts.get(ctx, 0)
        prob = (c_ng + k) / (c_ctx + k*V)
        log2p += math.log2(prob)
    return log2p

def corpus_metrics(sents: List[List[str]], n: int, counts, ctx_counts, V: int, k: float = 1.0) -> Tuple[float, float]:
    N_tokens = sum(len(s)+1 for s in sents)  # +EOS
    log2sum = 0.0
    for s in sents:
        log2sum += sent_log2prob(s, n, counts, ctx_counts, V, k)
    H = -log2sum / N_tokens
    ppl = 2 ** H
    return H, ppl

def ngram_sweep(tokens: List[List[str]], test_size: float = 0.1, seed: int = 42):
    tr, te = train_test_split(tokens, test_size=test_size, random_state=seed)
    vocab = build_vocab_ng(tr, min_count=3)
    tr = apply_vocab_ng(tr, vocab)
    te = apply_vocab_ng(te, vocab)
    out = []
    for n in [1,2,3]:
        cnts, ctx = ngram_counts(tr, n)
        for k in [0.1, 0.5, 1.0]:
            H, ppl = corpus_metrics(te, n, cnts, ctx, V=len(vocab), k=k)
            out.append({"n": n, "k": k, "entropy_bits": float(H), "perplexity": float(ppl)})
    return out

# --------------------------------------------------------------------------------------
# 5) Clasificación: pipeline, CV/holdout, latencia
# --------------------------------------------------------------------------------------

def make_pipe(min_df: int = 2, C: float = 1.0) -> Pipeline:
    """Pipeline estándar TF-IDF(1,2)+LinearSVC (misma lógica que notebook/CLI)."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=min_df)),
        ("clf", LinearSVC(C=C))
    ])

def split_train_test(texts: List[str], labels: List[str], test_size: float = 0.2, seed: int = 42):
    return train_test_split(texts, labels, test_size=test_size, random_state=seed, stratify=labels)

def train_and_eval_holdout(texts: List[str], labels: List[str], min_df: int = 2, C: float = 1.0):
    X_tr, X_te, y_tr, y_te = split_train_test(texts, labels)
    pipe = make_pipe(min_df=min_df, C=C).fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    acc = accuracy_score(y_te, pred)
    f1m = f1_score(y_te, pred, average="macro")
    labels_sorted = sorted(set(labels))
    cm = confusion_matrix(y_te, pred, labels=labels_sorted)
    return pipe, {"accuracy": float(acc), "f1_macro": float(f1m), "labels": labels_sorted, "cm": cm.tolist()}

def cross_val_summary(texts: List[str], labels: List[str], cv: int = 5, min_df: int = 2, C: float = 1.0, seed: int = 42) -> Dict:
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    pipe = make_pipe(min_df=min_df, C=C)
    accs, f1s, reports = [], [], []
    X = np.array(texts, dtype=object)
    y = np.array(labels, dtype=object)
    for tr, te in skf.split(X, y):
        pipe.fit(X[tr].tolist(), y[tr].tolist())
        pred = pipe.predict(X[te].tolist())
        accs.append(accuracy_score(y[te], pred))
        f1s.append(f1_score(y[te], pred, average="macro"))
        rep = classification_report(y[te], pred, zero_division=0, output_dict=True)
        reports.append(rep)
    # f1 por clase
    all_labels = sorted({c for rep in reports for c in rep if c not in ("accuracy", "macro avg", "weighted avg")})
    f1_per_class = {}
    for cls in all_labels:
        vals = [rep[cls]["f1-score"] for rep in reports if cls in rep]
        if vals:
            f1_per_class[cls] = float(np.mean(vals))
    return {
        "cv": cv,
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "f1_macro_mean": float(np.mean(f1s)),
        "f1_macro_std": float(np.std(f1s)),
        "per_class_f1_mean": f1_per_class
    }

def measure_latency(pipe: Pipeline, ejemplos: List[str], repeat_each: int = 70) -> Dict:
    """Mide latencia single-shot sobre textos cortos (aprox como en demo/SLA)."""
    samples = ejemplos * repeat_each
    _ = pipe.predict([ejemplos[0]])  # warm-up
    times = []
    for s in samples:
        t0 = time.perf_counter()
        _ = pipe.predict([s])[0]
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    p50 = float(np.percentile(times, 50))
    p95 = float(np.percentile(times, 95))
    return {"n_preds": len(times), "latencia_p50_ms": round(p50, 3), "latencia_p95_ms": round(p95, 3)}

# --------------------------------------------------------------------------------------
# 6) Registro de experimentos y modelos (versión/firma)
# --------------------------------------------------------------------------------------

def log_experiment(run: Dict, log_path: str = EXP_LOG):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(run, ensure_ascii=False) + "\n")

def register_model(pipe: Pipeline, base_name: str = "svm_tfidf") -> str:
    """Guarda modelo con versión autoincremental y actualiza models/registry.json."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    reg = {"versions": []}
    if os.path.exists(MODEL_REGISTRY):
        reg = load_json(MODEL_REGISTRY)
    version = len(reg["versions"]) + 1
    path = os.path.join(MODELS_DIR, f"{base_name}_v{version}.joblib")
    joblib.dump(pipe, path)
    entry = {
        "version": version,
        "path": path,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "signature": {"input": "str(text)", "output": "str(label)"}
    }
    reg["versions"].append(entry)
    save_json(reg, MODEL_REGISTRY)
    return path

# --------------------------------------------------------------------------------------
# 7) Predicción por CSV, catálogo de features, sweep HP
# --------------------------------------------------------------------------------------

def predict_csv(in_path: str, out_path: str, model_path: Optional[str] = None):
    if model_path is None:
        # el más reciente por mtime
        cand = [p for p in os.listdir(MODELS_DIR) if p.endswith(".joblib")]
        assert cand, "No hay modelos en models/"
        model_path = max(cand, key=lambda p: os.path.getmtime(os.path.join(MODELS_DIR, p)))
        model_path = os.path.join(MODELS_DIR, model_path)
    clf = joblib.load(model_path)
    df = pd.read_csv(in_path)
    assert "text" in df.columns, "El CSV debe tener columna 'text'."
    df["pred"] = clf.predict(df["text"].astype(str).tolist())
    df.to_csv(out_path, index=False)
    return out_path, model_path

def write_features_catalog():
    catalogo = {
        "preprocesamiento": {
            "lowercase": True, "strip_accents": True,
            "stopwords": "spanish", "digits_removed": True,
            "stemming": "Snowball(Spanish)", "lemmatization": "spaCy (opcional)"
        },
        "representaciones": {
            "bow_tfidf": {"ngram_range": [1,2], "min_df": 2},
            "cooc_ppmi": {"window": 4, "min_count": 5},
            "word2vec": {"sg": 1, "vector_size": 100, "window": 5, "min_count": 5, "epochs": 5}
        },
        "clasificador": {"name": "LinearSVC", "features": "TF-IDF(1,2), min_df=2"}
    }
    save_json(catalogo, os.path.join(DOCS_DIR, "features_catalog.json"))
    return catalogo

def hp_sweep(texts: List[str], labels: List[str], grid: List[Dict], cv: int = 3, seed: int = 42) -> List[Dict]:
    results = []
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    X = np.array(texts, dtype=object)
    y = np.array(labels, dtype=object)
    for hp in grid:
        f1s = []
        for tr, te in skf.split(X, y):
            pipe = make_pipe(min_df=hp["min_df"], C=hp.get("C", 1.0))
            pipe.fit(X[tr].tolist(), y[tr].tolist())
            pred = pipe.predict(X[te].tolist())
            f1s.append(f1_score(y[te], pred, average="macro"))
        results.append({**hp, "f1_macro_mean": float(np.mean(f1s))})
        log_experiment({
            "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": "LinearSVC",
            "feats": f"TFIDF(1,2), min_df={hp['min_df']}",
            "cv": cv, "params": {"C": hp.get("C", 1.0)},
            "metrics": {"f1_macro_mean": float(np.mean(f1s))}
        })
    return sorted(results, key=lambda r: -r["f1_macro_mean"])

# --------------------------------------------------------------------------------------
# 8) SLA y metas (goals), validación automática
# --------------------------------------------------------------------------------------

def write_goals(goals: Dict, path: str = os.path.join(DOCS_DIR, "objetivo_y_slas.json")) -> Dict:
    save_json(goals, path)
    return goals

def assert_goals(goals_path: str,
                 cv_summary_path: str = os.path.join(PROCESSED_DIR, "cv_summary.json"),
                 latency_summary_path: str = os.path.join(PROCESSED_DIR, "latency_summary.json")):
    goals = load_json(goals_path)
    cv = load_json(cv_summary_path)
    lat = load_json(latency_summary_path)
    f1_target = goals["metas"]["f1_macro_min"]
    assert cv["f1_macro_mean"] >= f1_target, f"Meta F1-macro NO cumplida: {cv['f1_macro_mean']:.4f} < {f1_target}"
    if cv.get("per_class_f1_mean"):
        f1_min_cls = min(cv["per_class_f1_mean"].values())
        assert f1_min_cls >= goals["metas"]["f1_por_clase_min"], \
            f"Fairness NO cumplida: F1 mínima {f1_min_cls:.4f} < {goals['metas']['f1_por_clase_min']}"
    assert lat["latencia_p95_ms"] <= goals["sla_demo"]["latencia_p95_ms"], \
        f"SLA de latencia NO cumplido: p95={lat['latencia_p95_ms']} ms"

# --------------------------------------------------------------------------------------
# 9) Monitoreo: drift (PSI) y bandera de retraining
# --------------------------------------------------------------------------------------

def population_stability_index(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sum((p - q) * np.log((p + 1e-12) / (q + 1e-12))))

def drift_report_and_flag(base_labels: pd.Series,
                          pred_labels: pd.Series,
                          goals_path: str = os.path.join(DOCS_DIR, "objetivo_y_slas.json"),
                          psi_threshold: float = 0.2) -> Dict:
    base_dist = (base_labels.astype(str).value_counts(normalize=True)).sort_index()
    pred_dist = (pred_labels.astype(str).value_counts(normalize=True)).sort_index()
    labels_union = sorted(set(base_dist.index) | set(pred_dist.index))
    p = np.array([base_dist.get(lbl, 1e-6) for lbl in labels_union])
    q = np.array([pred_dist.get(lbl, 1e-6) for lbl in labels_union])
    psi = population_stability_index(p, q)

    save_json({
        "labels": labels_union,
        "base_dist": {k: float(base_dist.get(k, 0.0)) for k in labels_union},
        "pred_dist": {k: float(pred_dist.get(k, 0.0)) for k in labels_union},
        "psi": psi
    }, os.path.join(PROCESSED_DIR, "drift_report.json"))

    goals = load_json(goals_path)
    reasons = []
    if psi > psi_threshold:
        reasons.append("drift_moderado_o_mayor")

    flag = {"retrain": bool(reasons), "reasons": reasons}
    save_json(flag, os.path.join(PROCESSED_DIR, "retrain_flag.json"))
    return flag

# --------------------------------------------------------------------------------------
# 10) Helpers para notebook/CLI (carga-preprocesado, entrenamiento end-to-end)
# --------------------------------------------------------------------------------------

def load_or_preprocess(csv_path: str = DATA_RAW_CSV,
                       parquet_path: str = PROCESSED_PARQUET,
                       use_spacy: bool = False) -> pd.DataFrame:
    """Carga parquet si existe; si no, lee CSV, preprocesa y guarda parquet."""
    if os.path.exists(parquet_path):
        return pd.read_parquet(parquet_path)
    df = load_dataset(csv_path)
    d = preprocess_dataframe(df, use_spacy=use_spacy)
    save_preprocessed_parquet(d, parquet_path)
    return d

def train_final_and_register(df: pd.DataFrame,
                             min_df: int = 2, C: float = 1.0,
                             save_path: str = os.path.join(MODELS_DIR, "svm_tfidf.joblib")) -> str:
    texts = (df["text_norm"] if "text_norm" in df.columns else df["text"]).astype(str).tolist()
    labels = df["label"].astype(str).tolist()
    pipe = make_pipe(min_df=min_df, C=C).fit(texts, labels)
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(pipe, save_path)
    # también registrar versión
    register_model(pipe, base_name="svm_tfidf")
    return save_path
