# pipeline.py
import os, json, ast, math, pickle, yaml, typer
import numpy as np, pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import joblib

app = typer.Typer(help="Pipeline NLP en Español")

# ---------- utilidades de texto ----------
import re, unicodedata
from nltk.corpus import stopwords
SPANISH_SW = set(stopwords.words("spanish"))
def normalize(txt:str)->str:
    txt = txt.lower()
    txt = "".join(c for c in unicodedata.normalize("NFD", txt) if unicodedata.category(c) != "Mn")
    txt = re.sub(r"[\r\n\t]+"," ", txt)
    txt = re.sub(r"\s+"," ", txt).strip()
    return txt
def tokenize_simple(txt:str):
    return re.findall(r"\b\w+\b", txt, flags=re.UNICODE)
def clean_tokens(tokens, remove_digits=True, remove_sw=True):
    out=[]
    for t in tokens:
        if remove_digits and t.isdigit(): continue
        if remove_sw and t in SPANISH_SW: continue
        out.append(t)
    return out

# ---------- helpers ----------
def load_cfg(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# ---------- etapas ----------
@app.command()
def preprocess(cfg_path: str = "config.yaml"):
    cfg = load_cfg(cfg_path)
    raw = cfg["paths"]["raw_csv"]
    outp = cfg["paths"]["processed_parquet"]
    os.makedirs(os.path.dirname(outp), exist_ok=True)

    df = pd.read_csv(raw).dropna(subset=["text"]).reset_index(drop=True)
    df["text_norm"] = df["text"].map(normalize)
    df["tokens"]    = df["text_norm"].map(tokenize_simple).map(clean_tokens)
    df.to_parquet(outp, index=False)
    typer.echo(f"✅ preprocess → {outp} ({df.shape})")

@app.command()
def featurize(cfg_path: str = "config.yaml", fast: int = 0):
    cfg = load_cfg(cfg_path)
    parq = cfg["paths"]["processed_parquet"]
    models_dir = cfg["paths"]["models_dir"]
    figs_dir   = cfg["paths"]["figures_dir"]
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(figs_dir, exist_ok=True)

    df = pd.read_parquet(parq)
    if fast > 0:
        df = df.sample(n=min(fast, len(df)), random_state=42)
    texts  = df["text_norm"].tolist()
    tokens = df["tokens"].tolist()

    # TF-IDF
    tfc = cfg["featurization"]["tfidf"]
    tfidf = TfidfVectorizer(ngram_range=(tfc["ngram_min"], tfc["ngram_max"]), min_df=tfc["min_df"])
    X_tfidf = tfidf.fit_transform(texts)
    joblib.dump(tfidf, os.path.join(models_dir, "tfidf_vectorizer.joblib"))

    # co-ocurrencias + PPMI
    if isinstance(tokens[0], str):
        tokens = [ast.literal_eval(t) for t in tokens]
    cc = cfg["featurization"]["cooc"]
    C, idx = cooc_matrix(tokens, window=cc["window"], min_count=cc["min_count"], top_k=cc["top_k"])
    X_ppmi = ppmi(C)
    np.save("data/processed/cooc.npy", C)
    np.save("data/processed/ppmi.npy", X_ppmi)
    with open("data/processed/cooc_idx.pkl","wb") as f: pickle.dump(idx, f)

    typer.echo(f"✅ featurize → TF-IDF {X_tfidf.shape} | PPMI {X_ppmi.shape}")

def cooc_matrix(tokenized_docs, window=4, min_count=5, top_k=None):
    vocab_counts = Counter(t for doc in tokenized_docs for t in doc)
    items = [(w,c) for w,c in vocab_counts.items() if c >= min_count]
    if top_k is not None and len(items) > top_k:
        items = sorted(items, key=lambda x: -x[1])[:top_k]
    idx = {w:i for i,(w,_) in enumerate(sorted(items, key=lambda x: -x[1]))}
    V = len(idx)
    C = np.zeros((V,V), dtype=np.float32)
    for doc in tokenized_docs:
        L = len(doc)
        for i,t in enumerate(doc):
            if t not in idx: continue
            wi = idx[t]
            left = max(0, i-window); right = min(L, i+window+1)
            for u in doc[left:i]:
                if u in idx: C[wi, idx[u]] += 1.0
            for u in doc[i+1:right]:
                if u in idx: C[wi, idx[u]] += 1.0
    return C, idx

def ppmi(C, eps=1e-8):
    total = C.sum()
    if total == 0: return C
    pi = C.sum(axis=1, keepdims=True)
    pj = C.sum(axis=0, keepdims=True)
    pij = C/(total+eps)
    denom = (pi @ pj) / (total**2 + eps)
    with np.errstate(divide='ignore'):
        pmi = np.log((pij+eps)/(denom+eps))
    return np.maximum(0.0, pmi)

@app.command()
def ngrams(cfg_path: str = "config.yaml"):
    cfg = load_cfg(cfg_path)
    parq = cfg["paths"]["processed_parquet"]
    df = pd.read_parquet(parq)
    tokens = df["tokens"].tolist()
    if isinstance(tokens[0], str):
        tokens = [ast.literal_eval(t) for t in tokens]

    train_sents, test_sents = train_test_split(tokens, test_size=0.1, random_state=42)
    BOS, EOS, UNK = "<s>", "</s>", "<unk>"
    def build_vocab(sents, min_count=3):
        cnt = Counter(w for s in sents for w in s)
        vocab = {w for w,c in cnt.items() if c >= min_count}
        vocab |= {BOS, EOS, UNK}
        return vocab
    def apply_vocab(sents, vocab):
        return [[w if w in vocab else UNK for w in s] for s in sents]
    vocab = build_vocab(train_sents, cfg["lm"]["min_count"])
    train_sents = apply_vocab(train_sents, vocab)
    test_sents  = apply_vocab(test_sents,  vocab)

    def add_bounds(s, n): return [BOS]*(n-1)+s+[EOS]
    def counts(sents, n):
        C=Counter(); CX=Counter()
        for s in sents:
            s=add_bounds(s,n)
            for i in range(n-1,len(s)):
                ng=tuple(s[i-n+1:i+1]); ctx=ng[:-1]
                C[ng]+=1; CX[ctx]+=1
        return C,CX
    def sent_log2(sent, n, C, CX, V, k):
        s=add_bounds(sent,n); log2p=0.0
        for i in range(n-1,len(s)):
            ng=tuple(s[i-n+1:i+1]); ctx=ng[:-1]
            c_ng=C.get(ng,0); c_ctx=CX.get(ctx,0)
            prob=(c_ng+k)/(c_ctx+k*V); log2p+=math.log2(prob)
        return log2p
    def metrics(sents, n, C, CX, V, k):
        N=sum(len(s)+1 for s in sents)
        logsum=sum(sent_log2(s,n,C,CX,V,k) for s in sents)
        H=-(logsum/N); ppl=2**H; return H,ppl

    res=[]
    V=len(vocab)
    for n in cfg["lm"]["ns"]:
        C,CX=counts(train_sents,n)
        for k in cfg["lm"]["k_values"]:
            H,ppl=metrics(test_sents,n,C,CX,V,k)
            res.append({"n":n,"k":k,"entropy_bits":float(H),"perplexity":float(ppl)})
    with open("data/processed/ngram_summary.json","w",encoding="utf-8") as f:
        json.dump({"vocab_size":len(vocab),"sweep":res},f,indent=2,ensure_ascii=False)
    typer.echo("✅ ngrams → data/processed/ngram_summary.json")

@app.command()
def train(cfg_path: str = "config.yaml"):
    cfg = load_cfg(cfg_path)
    parq = cfg["paths"]["processed_parquet"]
    models_dir = cfg["paths"]["models_dir"]
    figs_dir   = cfg["paths"]["figures_dir"]
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(figs_dir, exist_ok=True)

    df = pd.read_parquet(parq).dropna(subset=["text_norm","label"])
    Xtr, Xte, ytr, yte = train_test_split(
        df["text_norm"].astype(str).tolist(),
        df["label"].astype(str).tolist(),
        test_size=cfg["splits"]["test_size"],
        random_state=cfg["splits"]["random_state"],
        stratify=df["label"].astype(str).tolist()
    )
    # NB
    nb = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=2)), ("clf", MultinomialNB())]).fit(Xtr,ytr)
    p_nb = nb.predict(Xte)
    m_nb = {"accuracy": float(accuracy_score(yte,p_nb)), "f1_macro": float(f1_score(yte,p_nb,average="macro"))}
    joblib.dump(nb, os.path.join(models_dir,"nb_tfidf.joblib"))

    # SVM
    svm = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=2)), ("clf", LinearSVC())]).fit(Xtr,ytr)
    p_svm = svm.predict(Xte)
    m_svm = {"accuracy": float(accuracy_score(yte,p_svm)), "f1_macro": float(f1_score(yte,p_svm,average="macro"))}
    joblib.dump(svm, os.path.join(models_dir,"svm_tfidf.joblib"))

    with open("data/processed/classif_summary.json","w",encoding="utf-8") as f:
        json.dump({"NB":m_nb,"SVM":m_svm}, f, indent=2, ensure_ascii=False)

    typer.echo(f"✅ train → NB {m_nb} | SVM {m_svm}")

@app.command()
def all(cfg_path: str = "config.yaml", fast: int = 0):
    preprocess(cfg_path)
    featurize(cfg_path, fast=fast)
    ngrams(cfg_path)
    train(cfg_path)
    typer.echo("🚀 Pipeline completo.")

if __name__ == "__main__":
    app()
