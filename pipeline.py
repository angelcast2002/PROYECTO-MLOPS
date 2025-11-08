"""Compatibility shim for legacy imports.
Exposes normalize, tokenize_simple, clean_tokens for old tests.
Prefer importing from proyecto_mlops.data_preparation in new code.
"""
from proyecto_mlops.data_preparation import (
    normalize_text as normalize,
    tokenize_simple,
    clean_tokens as _clean_tokens,
)

# Preserve old parameter name remove_sw -> remove_stopwords

def clean_tokens(tokens, remove_digits=True, remove_sw=True):  # type: ignore
    return _clean_tokens(tokens, remove_digits=remove_digits, remove_stopwords=remove_sw)
