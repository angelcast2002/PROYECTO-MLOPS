"""Compatibility shim for legacy imports (tests).
Exposes normalize, tokenize_simple, clean_tokens for old tests.
Prefers proyecto_dp (standalone). If unavailable, uses a lightweight local fallback
that does not require external dependencies.
"""
try:
    from proyecto_dp import (
        normalize_text as normalize,
        tokenize_simple,
        clean_tokens as _dp_clean_tokens,
    )

    def clean_tokens(tokens, remove_digits=True, remove_sw=True):  # type: ignore
        return _dp_clean_tokens(tokens, remove_digits=remove_digits, remove_stopwords=remove_sw)

except Exception:
    # Minimal local fallback to keep tests running without proyecto_dp installed.
    import re
    import unicodedata

    _STOPWORDS_ES = {
        "de","la","que","el","en","y","a","los","del","se","las","por","un","para","con","no","una","su","al","lo","como","más","pero","sus","le","ya","o","este","sí","porque","esta","entre","cuando","muy","sin","sobre","también","me","hasta","hay","donde","quien","desde","todo","nos","durante","todos","uno","les","ni","contra","otros","ese","eso","ante","ellos","e","esto","mí","antes","algunos","qué","unos","yo","otro","otras","otra","él","tanto","esa","estos","mucho","quienes","nada","muchos","cual","poco","ella","estar","estas","algunas","algo","nosotros","mi","mis","tú","te","ti","tu","tus","ellas","nosotras","vosostros","vosostras","os","mío","mía","míos","mías","tuyo","tuya","tuyos","tuyas","suyo","suya","suyos","suyas","nuestro","nuestra","nuestros","nuestras","vuestro","vuestra","vuestros","vuestras","esos","esas","estoy","estás","está","estamos","estáis","están","esté","estés","estemos","estéis","estén","estaré","estarás","estará","estaremos","estaréis","estarán","estaría","estarías","estaríamos","estaríais","estarían","estaba","estabas","estábamos","estabais","estaban","estuve","estuviste","estuvo","estuvimos","estuvisteis","estuvieron","estuviera","estuvieras","estuviéramos","estuvierais","estuvieran","estuviese","estuvieses","estuviésemos","estuvieseis","estuviesen","estando","estado","estada","estados","estadas","estad"
    }

    def normalize(text: str) -> str:
        if text is None:
            return ""
        # Lower, strip, remove accents
        text = str(text).lower().strip()
        text = unicodedata.normalize("NFKD", text)
        text = "".join(c for c in text if not unicodedata.combining(c))
        # Collapse any whitespace (spaces, tabs, newlines) to a single space
        text = re.sub(r"\s+", " ", text).strip()
        return text

    _TOKEN_RE = re.compile(r"[a-záéíóúñü]+", re.IGNORECASE)

    def tokenize_simple(text: str):
        text = normalize(text)
        return _TOKEN_RE.findall(text)

    def clean_tokens(tokens, remove_digits: bool = True, remove_sw: bool = True):  # type: ignore
        out = []
        for tok in tokens or []:
            if remove_digits and any(ch.isdigit() for ch in tok):
                continue
            if remove_sw and tok in _STOPWORDS_ES:
                continue
            if len(tok) < 2:
                continue
            out.append(tok)
        return out
