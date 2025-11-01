
# -*- coding: utf-8 -*-
import re
import json
from typing import List, Dict, Iterable, Set, Tuple
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\u3000", " ").replace("\xa0", " ")
    s = re.sub(r"[，。、“”：；！【】（）—…·\-\–\—\/\\\|\[\]\(\)<>~`~!@#\$%\^&\*\+=?,.:;]", " ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

def load_skills_lexicon(path: str) -> List[str]:
    p = Path(path)
    if not p.exists():
        return []
    skills = []
    for line in p.read_text(encoding="utf-8").splitlines():
        t = line.strip()
        if not t or t.startswith("#"):
            continue
        skills.append(t.lower())
    return sorted(set(skills))

def _compile_skill_regex(skills: List[str]):
    import re as _re
    pats = [re.escape(k) for k in skills if k]
    if not pats:
        pats = ["python"]
    return _re.compile(r"\b(" + "|".join(pats) + r")\b", flags=_re.I)

def extract_skills_from_texts(texts: Iterable[str], skills_lexicon: List[str], top_k_fallback: int = 10) -> List[Set[str]]:
    regex = _compile_skill_regex(skills_lexicon) if skills_lexicon else None
    norm_texts = [normalize_text(t) for t in texts]
    results: List[Set[str]] = []

    tfidf = TfidfVectorizer(ngram_range=(1,2), min_df=1, max_features=5000)
    try:
        X = tfidf.fit_transform(norm_texts)
        vocab = {v:k for k,v in tfidf.vocabulary_.items()}
    except Exception:
        X = None; vocab = {}

    for i, t in enumerate(norm_texts):
        hits = set()
        if regex:
            for m in regex.finditer(t):
                hits.add(m.group(0).lower())
        if not hits and X is not None:
            row = X.getrow(i)
            if row.nnz > 0:
                nz = list(zip(row.indices, row.data))
                nz.sort(key=lambda x: x[1], reverse=True)
                for idx, val in nz[:top_k_fallback]:
                    tok = vocab.get(idx, "")
                    if tok and len(tok) > 1:
                        hits.add(tok.lower())
        results.append(hits)
    return results

def _sha1(s: str) -> str:
    import hashlib as _h
    return _h.sha1(s.encode("utf-8")).hexdigest()

def compute_job_key(row: dict) -> str:
    city_list = row.get("jobCityList", "[]")
    if isinstance(city_list, str):
        try:
            import json as _json
            cities = _json.loads(city_list)
        except Exception:
            cities = [city_list]
    else:
        cities = city_list
    base = "|".join([
        str(row.get("companyId", "")),
        str(row.get("bossUid", "")),
        str(row.get("jobName", "")),
        str(row.get("graduationYear", "")),
        str(row.get("eduLevel", "")),
        str(row.get("salaryType", "")),
        str(row.get("salaryMin", "")),
        str(row.get("salaryMax", "")),
        str(row.get("salaryMonth", "")),
        ",".join(sorted([str(c) for c in (cities or [])]))
    ])
    return _sha1(base)

def build_skill_set(job_csv_path: str, lexicon_path: str, output_json_path: str = None) -> str:
    df = pd.read_csv(job_csv_path)
    df["desc"] = (df.get("ext","").astype(str) + " " + df.get("jobName","").astype(str)).fillna("")
    skills = load_skills_lexicon(lexicon_path)
    skill_sets = extract_skills_from_texts(df["desc"].tolist(), skills)
    keys = []
    for _, row in df.iterrows():
        keys.append(compute_job_key(row.to_dict()))
    mapping = {k: sorted(list(s)) for k, s in zip(keys, skill_sets)}
    if output_json_path is None:
        from pathlib import Path as _P
        output_json_path = str(_P(job_csv_path).with_suffix(".skills.json"))
    Path(output_json_path).write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_json_path
