# -*- coding: utf-8 -*-
from __future__ import annotations
import re, html
from typing import Dict, List, Tuple, Any

# -------- 文本清洗 --------
TAG_RE = re.compile(r"<[^>]+>")
SCRIPT_STYLE_RE = re.compile(r"(?is)<(script|style).*?>.*?</\1>")
MULTI_WS_RE = re.compile(r"\s{2,}")
EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF]")

# 按优先级尝试聚合的 JD 字段
DESC_FIELDS_ORDER = [
    "description","jobDescription","detail","jd","content",
    "text","desc","requirement","requirements","duty","responsibility"
]

def strip_html(x: str) -> str:
    if not x: return ""
    t = html.unescape(x)
    t = SCRIPT_STYLE_RE.sub(" ", t)
    t = TAG_RE.sub(" ", t)
    t = EMOJI_RE.sub(" ", t)
    t = MULTI_WS_RE.sub(" ", t).strip()
    return t

def _to_text(v: Any) -> str:
    if v is None: return ""
    if isinstance(v, (list, tuple, set)):
        return " ".join([_to_text(i) for i in v])
    if isinstance(v, dict):
        return " ".join([_to_text(x) for x in v.values()])
    return str(v)

def merge_job_description(row: Dict[str, Any]) -> Tuple[str, float, List[str]]:
    """
    返回：(清洗后的JD文本, 质量分[0..1], 使用到的字段列表)
    质量分=长度覆盖+关键词命中-噪声惩罚
    """
    parts, sources = [], []
    for f in DESC_FIELDS_ORDER:
        v = _to_text(row.get(f, ""))
        if v and len(v) >= 10 and v.lower() not in ("nan","none","null"):
            sources.append(f); parts.append(v)

    if not parts:  # 兜底
        v = _to_text(row.get("title") or row.get("jobName"))
        if v: parts.append(v)

    text = strip_html("\n\n".join(parts))
    L, score = len(text), 0.0
    if L >= 60:  score += 0.4
    if L >= 200: score += 0.2
    if re.search(r"(岗位职责|任职要求|Responsibilit(y|ies)|Requirement[s]?)", text, flags=re.I): score += 0.2
    if re.search(r"(公司介绍|投递|简历|福利|五险一金)", text): score += 0.1
    if re.search(r"(微信|QQ|群|扫码|公众号)", text): score -= 0.2
    score = max(0.0, min(1.0, score))
    return text, score, sources

# -------- 学历规范化 --------
EDU_CANONICAL = {
    "博士": ["博士","phd","doctor","doctoral"],
    "硕士": ["硕士","研究生","master","msc","m.s.","m.eng","mba","mpa","mem"],
    "本科": ["本科","学士","bachelor","b.s.","beng","全日制本科","一本","二本"],
    "大专": ["大专","专科","高职","associate","college"],
    "高中/中专": ["高中","中专","中技","high school"],
    "不限": ["不限","不限制","不限学历","学历不限","no limit","any"],
}

def _normalize_edu(s: str) -> str:
    if not s: return ""
    t = _to_text(s).strip().lower()
    for canon, keys in EDU_CANONICAL.items():
        for k in keys:
            if k.lower() in t:
                return canon
    return ""

def infer_education(structured_text: str, jd_text: str) -> Tuple[str, float, str]:
    """
    返回：(规范学历, 置信度0..1, 来源 'structured|jd_regex' 或空)
    """
    edu = _normalize_edu(structured_text)
    if edu:
        return edu, 0.95, "structured"
    edu_jd = _normalize_edu(jd_text)
    if edu_jd:
        return edu_jd, (0.85 if edu_jd in ("硕士","博士") else 0.8), "jd_regex"
    return "", 0.0, ""

def enrich_job_record(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    对一条职位数据做增强（不改原 dict）：JD 合并清洗 + 学历推断
    """
    row = dict(raw)
    jd, jd_quality, srcs = merge_job_description(row)
    edu_norm, edu_conf, edu_src = infer_education(
        _to_text(row.get("education") or row.get("edu") or ""),
        jd
    )
    row["jobDescription"] = jd
    row["descQuality"]    = jd_quality
    row["descSource"]     = ",".join(srcs)
    row["eduNormalized"]  = edu_norm
    row["eduConfidence"]  = edu_conf
    row["eduSource"]      = edu_src
    return row
