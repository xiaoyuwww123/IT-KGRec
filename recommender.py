# -*- coding: utf-8 -*-
from typing import List, Dict, Any, Optional, Tuple
import re, string
from neo4j import GraphDatabase, basic_auth



def _connect(uri: str, no_auth: bool, user: str, password: str):
    return GraphDatabase.driver(uri, auth=None if no_auth else basic_auth(user, password))

def _tok_table():
    extra = "，。；：！？、（）【】《》“”‘’"
    return str.maketrans({c:" " for c in (string.punctuation + extra)})

def _tokenize(text: str) -> List[str]:
    if not text: return []
    s = str(text).lower().translate(_tok_table())
    s = re.sub(r"\s+", " ", s).strip()
    words, buff = [], []
    for ch in s:
        if '\u4e00' <= ch <= '\u9fff':
            words.append(ch)          # 中文以字为单位
        else:
            buff.append(ch)           # 英文/数字保留
    if buff:
        words.extend([w for w in " ".join(buff).split(" ") if len(w) > 1])
    return [w for w in words if w]

# —— 保留你原有的接口 —— #
def recommend_jobs(student_skills: List[str], city: str = "ALL", topk: int = 20,
                   uri: str = "bolt://localhost:7687", database: str = "neo4j",
                   no_auth: bool = True, user: str = "neo4j", password: str = "neo4j") -> List[Dict[str, Any]]:
    drv = _connect(uri, no_auth, user, password)
    ss = [s.strip().lower() for s in student_skills if s.strip()]
    rows: List[Dict[str, Any]] = []
    with drv.session(database=database) as sess:
        q = """
        MATCH (j:Job)
        OPTIONAL MATCH (j)-[:LOCATED_IN]->(ci:City)
        OPTIONAL MATCH (j)-[:REQUIRES_SKILL]->(s:Skill)
        WITH j, collect(DISTINCT toLower(s.name)) AS reqs, collect(DISTINCT ci.name) AS cities
        RETURN j.jobKey AS jobKey,
               coalesce(j.title, j.jobName) AS title,
               reqs AS reqs, cities AS cities,
               size([x IN reqs WHERE x IN $ss]) AS matched,
               size(reqs) AS total
        ORDER BY matched DESC, total ASC
        LIMIT $k
        """
        rows = [r.data() for r in sess.run(q, ss=ss, k=topk)]
    for r in rows:
        r["score"] = (r.get("matched") or 0) / (max(1, r.get("total") or 1))
    drv.close()
    return rows

def recommend_courses_for_gap(student_skills: List[str], jobKey: str, topk: int = 10,
                              uri: str = "bolt://localhost:7687", database: str = "neo4j",
                              no_auth: bool = True, user: str = "neo4j", password: str = "neo4j") -> List[Dict[str, Any]]:
    drv = _connect(uri, no_auth, user, password)
    ss = [s.strip().lower() for s in student_skills if s.strip()]
    with drv.session(database=database) as sess:
        qmiss = """
        MATCH (j:Job {jobKey:$jobKey})-[:REQUIRES_SKILL]->(s:Skill)
        WITH collect(DISTINCT toLower(s.name)) AS reqs
        RETURN [x IN reqs WHERE NOT x IN $ss] AS missing
        """
        rec = sess.run(qmiss, jobKey=jobKey, ss=ss).single()
        missing = rec["missing"] if rec else []
        if not missing:
            return []
        qcourse = """
        MATCH (c:Course)-[:HAS_KP]->(k:KP)
        OPTIONAL MATCH (k)-[:ASSOCIATED_WITH]->(s:Skill)
        WITH c, collect(DISTINCT toLower(k.kp_name)) AS kps, collect(DISTINCT toLower(s.name)) AS skills
        WITH c, apoc.coll.toSet(kps + skills) AS covers
        WITH c, size([x IN covers WHERE x IN $miss]) AS cover_cnt, covers
        WHERE cover_cnt > 0
        RETURN c.name AS course, cover_cnt AS cover_cnt, covers AS covers
        ORDER BY cover_cnt DESC
        LIMIT $k
        """
        rows = [r.data() for r in sess.run(qcourse, miss=missing, k=topk)]
    drv.close()
    return rows

def plan_learning_path(student_skills: List[str], jobKey: str, max_courses: int = 6,
                       uri: str = "bolt://localhost:7687", database: str = "neo4j",
                       no_auth: bool = True, user: str = "neo4j", password: str = "neo4j") -> Dict[str, Any]:
    miss_rows = recommend_courses_for_gap(student_skills, jobKey, topk=100, uri=uri, database=database,
                                          no_auth=no_auth, user=user, password=password)
    drv = _connect(uri, no_auth, user, password)
    with drv.session(database=database) as sess:
        qmiss = """
        MATCH (j:Job {jobKey:$jobKey})-[:REQUIRES_SKILL]->(s:Skill)
        WITH collect(DISTINCT toLower(s.name)) AS reqs
        RETURN [x IN reqs WHERE NOT x IN $ss] AS missing
        """
        rec = sess.run(qmiss, jobKey=jobKey, ss=[s.lower() for s in student_skills]).single()
        missing = set(rec["missing"] if rec else [])
    plan = []
    chosen = set()
    for _ in range(max_courses):
        best = None; gain = 0
        for r in miss_rows:
            if r["course"] in chosen: continue
            cov = set([x for x in (r.get("covers") or []) if x in missing])
            if len(cov) > gain:
                gain = len(cov); best = r
        if not best or gain == 0: break
        plan.append({"course": best["course"], "covers": sorted(set(best.get("covers") or []) & missing)})
        chosen.add(best["course"])
        missing -= set(best.get("covers") or [])
        if not missing: break
    return {"missing": sorted(missing), "path": plan}


# —— 新增：岗位文本 → 课程（基于知识点图谱） —— #
def recommend_courses_from_jobtext(job_title: str, job_desc: str, topn_kp: int = 60, topn_course: int = 15,
                                   uri: str = "bolt://localhost:7687", database: str = "neo4j",
                                   no_auth: bool = True, user: str = "neo4j", password: str = "neo4j") -> Dict[str, Any]:
    def _to_text(x):
        if x is None:
            return ""
        if isinstance(x, (list, tuple, set)):
            return " ".join(_to_text(i) for i in x)
        if isinstance(x, dict):
            return " ".join(_to_text(v) for v in x.values())
        return str(x)

    jt = _to_text(job_title).strip()
    jd = _to_text(job_desc).strip()
    if not (jt or jd):
        return {"courses": [], "kps": []}
    drv = _connect(uri, no_auth, user, password)
    jt = (job_title or "").strip(); jd = (job_desc or "").strip()
    jw = _tokenize(jt + "\n" + jd)
    if not jw:
        return {"courses": [], "tops": []}
    kp_rows = []
    with drv.session(database=database) as sess:
        res = sess.run("MATCH (k:KP) RETURN k.kid AS kid, k.kp_name AS name, k.kp_desc AS desc, k.course_name AS course")
        kp_rows = [r.data() for r in res]
    def score(row):
        name = row.get("name") or ""
        desc = row.get("desc") or ""
        nset = set(_tokenize(name)); dset = set(_tokenize(desc)); jset = set(jw)
        jn = len(nset & jset) / max(1, len(nset | jset))
        jd2 = len(dset & jset) / max(1, len(dset | jset))
        tf = sum((name.lower()).count(tok) for tok in jset) * 0.05
        return 0.7*jn + 0.3*jd2 + tf
    scored = [{**r, "score": score(r)} for r in kp_rows]
    scored = sorted([r for r in scored if r["score"] > 0], key=lambda x: x["score"], reverse=True)[:topn_kp]
    kp_scores = {r["kid"]: float(r["score"]) for r in scored}
    with drv.session(database=database) as sess:
        rel = sess.run("""
        MATCH (k:KP)-[r:PRE_REQ|ASSOCIATED_WITH|POST_REQ]-(n:KP)
        WHERE k.kid IN $kids
        RETURN k.kid AS src, type(r) AS t, n.kid AS nb
        """, kids=list(kp_scores.keys()))
        nbmap = {}
        for row in rel:
            nbmap.setdefault(row["src"], set()).add(row["nb"])
        for src, nbs in nbmap.items():
            share = kp_scores.get(src,0.0) * 0.15 / max(1, len(nbs))
            for nb in nbs:
                kp_scores[nb] = kp_scores.get(nb, 0.0) + share
        agg = sess.run("""
        UNWIND $pairs AS p
        MATCH (k:KP {kid:p.kid})<-[:HAS_KP]-(c:Course)
        RETURN c.name AS course, collect({kid:k.kid,score:p.score}) AS kps
        """, pairs=[{"kid":k, "score":s} for k,s in kp_scores.items()])
        rows = []
        for r in agg:
            ks = r["kps"]; sc = sum(x["score"] for x in ks) + 0.05*len(ks)
            rows.append({"course": r["course"], "score": sc, "kp_count": len(ks), "kps": ks})
        courses = sorted(rows, key=lambda x: x["score"], reverse=True)[:topn_course]
    drv.close()
    return {"courses": courses, "tops": scored}


