# -*- coding: utf-8 -*-
from typing import Any, Dict, List, Callable, Optional
import os, json, glob
from neo4j import GraphDatabase, basic_auth

LogFn = Callable[[str], None]

# 关系统一用英文名，避免工具/语法兼容性问题；中文关系名写入 label_cn 属性
REL_PRE   = "PRE_REQ"          # 前置（单向）
REL_POST  = "POST_REQ"         # 后置（单向）
REL_ASSOC = "ASSOCIATED_WITH"  # 关联（双向）

def _coerce_list(v: Any) -> List[str]:
    if v is None: return []
    if isinstance(v, list): return [str(x).strip() for x in v if str(x).strip()]
    s = str(v).strip().replace("，", ",").replace("；", ";")
    if not s: return []
    out = []
    for seg in s.split(";"):
        seg = seg.strip()
        if not seg: continue
        out.extend([t.strip() for t in seg.split(",") if t.strip()])
    return out

def _connect(uri: str, no_auth: bool, user: str, password: str):
    return GraphDatabase.driver(uri, auth=None if no_auth else basic_auth(user, password))

def _ensure_constraints(tx):
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (k:KP) REQUIRE k.kid IS UNIQUE")
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Course) REQUIRE c.name IS UNIQUE")

def _merge_course(tx, name: str):
    tx.run("MERGE (c:Course {name:$n}) SET c.`课程名称`=$n", n=name)

def _merge_kp(tx, r: Dict[str, Any]):
    course = r.get("课程名称") or r.get("课程")
    kid    = r.get("知识点ID") or r.get("id") or r.get("kp_id")
    kpname = r.get("知识点名称") or r.get("name") or r.get("kp_name")
    kpdesc = r.get("知识点说明") or r.get("desc") or r.get("kp_desc") or ""
    props = {
        "课程名称": course, "知识点ID": kid, "知识点名称": kpname, "知识点说明": kpdesc,
        "子知识点": r.get("子知识点", ""), "前置知识点": r.get("前置知识点",""),
        "后置知识点": r.get("后置知识点",""), "关联知识点": r.get("关联知识点",""),
        # 冗余英文字段，便于检索/算法
        "course_name": course, "kid": kid, "kp_name": kpname, "kp_desc": kpdesc
    }
    tx.run("MERGE (k:KP {kid:$kid}) SET k += $props", kid=kid, props=props)
    tx.run("MATCH (c:Course {name:$c}),(k:KP {kid:$kid}) MERGE (c)-[:HAS_KP]->(k)", c=course, kid=kid)

def _link(tx, a: str, b: str, t: str, cn: str):
    tx.run(f"MATCH (a:KP{{kid:$a}}),(b:KP{{kid:$b}}) MERGE (a)-[:{t} {{label_cn:$cn}}]->(b)", a=a, b=b, cn=cn)

def _read_json_records(fp: str) -> List[Dict[str, Any]]:
    with open(fp, "r", encoding="utf-8") as f:
        data = json.load(f)
    recs: List[Dict[str, Any]] = []
    # 兼容三种常见结构
    if isinstance(data, list) and data and isinstance(data[0], dict) and "课程名称" in data[0]:
        recs = data
    elif isinstance(data, list) and data and isinstance(data[0], dict) and "课程" in data[0] and "知识点" in data[0]:
        for blk in data:
            course = blk.get("课程")
            for kp in blk.get("知识点", []):
                x = dict(kp); x["课程名称"] = course; recs.append(x)
    elif isinstance(data, dict) and "知识点" in data:
        course = data.get("课程") or data.get("课程名称")
        for kp in data["知识点"]:
            x = dict(kp); x["课程名称"] = course; recs.append(x)
    else:
        raise ValueError(f"不支持的 JSON 结构：{fp}")
    return recs

def ingest_course_json(dir_path: str, bolt_uri: str, database: str,
                       no_auth: bool, user: str, password: str,
                       on_log: Optional[LogFn] = None) -> Dict[str, int]:
    on_log = on_log or (lambda s: None)
    files = sorted(glob.glob(os.path.join(dir_path, "*.json")))
    if not files:
        raise FileNotFoundError(f"未在 {dir_path} 找到 .json")
    drv = _connect(bolt_uri, no_auth, user, password)
    created = 0; linked = 0
    with drv.session(database=database) as sess:
        sess.execute_write(_ensure_constraints)
        # 先写节点
        for fp in files:
            on_log(f"[节点] 读取 {os.path.basename(fp)}")
            recs = _read_json_records(fp)
            for course in sorted({r.get("课程名称") or r.get("课程") for r in recs if (r.get("课程名称") or r.get("课程"))}):
                sess.execute_write(_merge_course, course)
            for r in recs:
                if not (r.get("课程名称") and r.get("知识点ID") and r.get("知识点名称")):
                    on_log(f"跳过缺字段记录：{r}")
                    continue
                sess.execute_write(_merge_kp, r); created += 1
        # 再写关系
        for fp in files:
            recs = _read_json_records(fp)
            for r in recs:
                src = (r.get("知识点ID") or "").strip()
                if not src: continue
                for pre in _coerce_list(r.get("前置知识点")):
                    sess.execute_write(_link, src, pre, REL_PRE,  "前置知识点"); linked += 1
                for post in _coerce_list(r.get("后置知识点")):
                    sess.execute_write(_link, src, post, REL_POST, "后置知识点"); linked += 1
                for rel in _coerce_list(r.get("关联知识点")):
                    sess.execute_write(_link, src, rel, REL_ASSOC, "关联知识点")
                    sess.execute_write(_link, rel, src, REL_ASSOC, "关联知识点"); linked += 2
    drv.close()
    on_log(f"[完成] KP={created}, REL={linked}")
    return {"kp": created, "rel": linked}

# —— 以下两个函数保留原名，避免破坏你已有调用 —— #
def ingest_courses(csv_path: str, bolt_uri: str, database: str,
                   no_auth: bool, user: str, password: str,
                   on_log: Optional[LogFn] = None):
    on_log = (on_log or (lambda s: None))
    on_log(f"[兼容占位] ingest_courses 加载 {csv_path} （此实现不改动数据库）")

def link_jobs_to_skills(job_csv: str, bolt_uri: str, database: str,
                        no_auth: bool, user: str, password: str,
                        on_log: Optional[LogFn] = None):
    on_log = (on_log or (lambda s: None))
    on_log(f"[兼容占位] link_jobs_to_skills 读取 {job_csv} （此实现不改动数据库）")
