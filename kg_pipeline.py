
# -*- coding: utf-8 -*-
import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple, Iterable, Optional, Callable

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tenacity import retry, stop_after_attempt, wait_random_exponential
from neo4j import GraphDatabase

from kg_enrich_hooks import enrich_job_record, upsert_job_with_edu


from config import BOLT_URI, DATABASE, USER, PASSWORD, NO_AUTH, EXPORT_DIR, SLEEP_MIN, SLEEP_MAX

NOWPICK_URL = "https://nowpick.nowcoder.com/u/job/main-search-job"

def sha1(s: str) -> str:
    import hashlib as _h
    return _h.sha1(s.encode("utf-8")).hexdigest()

def json_dumps_cn(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)

def build_session() -> requests.Session:
    s = requests.Session()
    retry_cfg = Retry(
        total=5,
        backoff_factor=0.4,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS", "POST"],
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retry_cfg))
    s.headers.update({"user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122 Safari/537.36"})
    return s

@retry(stop=stop_after_attempt(3), wait=wait_random_exponential(multiplier=0.5, max=4))
def fetch_page(session: requests.Session, page: int, page_size: int, query: str) -> List[Dict[str, Any]]:
    params = {"_": int(time.time() * 1000)}
    form = {"page": page, "pageSize": page_size, "query": query, "requestFrom": 3}
    resp = session.post(NOWPICK_URL, params=params, data=form, timeout=15)
    resp.raise_for_status()
    j = resp.json()
    data = j.get("data", {}) or {}
    arr = data.get("datas", []) or []
    items = []
    for d in arr:
        item = d.get("data", {}) or {}
        if item:
            items.append(item)
    return items

def _try_parse_json(text: str) -> Optional[Dict[str, Any]]:
    if not isinstance(text, str):
        return None
    t = text.strip()
    if not t or (not t.startswith("{") and not t.startswith("[")):
        return None
    try:
        val = json.loads(t)
        if isinstance(val, dict):
            return val
        return None
    except Exception:
        return None

DESC_KEYS = [
    "jobDesc","jobDescription","description","content","jobContent","jd","jdContent","detail","details",
    "岗位描述","岗位职责","工作内容","职位描述","职责","介绍"
]

def extract_title_and_desc(item: Dict[str, Any]) -> Tuple[str, str]:
    title = str(item.get("jobName") or item.get("jobTitle") or "").strip()
    for k in ["jobDescription","jobDesc","description"]:
        v = item.get(k)
        if isinstance(v, str) and v.strip():
            return title or "", v.strip()
    ext = item.get("ext")
    if isinstance(ext, str) and ext.strip():
        parsed = _try_parse_json(ext)
        if isinstance(parsed, dict):
            for k in DESC_KEYS:
                if k in parsed and isinstance(parsed[k], str) and parsed[k].strip():
                    return title or "", parsed[k].strip()
            for k, v in parsed.items():
                if isinstance(v, list) and v and all(isinstance(x, str) for x in v):
                    joined = "\n".join([x for x in v if x.strip()])
                    if joined.strip():
                        return title or "", joined.strip()
        else:
            return title or "", ext.strip()
    return title or "", ""

def normalize_item(item: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    jobCityList = item.get("jobCityList") or []
    jobCityList = [str(c) for c in jobCityList] if isinstance(jobCityList, (list, tuple)) else [str(jobCityList)]
    title, desc = extract_title_and_desc(item)
    complete_item = {
        "companyId": item.get("companyId", ""),
        "bossUid": item.get("bossUid", ""),
        "jobName": item.get("jobName", ""),
        "ext": item.get("ext", ""),
        "jobCityList": jobCityList,
        "graduationYear": item.get("graduationYear", ""),
        "eduLevel": item.get("eduLevel", ""),
        "salaryType": item.get("salaryType", ""),
        "salaryMin": item.get("salaryMin", ""),
        "salaryMax": item.get("salaryMax", ""),
        "salaryMonth": item.get("salaryMonth", ""),
        "jobTitle": title,
        "jobDescription": desc,
    }
    company_info = item.get("recommendInternCompany", {}) or {}
    complete_company = {
        "companyName": company_info.get("companyName", ""),
        "scaleTagName": company_info.get("scaleTagName", ""),
        "personScales": company_info.get("personScales", ""),
        "companyShortName": company_info.get("companyShortName", ""),
        "address": company_info.get("address", ""),
    }
    return complete_item, complete_company

def is_complete(complete_item: Dict[str, Any], complete_company: Dict[str, Any]) -> bool:
    base_keys = ['companyId','bossUid','jobName','ext','jobCityList','graduationYear','eduLevel',
                 'salaryType','salaryMin','salaryMax','salaryMonth']
    if not all(k in complete_item for k in base_keys):
        return False
    ok1 = all(complete_item[k] != "" and complete_item[k] is not None for k in base_keys if k != 'jobCityList')
    ok2 = complete_item.get('jobCityList') is not None
    ok3 = all(v != "" and v is not None for v in complete_company.values())
    return ok1 and ok2 and ok3

def build_job_key(ci: Dict[str, Any]) -> str:
    base = "|".join([
        str(ci.get("companyId", "")),
        str(ci.get("bossUid", "")),
        str(ci.get("jobName", "")),
        str(ci.get("graduationYear", "")),
        str(ci.get("eduLevel", "")),
        str(ci.get("salaryType", "")),
        str(ci.get("salaryMin", "")),
        str(ci.get("salaryMax", "")),
        str(ci.get("salaryMonth", "")),
        ",".join(sorted([str(c) for c in (ci.get("jobCityList") or [])]))
    ])
    return sha1(base)

def dedup(seq: Iterable[Tuple[Dict[str, Any], Dict[str, Any]]]) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    seen = set()
    out = []
    for ci, cc in seq:
        key = (ci["companyId"], ci["bossUid"], ci["jobName"], json_dumps_cn(sorted(ci["jobCityList"])))
        if key in seen:
            continue
        seen.add(key)
        out.append((ci, cc))
    return out

@dataclass
class GraphTables:
    companies: List[Dict[str, Any]]
    jobs: List[Dict[str, Any]]
    cities: List[Dict[str, Any]]
    edus: List[Dict[str, Any]]
    rel_posted: List[Dict[str, Any]]
    rel_located_in: List[Dict[str, Any]]
    rel_requires_edu: List[Dict[str, Any]]

def edulevel_label(v: Any) -> str:
    return str(v)

def shape_graph(records: List[Tuple[Dict[str, Any], Dict[str, Any]]]) -> GraphTables:
    companies: Dict[str, Dict[str, Any]] = {}
    jobs: Dict[str, Dict[str, Any]] = {}
    cities: Dict[str, Dict[str, Any]] = {}
    edus: Dict[str, Dict[str, Any]] = {}
    rel_posted, rel_located_in, rel_requires_edu = [], [], []
    for ci, cc in records:
        companyId = str(ci["companyId"])
        if companyId not in companies:
            companies[companyId] = {
                "companyId": companyId,
                "companyName": str(cc["companyName"]),
                "scaleTagName": str(cc["scaleTagName"]),
                "personScales": str(cc["personScales"]),
                "companyShortName": str(cc["companyShortName"]),
                "address": str(cc["address"]),
            }
        jobKey = build_job_key(ci)
        if jobKey not in jobs:
            title = (ci.get("jobTitle") or ci.get("jobName") or "").strip()
            desc = (ci.get("jobDescription") or "").strip()
            jobs[jobKey] = {
                "jobKey": jobKey,
                "companyId": companyId,
                "bossUid": str(ci["bossUid"]),
                "jobName": str(ci["jobName"]),
                "title": title,
                "description": desc,
                "ext": str(ci["ext"]),
                "graduationYear": str(ci["graduationYear"]),
                "eduLevel": str(ci["eduLevel"]),
                "eduLevelLabel": edulevel_label(ci["eduLevel"]),
                "salaryType": str(ci["salaryType"]),
                "salaryMin": str(ci["salaryMin"]),
                "salaryMax": str(ci["salaryMax"]),
                "salaryMonth": str(ci["salaryMonth"]),
            }
            rel_posted.append({"companyId": companyId, "jobKey": jobKey})
        for city in (ci.get("jobCityList") or []):
            cityName = str(city)
            if cityName not in cities:
                cities[cityName] = {"name": cityName}
            rel_located_in.append({"jobKey": jobKey, "cityName": cityName})
        edu_name = edulevel_label(ci["eduLevel"])
        if edu_name not in edus:
            edus[edu_name] = {"name": edu_name}
        rel_requires_edu.append({"jobKey": jobKey, "eduName": edu_name})
    return GraphTables(
        companies=list(companies.values()),
        jobs=list(jobs.values()),
        cities=list(cities.values()),
        edus=list(edus.values()),
        rel_posted=rel_posted,
        rel_located_in=rel_located_in,
        rel_requires_edu=rel_requires_edu,
    )

def chunked(seq: List[Dict[str, Any]], size: int):
    for i in range(0, len(seq), size):
        yield seq[i:i+size]

CONSTRAINTS = [
    "CREATE CONSTRAINT company_id IF NOT EXISTS FOR (c:Company) REQUIRE c.companyId IS UNIQUE",
    "CREATE CONSTRAINT job_key IF NOT EXISTS FOR (j:Job) REQUIRE j.jobKey IS UNIQUE",
    "CREATE CONSTRAINT city_name IF NOT EXISTS FOR (ci:City) REQUIRE ci.name IS UNIQUE",
    "CREATE CONSTRAINT edu_name IF NOT EXISTS FOR (e:EducationLevel) REQUIRE e.name IS UNIQUE",
]

Q_MERGE_COMPANY = """
UNWIND $rows AS row
MERGE (c:Company {companyId: row.companyId})
SET c.companyName = row.companyName,
    c.scaleTagName = row.scaleTagName,
    c.personScales = row.personScales,
    c.companyShortName = row.companyShortName,
    c.address = row.address
"""

Q_MERGE_JOB = """
UNWIND $rows AS row
MERGE (j:Job {jobKey: row.jobKey})
SET j.companyId = row.companyId,
    j.bossUid = row.bossUid,
    j.jobName = row.jobName,
    j.title = row.title,
    j.description = row.description,
    j.ext = row.ext,
    j.graduationYear = row.graduationYear,
    j.eduLevel = row.eduLevel,
    j.eduLevelLabel = row.eduLevelLabel,
    j.salaryType = row.salaryType,
    j.salaryMin = toFloat(row.salaryMin),
    j.salaryMax = toFloat(row.salaryMax),
    j.salaryMonth = toInteger(row.salaryMonth)
"""

Q_MERGE_CITY = """
UNWIND $rows AS row
MERGE (ci:City {name: row.name})
"""

Q_MERGE_EDU = """
UNWIND $rows AS row
MERGE (e:EducationLevel {name: row.name})
"""

Q_REL_POSTED = """
UNWIND $rows AS row
MATCH (c:Company {companyId: row.companyId})
MATCH (j:Job {jobKey: row.jobKey})
MERGE (c)-[:POSTED]->(j)
"""

Q_REL_LOCATED_IN = """
UNWIND $rows AS row
MATCH (j:Job {jobKey: row.jobKey})
MATCH (ci:City {name: row.cityName})
MERGE (j)-[:LOCATED_IN]->(ci)
"""

Q_REL_REQUIRES_EDU = """
UNWIND $rows AS row
MATCH (j:Job {jobKey: row.jobKey})
MATCH (e:EducationLevel {name: row.eduName})
MERGE (j)-[:REQUIRES_EDU]->(e)
"""

def write_graph(uri: str, auth: Optional[tuple], database: str, g: GraphTables, batch: int = 1000, on_log: Optional[Callable[[str], None]] = None):
    log = on_log or (lambda s: None)
    driver = GraphDatabase.driver(uri, auth=auth)
    log("连接 Neo4j 成功，开始建约束/写入节点关系...")
    with driver.session(database=database) as session:
        for c in CONSTRAINTS:
            session.run(c)
        log("约束检查完成。")
        for rows in chunked(g.companies, batch):
            session.run(Q_MERGE_COMPANY, rows=rows)
        log(f"Company 节点：{len(g.companies)} 条")
        for rows in chunked(g.jobs, batch):
            session.run(Q_MERGE_JOB, rows=rows)
        log(f"Job 节点：{len(g.jobs)} 条")
        for rows in chunked(g.cities, batch):
            session.run(Q_MERGE_CITY, rows=rows)
        log(f"City 节点：{len(g.cities)} 条")
        for rows in chunked(g.edus, batch):
            session.run(Q_MERGE_EDU, rows=rows)
        log(f"EducationLevel 节点：{len(g.edus)} 条")
        for rows in chunked(g.rel_posted, batch):
            session.run(Q_REL_POSTED, rows=rows)
        log(f"POSTED 关系：{len(g.rel_posted)} 条")
        for rows in chunked(g.rel_located_in, batch):
            session.run(Q_REL_LOCATED_IN, rows=rows)
        log(f"LOCATED_IN 关系：{len(g.rel_located_in)} 条")
        for rows in chunked(g.rel_requires_edu, batch):
            session.run(Q_REL_REQUIRES_EDU, rows=rows)
        log(f"REQUIRES_EDU 关系：{len(g.rel_requires_edu)} 条")
    driver.close()
    log("写入完成。")

def save_flat_csv(records: List[Tuple[Dict[str, Any], Dict[str, Any]]], query: str, out_dir: str) -> str:
    title = ['companyId', 'bossUid', 'jobName', 'ext', 'jobCityList', 'graduationYear', 'eduLevel',
             'salaryType', 'salaryMin', 'salaryMax', 'salaryMonth']
    companyContent = ['companyName', 'scaleTagName', 'personScales', 'companyShortName', 'address']
    out_rows = []
    for ci, cc in records:
        row = {k: ci.get(k) for k in title}
        row["jobCityList"] = json_dumps_cn(ci.get("jobCityList", []))
        row.update({k: cc.get(k) for k in companyContent})
        out_rows.append(row)
    df = pd.DataFrame(out_rows, columns=title + companyContent)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = str(Path(out_dir) / f"{query}_offer.csv")
    df.to_csv(path, index=False)
    return path

def run_pipeline(query: str, pages: int, page_size: int, bolt_uri: str = BOLT_URI, database: str = DATABASE,
                 no_auth: bool = NO_AUTH, user: str = USER, password: str = PASSWORD,
                 sleep_min: float = SLEEP_MIN, sleep_max: float = SLEEP_MAX,
                 on_log: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    log = on_log or print
    log(f"开始抓取：query='{query}', pages={pages}, page_size={page_size}")
    session = build_session()
    records: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for page in range(1, pages + 1):
        time.sleep(random.uniform(sleep_min, sleep_max))
        try:
            items = fetch_page(session, page, page_size, query)
            log(f"第 {page}/{pages} 页返回 {len(items)} 条")
        except Exception as e:
            log(f"[WARN] 第 {page} 页抓取失败：{e}")
            continue
        cnt = 0
        for item in items[:page_size]:
            ci, cc = normalize_item(item)
            if is_complete(ci, cc):
                records.append((ci, cc)); cnt += 1
        log(f"第 {page} 页可用 {cnt} 条，累计 {len(records)} 条")
    records = dedup(records)
    log(f"去重后记录数：{len(records)}")

    # === 新增：单条增强 + 写库（Job + Education + REQUIRES_EDU） ===
    auth = None if no_auth else (user, password)
    driver = GraphDatabase.driver(bolt_uri, auth=auth)
    with driver.session(database=database) as session:
        for ci, cc in records:
            rec = {
                "jobKey": build_job_key(ci),
                "companyId": str(ci.get("companyId", "")),
                "jobName": str(ci.get("jobName", "")),
                "title": (ci.get("jobTitle") or ci.get("jobName") or "").strip(),
                "description": (ci.get("jobDescription") or "").strip(),
                "eduLevel": str(ci.get("eduLevel") or ""),
                "salaryMin": ci.get("salaryMin"),
                "salaryMax": ci.get("salaryMax"),
                "salaryMonth": ci.get("salaryMonth"),
                "ext": ci.get("ext", ""),
            }
            # 1) 写库前做增强（JD清洗 + 学历推断等）
            rec = enrich_job_record(rec)
            # 2) 写库（含建立 REQUIRES_EDU）
            session.execute_write(upsert_job_with_edu, rec)

    # 继续原流程
    csv_path = save_flat_csv(records, query, EXPORT_DIR)
    log(f"CSV 已保存：{csv_path}")
    g = shape_graph(records)
    log(f"节点 Company:{len(g.companies)} Job:{len(g.jobs)} City:{len(g.cities)} Edu:{len(g.edus)}")
    log(f"关系 POSTED:{len(g.rel_posted)} LOCATED_IN:{len(g.rel_located_in)} REQUIRES_EDU:{len(g.rel_requires_edu)}")
    auth = None if no_auth else (user, password)
    write_graph(uri=bolt_uri, auth=auth, database=database, g=g, batch=1000, on_log=log)
    log("全部流程完成。")
    return {
        "csv_path": csv_path,
        "counts": {
            "companies": len(g.companies),
            "jobs": len(g.jobs),
            "cities": len(g.cities),
            "edus": len(g.edus),
        }
    }

def main():
    ap = argparse.ArgumentParser(description="Nowcoder -> CSV(同格式) + Neo4j import（含title/description，无认证可选）")
    ap.add_argument("--query", type=str, default="python")
    ap.add_argument("--pages", type=int, default=10)
    ap.add_argument("--page-size", type=int, default=20)
    ap.add_argument("--bolt-uri", type=str, default=BOLT_URI)
    ap.add_argument("--user", type=str, default=USER)
    ap.add_argument("--password", type=str, default=PASSWORD)
    ap.add_argument("--database", type=str, default=DATABASE)
    ap.add_argument("--no-auth", action="store_true" if not NO_AUTH else "store_false")
    args = ap.parse_args()
    no_auth = args.no_auth if args.no_auth is not None else NO_AUTH
    def _println(msg: str):
        print(msg, flush=True)
    res = run_pipeline(
        query=args.query, pages=args.pages, page_size=args.page_size,
        bolt_uri=args.bolt_uri, database=args.database,
        no_auth=no_auth, user=args.user, password=args.password,
        on_log=_println
    )
    print("[OK] CSV:", res["csv_path"])
    print("[OK] counts:", res["counts"])

if __name__ == "__main__":
    main()
