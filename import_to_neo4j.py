# -*- coding: utf-8 -*-
"""
kg_importer.py
----------------
读取 ./course 目录下的课程知识点 JSON（示例：课程知识点图谱_ALL.json），
将“课程—知识点”导入 Neo4j 社区版，并建立三类关系：
- 前置知识点（单向）：:PRE_REQ
- 后置知识点（单向）：:POST_REQ
- 关联知识点（双向）：:ASSOCIATED_WITH

节点属性：
- 对每条 JSON 记录均创建 :KP 节点，包含以下属性（与你给的列同名）：
  课程名称、知识点ID、知识点名称、知识点说明、子知识点、前置知识点、后置知识点、关联知识点
  同时补充英文别名便于索引：course_name, kid, kp_name, kp_desc
- 同时为每门课程创建 :Course 节点（唯一键 name=课程名称），并建立 (:Course)-[:HAS_KP]->(:KP)

用法：
  # 无认证（Neo4j 未启用 auth）
  python kg_importer.py --uri bolt://localhost:7687 --no-auth --course-dir ./course

  # 使用认证
  python kg_importer.py --uri bolt://localhost:7687 --user neo4j --password neo4j --course-dir ./course

  # 仅验证 JSON（不写库）
  python kg_importer.py --course-dir ./course --dry-run

依赖：pip install neo4j==5.*
"""
import os, json, glob, argparse
from typing import Dict, Any, List
from neo4j import GraphDatabase, basic_auth

REL_LABELS = {
    "前置知识点": "PRE_REQ",
    "后置知识点": "POST_REQ",
    "关联知识点": "ASSOCIATED_WITH"
}

def read_json_records(fp: str) -> List[Dict[str, Any]]:
    with open(fp, "r", encoding="utf-8") as f:
        data = json.load(f)
    recs: List[Dict[str, Any]] = []
    # 兼容三种结构
    if isinstance(data, list) and data and isinstance(data[0], dict) and "课程名称" in data[0]:
        recs = data
    elif isinstance(data, list) and data and isinstance(data[0], dict) and "课程" in data[0] and "知识点" in data[0]:
        for block in data:
            course = block.get("课程")
            for kp in block.get("知识点", []):
                kp = dict(kp); kp["课程名称"] = course; recs.append(kp)
    elif isinstance(data, dict) and "知识点" in data:
        course = data.get("课程") or data.get("课程名称")
        for kp in data["知识点"]:
            kp = dict(kp); kp["课程名称"] = course; recs.append(kp)
    else:
        raise ValueError("未识别的 JSON 结构，请确认顶层是否为列表或包含 '知识点' 字段。")
    return recs

def coerce_list(field: Any) -> List[str]:
    if field is None: return []
    if isinstance(field, list): return [str(x).strip() for x in field if str(x).strip()]
    s = str(field).strip()
    if not s: return []
    parts = [p.strip() for p in s.replace("，", ",").replace("；", ";").replace("|", ";").split(";")]
    out = []
    for p in parts:
        if not p: continue
        out.extend([q.strip() for q in p.split(",") if q.strip()])
    return out

def ensure_constraints(tx):
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (k:KP) REQUIRE k.kid IS UNIQUE")
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Course) REQUIRE c.name IS UNIQUE")

def merge_course(tx, course_name: str):
    tx.run("MERGE (c:Course {name:$name}) SET c.`课程名称` = $name", name=course_name)

def merge_kp(tx, rec: Dict[str, Any]):
    course = rec.get("课程名称") or rec.get("课程")
    kid    = rec.get("知识点ID") or rec.get("id") or rec.get("kp_id")
    kpname = rec.get("知识点名称") or rec.get("name") or rec.get("kp_name")
    kpdesc = rec.get("知识点说明") or rec.get("desc") or rec.get("kp_desc") or ""
    if not (course and kid and kpname):
        raise ValueError(f"记录缺少关键字段：课程名称/知识点ID/知识点名称: {rec}")

    props = {
        "课程名称": course, "知识点ID": kid, "知识点名称": kpname, "知识点说明": kpdesc,
        "子知识点": rec.get("子知识点", ""), "前置知识点": rec.get("前置知识点", ""),
        "后置知识点": rec.get("后置知识点", ""), "关联知识点": rec.get("关联知识点", ""),
        "course_name": course, "kid": kid, "kp_name": kpname, "kp_desc": kpdesc
    }
    tx.run("MERGE (k:KP {kid:$kid}) SET k += $props", kid=kid, props=props)
    tx.run("MATCH (c:Course {name:$cname}), (k:KP {kid:$kid}) MERGE (c)-[:HAS_KP]->(k)",
           cname=course, kid=kid)

def link_kp(tx, src_id: str, dst_id: str, rel_type: str, label_cn: str):
    if not (src_id and dst_id): return
    tx.run(
        f"MATCH (a:KP {{kid:$a}}), (b:KP {{kid:$b}}) "
        f"MERGE (a)-[:{rel_type} {{label_cn:$cn}}]->(b)",
        a=src_id, b=dst_id, cn=label_cn
    )

def dry_validate(course_dir: str):
    files = sorted(glob.glob(os.path.join(course_dir, "*.json")))
    if not files: raise FileNotFoundError(f"未在 {course_dir} 下找到 .json 文件")
    nodes = 0; pre=post=assoc=0
    for fp in files:
        recs = read_json_records(fp)
        nodes += len(recs)
        for r in recs:
            pre  += len(coerce_list(r.get("前置知识点")))
            post += len(coerce_list(r.get("后置知识点")))
            assoc+= len(coerce_list(r.get("关联知识点"))) * 2  # 双向各建一条
    print(f"[DRY] 预计节点(KP)={nodes}, 预创建关系: PRE_REQ={pre}, POST_REQ={post}, ASSOCIATED_WITH={assoc}")

def import_course_dir(uri: str, auth, course_dir: str):
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(uri, auth=auth, max_connection_lifetime=300)
    files = sorted(glob.glob(os.path.join(course_dir, "*.json")))
    if not files:
        raise FileNotFoundError(f"未在 {course_dir} 下找到 .json 文件")
    with driver.session() as sess:
        sess.execute_write(ensure_constraints)
        # 节点
        for fp in files:
            recs = read_json_records(fp)
            for name in sorted({r.get("课程名称") or r.get("课程") for r in recs if (r.get("课程名称") or r.get("课程"))}):
                sess.execute_write(merge_course, name)
            for r in recs:
                sess.execute_write(merge_kp, r)
        # 关系
        for fp in files:
            recs = read_json_records(fp)
            for r in recs:
                src = (r.get("知识点ID") or "").strip()
                if not src: continue
                for pre in coerce_list(r.get("前置知识点")):
                    sess.execute_write(link_kp, src, pre, REL_LABELS["前置知识点"], "前置知识点")
                for post in coerce_list(r.get("后置知识点")):
                    sess.execute_write(link_kp, src, post, REL_LABELS["后置知识点"], "后置知识点")
                for rel in coerce_list(r.get("关联知识点")):
                    sess.execute_write(link_kp, src, rel, REL_LABELS["关联知识点"], "关联知识点")
                    sess.execute_write(link_kp, rel, src, REL_LABELS["关联知识点"], "关联知识点")
    driver.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", default="bolt://localhost:7687")
    ap.add_argument("--user", default="neo4j")
    ap.add_argument("--password", default="neo4j")
    ap.add_argument("--no-auth", action="store_true", help="Neo4j 未开启认证时使用")
    ap.add_argument("--course-dir", default="./course")
    ap.add_argument("--dry-run", action="store_true", help="仅验证 JSON，不写入数据库")
    args = ap.parse_args()
    if args.dry_run:
        dry_validate(args.course_dir)
        return
    auth = None if args.no_auth else basic_auth(args.user, args.password)
    print(f"[INFO] Neo4j: {args.uri} 认证: {'无' if args.no_auth else args.user}")
    print(f"[INFO] 导入目录: {args.course_dir}")
    import_course_dir(args.uri, auth, args.course_dir)
    print("[OK] 导入完成")

if __name__ == "__main__":
    main()
