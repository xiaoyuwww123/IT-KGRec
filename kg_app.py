from typing import List, Dict, Any, Optional, Tuple
import re
import pandas as pd
import streamlit as st
from neo4j import GraphDatabase
from pyvis.network import Network

import kg_pipeline
import edu_ingest
import recommender
import llm_helper
from config import (
    BOLT_URI, DATABASE, USER, PASSWORD, NO_AUTH,
    COURSE_JSON_DIR, OPENAI_API_KEY, OPENAI_MODEL
)


def _to_text(x):
    if x is None:
        return ""
    if isinstance(x, (list, tuple, set)):
        return " ".join(_to_text(i) for i in x)
    if isinstance(x, dict):
        return " ".join(_to_text(v) for v in x.values())
    return str(x)


# ==================== 全局样式（修复顶部遮挡、统一字体与边距） ====================
st.set_page_config(page_title="教育知识图谱系统", layout="wide")
st.markdown('''
<style>
    header[data-testid="stHeader"] {backdrop-filter: blur(6px);}
    .block-container {padding-top: 3.5rem; padding-bottom: 2rem;}
    h1, h2, h3 { font-weight: 700; }
    .stButton>button { border-radius: 8px; }
    .stTextInput>div>div>input, .stSelectbox>div>div, .stNumberInput input { border-radius: 6px; }
    .stDataFrame { border: 1px solid #eee; border-radius: 10px; }
    .cute {padding:14px 16px; background:#fff7fb; border:1px dashed #ffa7d1; border-radius:10px;}
</style>
''', unsafe_allow_html=True)

# ==================== 连接区 ====================
st.sidebar.header("Neo4j 连接")
bolt_uri = st.sidebar.text_input("Bolt URI", value=BOLT_URI, key="bolt_uri")
no_auth  = st.sidebar.checkbox("无身份验证", value=NO_AUTH, key="noauth")
user     = st.sidebar.text_input("用户", value=USER, disabled=no_auth, key="user")
password = st.sidebar.text_input("密码", value=PASSWORD, type="password", disabled=no_auth, key="password")
database = st.sidebar.text_input("数据库", value=DATABASE, key="database")

st.sidebar.markdown("---")
st.sidebar.header("OpenAI 配置")
api_key = st.sidebar.text_input("OpenAI API Key", value=OPENAI_API_KEY or "", type="password", key="openai_key")
model   = st.sidebar.text_input("模型", value=OPENAI_MODEL or "gpt-4o", key="openai_model")
if st.sidebar.button("测试 OpenAI 连通性", key="test_openai"):
    try:
        import time
        t0 = time.time()
        for _ in llm_helper.stream_plan(api_key, model, "请只回复：OK", history=[{"role":"system","content":"用一个词回答"}]):
            break
        st.sidebar.success(f"OpenAI 可用，首字节延迟 ~{time.time()-t0:.2f}s")
    except Exception as e:
        st.sidebar.error(f"OpenAI 不可用：{e}")

if st.sidebar.button("测试 Neo4j 连接", key="test_conn"):
    try:
        with GraphDatabase.driver(bolt_uri, auth=None if no_auth else (user, password)).session(database=database) as sess:
            ok = sess.run("RETURN 1 AS ok").single().get("ok")
        st.sidebar.success(f"连接成功：{ok}")
    except Exception as e:
        st.sidebar.error(f"连接失败：{e}")

@st.cache_resource(show_spinner=False)
def get_driver(uri: str, user: str, password: str, no_auth: bool):
    if no_auth: return GraphDatabase.driver(uri, auth=None)
    return GraphDatabase.driver(uri, auth=(user, password))

def _execute_read(sess, func):
    try: return sess.execute_read(func)
    except AttributeError: return sess.read_transaction(func)

def run_query(tx, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    result = tx.run(query, params or {})
    return [r.data() for r in result]

# ==================== 薪资解析与标准化 ====================
SALARY_RE_LIST = [
    re.compile(r'(?P<min>\d+(?:\.\d+)?)\s*[-~]\s*(?P<max>\d+(?:\.\d+)?)\s*[kK]'),
    re.compile(r'(?P<min>\d+(?:\.\d+)?)\s*[-~]\s*(?P<max>\d+(?:\.\d+)?)\s*万\s*/\s*(?P<per>[年月天])'),
    re.compile(r'(?P<val>\d+(?:\.\d+)?)\s*[kK](?:\s*·\s*(?P<month>\d+)\s*薪)?'),
    re.compile(r'(?P<val>\d+(?:\.\d+)?)\s*万\s*/\s*(?P<per>[年月天])'),
]
def parse_salary_text(text: str):
    if not text: return None
    t = text.replace("／","/").replace("・","·").lower()
    for pat in SALARY_RE_LIST:
        m = pat.search(t)
        if not m: continue
        gd = m.groupdict()
        if gd.get("min"):
            mi, ma = float(gd["min"]), float(gd["max"])
            per = gd.get("per")
            if per == "年": mi, ma, mm = mi*10000/12, ma*10000/12, 12
            elif per == "月": mi, ma, mm = mi*10000, ma*10000, 12
            elif per == "天": mi, ma, mm = mi*22, ma*22, 12
            else: mi, ma, mm = mi*1000, ma*1000, 12
            return (mi, ma, mm)
        if gd.get("val"):
            v = float(gd["val"])
            per = gd.get("per"); month = gd.get("month")
            if per == "年": monthly = v*10000/12
            elif per == "月": monthly = v*10000
            elif per == "天": monthly = v*22
            else: monthly = v*1000
            return (monthly, monthly, int(month or 12))
    return None

def normalize_salary(smin, smax, smonth, stext):
    parsed = parse_salary_text(stext or "")
    if parsed:
        mi, ma, mm = parsed
    else:
        mi = float(smin or 0); ma = float(smax or 0); mm = int(smonth or 12)
        def fix(v: float) -> float:
            if v >= 1000: return v
            if 50 <= v <= 1000: return v*100.0
            if v == 0: return 0.0
            return v*1000.0
        mi, ma = fix(mi), fix(ma)
        if ma == 9_999_999: ma = mi
        if mm in (0, None): mm = 12
    kfmt = lambda x: f"{x/1000:.0f}K" if x >= 1000 else f"{x:.0f}"
    return {"min_month_cny": mi, "max_month_cny": ma, "months": mm,
            "pretty": f"{kfmt(mi)} - {kfmt(ma)}·{mm}薪",
            "annual_cny": (mi+ma)/2*mm}

# ==================== 学历推断（从职位描述中补齐） ====================
EDU_MAP = [
    ("博士", ["博士","phd","doctor"]),
    ("硕士", ["硕士","研究生","master","msc","m.s.","m.eng","mba"]),
    ("本科", ["本科","学士","bachelor","b.s.","beng","一本","二本","全日制本科"]),
    ("大专", ["大专","专科","高职","associate"]),
    ("高中/中专", ["高中","中专","中技"]),
    ("不限", ["不限","不限制","不限学历","学历不限"]),
]
def infer_edu_from_text(text: str) -> str:
    if not text: return ""
    t = text.lower()
    for name, keys in EDU_MAP:
        for k in keys:
            if k in t:
                return name
    return ""

# ==================== Tabs ====================
tab_data, tab_explore, tab_rec = st.tabs(["数据采集与入库", "知识图谱检索", "画像与推荐"])

# ==================== 数据采集与入库 ====================
with tab_data:
    st.subheader("招聘抓取 → 入库")
    c1, c2, c3, c4 = st.columns([1,1,1,1])
    q = c1.text_input("抓取关键词", value="python", key="crawl_query")
    pages = c2.number_input("页数", min_value=1, max_value=100, value=5, step=1, key="crawl_pages")
    page_size = c3.number_input("每页数量", min_value=5, max_value=50, value=20, step=5, key="crawl_pagesize")
    start = c4.button("开始抓取 → 入库", type="primary", key="run_pipeline")
    if start:
        with st.status("查询中... 正在抓取并入库", expanded=True) as status:
            log_box = st.empty(); logs = []
            def on_log(msg: str):
                logs.append(str(msg)); log_box.code("\n".join(logs[-200:]))
            try:
                res = kg_pipeline.run_pipeline(
                    query=q, pages=int(pages), page_size=int(page_size),
                    bolt_uri=bolt_uri, database=database,
                    no_auth=bool(no_auth), user=user, password=password,
                    on_log=on_log
                )
                status.update(label="入库完成 ✅", state="complete")
                st.success(f"入库统计：{res['counts']}")
            except Exception as e:
                status.update(label=f"入库失败：{e}", state="error"); st.error(f"入库失败：{e}")

    st.markdown("---")
    st.subheader("导入课程知识点 JSON")
    st.caption(f"将 JSON 放在 `{COURSE_JSON_DIR}`，字段：课程名称、知识点ID、知识点名称、知识点说明、子知识点、前置知识点、后置知识点、关联知识点。")
    if st.button("导入 ./course/*.json → Neo4j", key="ingest_course_json"):
        with st.status("查询中… 导入 JSON → Neo4j", expanded=True) as status:
            box = st.empty()
            def on_log(msg: str): box.code(str(msg))
            try:
                stats = edu_ingest.ingest_course_json(dir_path=COURSE_JSON_DIR,
                                                      bolt_uri=bolt_uri, database=database,
                                                      no_auth=bool(no_auth), user=user, password=password,
                                                      on_log=on_log)
                status.update(label="JSON 导入完成 ✅", state="complete")
                st.success(f"KP 节点：{stats['kp']}，关系：{stats['rel']}（前置/后置为单向，关联为双向）")
            except Exception as e:
                status.update(label=f"导入失败：{e}", state="error"); st.error(f"导入失败：{e}")
    st.markdown("---")
    with st.expander("一键清空图谱", expanded=False):
        colc1, colc2 = st.columns([3, 1])
        confirm = colc1.checkbox("该操作无法撤销，是否确认清空 Company/Job/City/EducationLevel/KP/Course 等所有节点与关系？")
        if colc2.button("一键清空现存图谱", type="secondary", use_container_width=True, disabled=not confirm):
            try:
                with get_driver(bolt_uri, user, password, no_auth).session(database=database) as sess:
                    sess.run("MATCH (n) DETACH DELETE n")
                st.success("已清空图谱。")
            except Exception as e:
                st.error(f"清空失败：{e}")


# ==================== 检索辅助 ====================
def list_cities():
    query = "MATCH (ci:City) RETURN ci.name AS name ORDER BY name"
    with get_driver(bolt_uri, user, password, no_auth).session(database=database) as sess:
        rows = _execute_read(sess, lambda tx: run_query(tx, query, {}))
    return ["ALL"] + [r["name"] for r in rows]

def list_edus():
    query = "MATCH (e:EducationLevel) RETURN e.name AS name ORDER BY name"
    with get_driver(bolt_uri, user, password, no_auth).session(database=database) as sess:
        rows = _execute_read(sess, lambda tx: run_query(tx, query, {}))
    return ["ALL"] + [r["name"] for r in rows]

def row_has_keyword(row: Dict[str, Any], q: str, strict_company: bool) -> bool:
    if not q: return True
    ql = q.lower()
    # 简单同义：华为→huawei
    aliases = [ql]
    if ql == "华为": aliases.append("huawei")
    fields = ["companyName"] if strict_company else ["companyName","jobTitle","jobName","jobDescription","salaryText"]
    for f in fields:
        v = row.get(f) or ""
        if isinstance(v, list):
            v = " ".join([str(x) for x in v])
        if ql in str(v).lower(): return True
        for a in aliases[1:]:
            if a in str(v).lower(): return True
    return False

def draw_pyvis_from_company_or_job(rows: List[Dict[str, Any]]):
    net = Network(height="650px", width="100%", bgcolor="#ffffff", font_color="#111")
    def add_node(nid, label, title, group): net.add_node(nid, label=label, title=title, group=group)
    def add_edge(src, dst, label): net.add_edge(src, dst, title=label, label=label)
    if rows and "j" in rows[0]:  # by job
        r = rows[0]; j = r.get("j"); c = r.get("c"); cities = r.get("cities") or []; edus = r.get("edus") or []
        if j:
            node = j.get("j", j); jk = node.get("jobKey", ""); jname = node.get("title", "") or node.get("jobName", "")
            tip = (node.get("description") or node.get("jobDescription") or node.get("detail") or node.get("desc") or "")[:400]
            add_node(f"job:{jk}", f"岗位\n{jname}", tip or f"Job {jk}", "Job")
        if c:
            node = c.get("c", c); cid = node.get("companyId", ""); cname = node.get("companyName", "")
            add_node(f"company:{cid}", f"公司\n{cname}", f"Company {cid}", "Company")
            if j: add_edge(f"company:{cid}", f"job:{jk}", "POSTED")
        for ci in cities:
            node = ci.get("ci", ci); name = node.get("name", "")
            if name:
                add_node(f"city:{name}", f"城市\n{name}", name, "City")
                if j: add_edge(f"job:{jk}", f"city:{name}", "LOCATED_IN")
        for e in edus:
            node = e.get("e", e); name = node.get("name", "")
            if name:
                add_node(f"edu:{name}", f"学历\n{name}", name, "EducationLevel")
                if j: add_edge(f"job:{jk}", f"edu:{name}", "REQUIRES_EDU")
    else:
        if not rows: return net
        r = rows[0]; c = r.get("c"); jobs = r.get("jobs") or []; cities = r.get("cities") or []; edus = r.get("edus") or []
        if c:
            node = c.get("c", c); cid = node.get("companyId", ""); cname = node.get("companyName", "")
            add_node(f"company:{cid}", f"公司\n{cname}", f"Company {cid}", "Company")
        for j in jobs:
            node = j.get("j", j); jk = node.get("jobKey", ""); jname = node.get("title", "") or node.get("jobName", "")
            tip = (node.get("description") or node.get("jobDescription") or node.get("detail") or node.get("desc") or "")[:400]
            if jk:
                add_node(f"job:{jk}", f"岗位\n{jname}", tip or f"Job {jk}", "Job")
                if c: add_edge(f"company:{cid}", f"job:{jk}", "POSTED")
        names = set()
        for ci in cities:
            node = ci.get("ci", ci); name = node.get("name", "")
            if name and name not in names:
                add_node(f"city:{name}", f"城市\n{name}", name, "City"); names.add(name)
        ed = set()
        for e in edus:
            node = e.get("e", e); name = node.get("name", "")
            if name and name not in ed:
                add_node(f"edu:{name}", f"学历\n{name}", name, "EducationLevel"); ed.add(name)
    net.repulsion(node_distance=200, central_gravity=0.15, spring_length=120, spring_strength=0.06, damping=0.9)
    return net

# ==================== 知识图谱检索 ====================
with tab_explore:
    st.subheader("条件检索")
    cities = list_cities(); edus = list_edus()
    colf = st.columns([1.6,1,1,1,1,1.2,1.2])
    kw   = colf[0].text_input("关键词（职位/标题/公司）", value="", key="kw")
    city = colf[1].selectbox("城市", options=cities, index=0, key="city_filter")
    edu  = colf[2].selectbox("学历", options=edus, index=0, key="edu_filter")
    smin = colf[3].number_input("最低薪资（元/月）", min_value=0.0, value=0.0, step=1000.0, key="smin")
    smax = colf[4].number_input("最高薪资（0=不限）", min_value=0.0, value=0.0, step=1000.0, key="smax")
    strict = colf[5].selectbox("匹配模式", options=["模糊匹配","严格（仅公司名）"], index=0, key="match_mode")
    limit  = colf[6].slider("最大条数", min_value=50, max_value=1000, value=200, step=50, key="limit")

    def search_jobs_and_companies(q: str, city: Optional[str], edu: Optional[str],
                                  salary_min: Optional[float], salary_max: Optional[float], limit: int = 200,
                                  strict_company: bool = False):
        conds, params = [], {"q": q, "limit": limit}
        if q:
            if strict_company:
                conds.append("toLower(c.companyName) CONTAINS toLower($q)")
            else:
                conds.append("(toLower(j.jobName) CONTAINS toLower($q) OR toLower(coalesce(j.title,'')) CONTAINS toLower($q) OR toLower(c.companyName) CONTAINS toLower($q))")
        else:
            conds.append("true")
        if city and city != "ALL": conds.append("ci.name = $city"); params["city"] = city
        if edu and edu != "ALL":  conds.append("e.name = $edu");  params["edu"]  = edu
        where_clause = " AND ".join(conds)
        query = f"""
        MATCH (c:Company)-[:POSTED]->(j:Job)
        OPTIONAL MATCH (j)-[:LOCATED_IN]->(ci:City)
        OPTIONAL MATCH (j)-[:REQUIRES_EDU]->(e:EducationLevel)
        WHERE {where_clause}
        RETURN c.companyId AS companyId,
               c.companyName AS companyName,
               j.jobKey AS jobKey,
               coalesce(j.title, j.jobName) AS jobTitle,
               j.jobName AS jobName,
               coalesce(j.description, j.jobDescription, j.detail, j.jd, j.text, j.desc, j.content, j.requirement, j.requirements, "") AS jobDescription,
               coalesce(j.salaryMin, 0) AS salaryMin,
               coalesce(j.salaryMax, 0) AS salaryMax,
               coalesce(j.salaryMonth, 12) AS salaryMonth,
               coalesce(j.salaryText, "") AS salaryText,
               collect(DISTINCT ci.name) AS cities,
               coalesce(j.graduationYear, "") AS gradYear
        LIMIT $limit
        """
        with get_driver(bolt_uri, user, password, no_auth).session(database=database) as sess:
            rows = _execute_read(sess, lambda tx: run_query(tx, query, params))

        # 行级后置过滤 + 学历推断 + 薪资标准化
        normed = []
        for r in rows:
            grad_val = (r.get("gradYear") or "").strip()
            r["gradReq"] = grad_val if grad_val else "毕业不限"

            # 薪资标准化
            sal = normalize_salary(r.get("salaryMin"), r.get("salaryMax"), r.get("salaryMonth"), r.get("salaryText"))
            r["salaryPretty"] = sal["pretty"]
            r["salaryMonthlyMin"] = sal["min_month_cny"]
            r["salaryMonthlyMax"] = sal["max_month_cny"]
            # 金额过滤
            if salary_min and sal["min_month_cny"] < salary_min: continue
            if salary_max and salary_max > 0 and sal["max_month_cny"] > salary_max: continue
            # 行级关键词过滤
            if not row_has_keyword(r, q, strict_company): continue
            normed.append(r)
        return normed[:limit]

    if st.button("检索 / 刷新", type="primary", key="do_search"):
        with st.spinner("查询中... 执行检索"):
            smax_val = None if smax == 0.0 else smax
            smin_val = smin if smin > 0 else None
            results = search_jobs_and_companies(kw, city, edu, smin_val, smax_val, limit,
                                                strict_company=(strict=="严格（仅公司名）"))
            st.session_state["results"] = results

    results = st.session_state.get("results", [])
    st.subheader("结果")
    if results:
        df = pd.DataFrame(results)
        show = df[[
            "companyName","jobTitle","jobName","salaryPretty","cities","edu"
        ]].rename(columns={
            "companyName":"公司","jobTitle":"职位标题","jobName":"职位名称","salaryPretty":"薪资",
            "cities":"城市","gradReq":"毕业要求"
        })
        st.dataframe(show, use_container_width=True)
    else:
        st.markdown('<div class="cute">没有检索到内容 (ฅ•ω•ฅ)♡ 请尝试更换关键词或放宽条件~</div>', unsafe_allow_html=True)

    st.subheader("邻域图（拾取一行）")
    if results:
        df = pd.DataFrame(results); max_idx = len(df) - 1
        idx = st.number_input("选择行索引", min_value=0, max_value=max_idx, value=0, step=1, key="row_index_unique")
        selected = df.iloc[int(idx)].to_dict()

        def neighborhood(jobKey: Optional[str] = None, companyId: Optional[str] = None):
            if jobKey:
                query = """
                MATCH (j:Job {jobKey: $jobKey})
                OPTIONAL MATCH (c:Company)-[:POSTED]->(j)
                OPTIONAL MATCH (j)-[:LOCATED_IN]->(ci:City)
                OPTIONAL MATCH (j)-[:REQUIRES_EDU]->(e:EducationLevel)
                RETURN j, c, collect(DISTINCT ci) AS cities, collect(DISTINCT e) AS edus
                """
                params = {"jobKey": jobKey}
            else:
                query = """
                MATCH (c:Company {companyId: $companyId})-[:POSTED]->(j:Job)
                OPTIONAL MATCH (j)-[:LOCATED_IN]->(ci:City)
                OPTIONAL MATCH (j)-[:REQUIRES_EDU]->(e:EducationLevel)
                RETURN collect(DISTINCT j) AS jobs, c, collect(DISTINCT ci) AS cities, collect(DISTINCT e) AS edus
                """
                params = {"companyId": companyId}
            with get_driver(bolt_uri, user, password, no_auth).session(database=database) as sess:
                rows = _execute_read(sess, lambda tx: run_query(tx, query, params))
            return rows

        view_choice = st.radio("视图", options=["按职位", "按公司"], horizontal=True, key="view_choice_unique")
        rows = neighborhood(jobKey=selected["jobKey"]) if view_choice == "按职位" else neighborhood(companyId=selected["companyId"])
        net = draw_pyvis_from_company_or_job(rows)
        st.components.v1.html(net.generate_html(), height=680, scrolling=True)

# ==================== 画像与推荐（LLM + KG） ====================
with tab_rec:
    st.subheader("画像与推荐（LLM + 图谱）")
    st.caption("先用大模型生成『职业方向 + 学习路径初稿 + 需要补充信息』，可继续对话或直接用于图谱检索。")

    uinput = st.text_area("请输入：已掌握技能 / 兴趣方向 / 学历 / 可投入时间 / 期望城市 / 其它约束（自由描述即可）",
                          height=140, key="portrait_input")
    colx = st.columns(3)
    btn_analyze = colx[0].button("① 生成画像与路径（流式）", type="primary", key="btn_llm_plan")
    btn_to_kg  = colx[1].button("② 基于当前条件做图谱推荐", key="btn_to_kg")
    btn_reset  = colx[2].button("清空对话", key="btn_reset_conv")

    if btn_reset:
        st.session_state.pop("llm_raw_output", None)
        st.session_state.pop("llm_constraints", None)

    if btn_analyze:
        if not api_key:
            st.error("请在左侧填入 OpenAI API Key。")
        else:
            holder = st.empty(); buf = []
            with st.status("流式生成中…", expanded=True) as status:
                try:
                    for chunk in llm_helper.stream_plan(api_key, model, uinput, history=None):
                        buf.append(chunk); holder.markdown("".join(buf))
                    status.update(label="已生成初稿 ✅", state="complete")
                except Exception as e:
                    status.update(label=f"生成失败：{e}", state="error")
            st.session_state["llm_raw_output"] = "".join(buf)
            st.markdown("---")
            st.markdown("**初稿内容（可继续修改/补充）：**")
            st.text_area("大模型生成内容", value=st.session_state["llm_raw_output"], height=260, key="llm_raw_view")

    have_text = st.session_state.get("llm_raw_output") or uinput
    if have_text and st.button("抽取结构化条件（供图谱检索使用）", key="btn_extract"):
        if not api_key:
            st.error("请在左侧填入 OpenAI API Key。")
        else:
            with st.spinner("抽取中…"):
                constraints = llm_helper.extract_constraints(api_key, model, have_text)
            st.session_state["llm_constraints"] = constraints
            st.success("已抽取为结构化条件")
            st.json(constraints)

    cons = st.session_state.get("llm_constraints", {})
    if cons:
        st.markdown("---")
        st.subheader("岗位与课程推荐")

        skills = [s.strip().lower() for s in cons.get("skills_known", []) if s.strip()]
        city_pref_list = cons.get("preferred_cities", []) or ["ALL"]
        city_pref = city_pref_list[0] if city_pref_list else "ALL"
        job_desc_text = cons.get("job_desc_text", "") or ("; ".join(cons.get("target_roles", []) or []))

        colr = st.columns(2)
        with colr[0]:
            st.caption("岗位匹配（Skills→Jobs）")
            with st.spinner("计算匹配度…"):
                jobs = recommender.recommend_jobs(student_skills=skills, city=city_pref, topk=30,
                                                  uri=bolt_uri, database=database, no_auth=bool(no_auth),
                                                  user=user, password=password)
            if jobs:
                dfj = pd.DataFrame(jobs)
                st.dataframe(dfj[["title","jobKey","matched","total","score","cities"]]\
                             .rename(columns={"title":"职位","matched":"匹配技能数","total":"目标技能数","score":"得分","cities":"城市"}),
                             use_container_width=True)
                st.session_state["rec_jobs"] = jobs
            else:
                st.markdown('<div class="cute">还没有合适岗位 ( •́ .̫ •̀ ) 先补充一下你的技能/兴趣吧~</div>',
                            unsafe_allow_html=True)

        with colr[1]:
            st.caption("岗位文本→课程（KP召回）")
            with st.spinner("从岗位文本召回课程…"):
                kp_rec = recommender.recommend_courses_from_jobtext(
                    job_title=_to_text(cons.get("target_roles", [])[:3]).strip(),
                    job_desc=_to_text(cons.get("job_desc_text") or cons.get("target_roles") or "").strip(), topn_kp=60, topn_course=15,
                    uri=bolt_uri, database=database, no_auth=bool(no_auth),
                    user=user, password=password
                )
            courses = kp_rec.get("courses", [])
            if courses:
                st.dataframe(pd.DataFrame([{"课程名称": c["course"], "课程得分": c["score"], "覆盖KP数": c["kp_count"]} for c in courses]), use_container_width=True)
            else:
                st.markdown('<div class="cute">未召回课程 (｡•́︿•̀｡) 可以提供更具体的岗位描述试试~</div>', unsafe_allow_html=True)

        jobs = st.session_state.get("rec_jobs", [])
        if jobs:
            st.markdown("---")
            st.caption("学习路径（选择岗位，按缺失技能贪心覆盖）")
            idx = st.number_input("选择岗位索引", min_value=0, max_value=len(jobs)-1, value=0, step=1, key="pick_job_idx_llm")
            jobKey = jobs[int(idx)]["jobKey"]
            with st.spinner("计算课程覆盖与学习路径…"):
                gap_courses = recommender.recommend_courses_for_gap(skills, jobKey, topk=10,
                                                                    uri=bolt_uri, database=database, no_auth=bool(no_auth),
                                                                    user=user, password=password)
                path = recommender.plan_learning_path(skills, jobKey, max_courses=6,
                                                      uri=bolt_uri, database=database, no_auth=bool(no_auth),
                                                      user=user, password=password)
            if gap_courses:
                st.write("可覆盖缺失技能的课程（Top10）：")
                st.dataframe(pd.DataFrame(gap_courses), use_container_width=True)
            if path:
                st.write("缺失技能：", ", ".join(path.get("missing", [])))
                st.write("推荐学习路径（贪心覆盖）：")
                st.dataframe(pd.DataFrame(path.get("path", [])), use_container_width=True)
