
#  ​IT-KGRec：基于知识图谱的智能行业推荐系统

## 功能一览
1. **数据采集**：抓取招聘岗位（Nowcoder），保持既有 CSV 格式，补充 `title/description`。
2. **清洗整合**：`clean_transform.py` 解析职位文本提取技能，生成 `*.skills.json`。
3. **图谱构建**：`kg_pipeline.py` 建立 Company / Job / City / EducationLevel；`edu_ingest.py` 导入 Course / KnowledgePoint / Skill，并建立关联。
4. **算法设计**：`recommender.py` 基于图谱的岗位匹配、课程补足与学习路径（贪心覆盖）。
5. **前端展示**：`kg_app.py`（Streamlit）一站式：抓取入库、课程导入、检索&图谱、智能推荐（带流式日志）。

## 快速开始
```bash
pip install -r requirements.txt
streamlit run kg_app.py
```

侧边栏配置 Neo4j（支持 **无认证**）。

### 采集入库（侧边栏→“数据与入库”）
- 填写关键词/页数/每页 → 点击 **开始抓取 → 入库**。
- 日志实时刷新，完成后可在“图谱探索”检索。

### 课程导入
- 默认示例：`courses_sample.csv`，可替换为你自己的（列：`courseId,name,school,url,points,skills`，其中 `points`/`skills` 用分号分隔）。

### 生成 Job 技能并入库
```python
import clean_transform as ct
ct.build_skill_set("./export/python_offer.csv", "./skills_lexicon.txt")  # 生成 ./export/python_offer.skills.json
```
再在前端点击“写入 Job→Skill 关联”。

### 推荐
- 在“智能推荐”输入学生技能和偏好城市，点击“匹配岗位”；
- 选中岗位后，会给出课程补足和学习路径。

## 结构化 Schema
- `(Company)-[:POSTED]->(Job)`
- `(Job)-[:LOCATED_IN]->(City)`
- `(Job)-[:REQUIRES_EDU]->(EducationLevel)`
- `(Job)-[:REQUIRES_SKILL]->(Skill)`
- `(Course)-[:COURSE_HAS_POINT]->(KnowledgePoint)-[:TEACHES]->(Skill)`

## 注意
- 若你的 Neo4j 未启用 APOC，也可使用（算法端不依赖 APOC）。
- **务必使用** `streamlit run kg_app.py` 启动前端。


# 教育知识图谱与岗位推荐（Nowcoder → Neo4j + Streamlit）

> 一套从 **岗位数据抓取与清洗 → 图谱建模与入库 → 课程知识点导入 → 人岗匹配、能力补足与学习路径推荐** 的端到端原型。前端基于 Streamlit，图数据库使用 Neo4j。

## ✨ 功能一览
- **岗位抓取**：从 Nowcoder 搜索接口拉取岗位数据，支持分页、自动重试与去重（`kg_pipeline.py`）。
- **文本清洗与增强**：聚合 JD 字段，正则提取学历/薪资等结构化要素（`nlp_utils.py`、`kg_enrich_hooks.py`）。
- **图谱构建**：写入 Company / Job / City / EducationLevel / Skill 等节点与关系（`kg_pipeline.py`）。
- **课程知识点导入**：按“课程—知识点（KP）—前置/后置/关联”导入（`edu_ingest.py`）。
- **推荐与画像**：基于图谱检索与规则匹配的人岗推荐、缺失技能补课课程、学习路径（贪心覆盖）（`recommender.py`）。
- **可视化与交互**：Streamlit 一站式操作 + PyVis 图可视化（`kg_app.py`）。

## 🧱 项目结构
```
README.md
clean_transform.py
config.py
course/
course/┐╬│╠╓¬╩╢╡π═╝╞╫_ALL.json
edu_ingest.py
export/
export/c++_offer.csv
export/python_offer.csv
import_to_neo4j.py
kg_app.py
kg_enrich_hooks.py
kg_pipeline.py
llm_helper.py
nlp_utils.py
postprocess_enrich.py
recommender.py
```

- `kg_app.py`：Streamlit 前端入口，集抓取、导入、检索、推荐于一体。
- `kg_pipeline.py`：岗位抓取与图谱入库的主流程（含重试与数据增强）。
- `edu_ingest.py`：课程/知识点 JSON 导入 Neo4j（含 PRE_REQ/POST_REQ/ASSOCIATED_WITH 关系）。
- `recommender.py`：人岗匹配、缺口技能与课程补足、学习路径生成。
- `nlp_utils.py`：JD 清洗、学历推断等 NLP 辅助方法。
- `kg_enrich_hooks.py`：写库时的字段增强与 UPSERT。
- `postprocess_enrich.py`：对既有 Job 节点做二次增强。
- `config.py`：集中配置（Neo4j、OpenAI、目录等）。
- `export/`：抓取后的扁平 CSV（示例 `python_offer.csv` 等）。
- `course/`：课程知识点 JSON（示例见下）。

## 🚀 快速开始

### 1) 依赖
- Python 3.10+
- Neo4j 5.x（社区版即可；**APOC 非必需**）
- 推荐：Chrome/Edge 用于调试抓取

### 2) 安装
```bash
pip install -U pip
pip install neo4j pandas requests urllib3 tenacity scikit-learn streamlit pyvis openai>=1.51.0
```

### 3) 环境配置（建议使用环境变量或 `.env`，不要把密钥写进仓库）
支持的变量（对应 `config.py` 默认值）：

- `NEO4J_BOLT_URI`（默认 `bolt://localhost:7687`）
- `NEO4J_DATABASE`（默认 `neo4j`）
- `NEO4J_USER` / `NEO4J_PASSWORD`（`NEO4J_NO_AUTH=true` 时可忽略）
- `NEO4J_NO_AUTH`（`true|false`，默认 `true`）
- `EXPORT_DIR`（默认 `./export`）
- `COURSE_JSON_DIR`（默认 `./course`）
- `OPENAI_API_KEY`（用于画像与建议）
- `OPENAI_MODEL`（默认 `gpt-4o`）

示例：
```bash
export NEO4J_BOLT_URI="bolt://localhost:7687"
export NEO4J_DATABASE="neo4j"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="your_password"
export NEO4J_NO_AUTH="false"
export OPENAI_API_KEY="sk-..."
```

### 4) 运行前端
```bash
streamlit run kg_app.py
```
界面包含：**招聘抓取 → 课程 JSON 导入 → 条件检索 → 邻域图 → 画像与推荐**。

## 🔌 命令行用法（可选）

**岗位抓取 + 入库**
```bash
python kg_pipeline.py --query "python" --pages 10 --page-size 20 \
  --bolt-uri "$NEO4J_BOLT_URI" --database "$NEO4J_DATABASE" \  --user "$NEO4J_USER" --password "$NEO4J_PASSWORD" --no-auth
```
参数：
- `--query` 关键词（默认 `python`）
- `--pages` 页数（默认 10）
- `--page-size` 每页条数（默认 20）
- `--no-auth` 与 `config.py` 一致（如 Neo4j 免密，传入则不认证）

**二次增强（例如为既有 Job 推断学历/清洗 JD）**
```bash
python postprocess_enrich.py
```

> 课程/知识点导入在前端中调用 `edu_ingest.ingest_courses_json()` 完成，如需 CLI 可自行封装调用。

## 🗂️ 数据格式

**课程知识点 JSON（示例前两条）**
```json
[
  {
    "课程名称": "中国近现代史纲要",
    "知识点ID": "ZGJXDSGY-01",
    "知识点名称": "近代中国的开端：鸦片战争",
    "知识点说明": "明确鸦片战争对中国社会结构与主权的冲击与近代化开端。",
    "子知识点": "",
    "前置知识点": "",
    "后置知识点": "ZGJXDSGY-02",
    "关联知识点": ""
  },
  {
    "课程名称": "中国近现代史纲要",
    "知识点ID": "ZGJXDSGY-02",
    "知识点名称": "自强求富：洋务运动",
    "知识点说明": "理解“师夷长技以制夷”的历史选择与成败得失。",
    "子知识点": "",
    "前置知识点": "ZGJXDSGY-01",
    "后置知识点": "ZGJXDSGY-03",
    "关联知识点": ""
  }
]
```

必备字段：
- `课程名称`、`知识点ID`、`知识点名称`、`知识点说明`
- 关系字段：`子知识点`、`前置知识点`、`后置知识点`、`关联知识点`（可留空；多值用逗号/分号分隔或数组）

导入后结构：
- `(:Course {name}) -[:HAS_KP]-> (:KP {kid})`
- `(:KP)-[:PRE_REQ|POST_REQ|ASSOCIATED_WITH]->(:KP)`（含 `label_cn` 属性）

**岗位 CSV（示例前两行）**
|   companyId |   bossUid | jobName                         | ext                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | jobCityList   | graduationYear   |   eduLevel |   salaryType |   salaryMin |   salaryMax |   salaryMonth | companyName   | scaleTagName   | personScales   | companyShortName   | address                            |
|------------:|----------:|:--------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------|:-----------------|-----------:|-------------:|------------:|------------:|--------------:|:--------------|:---------------|:---------------|:-------------------|:-----------------------------------|
|         139 |      9034 | AI agent 客户端SDK研发 - python | {"requirements":"1、熟悉python开发，对python 并发编程，网络编程有一定了解\n2、了解或者使用过GPT类大模型的客户端SDK优先，比如langchain、openai等\n3、对数据结构、设计模式一定理解\n4、了解VUE前端或者go相关的知识优先","infos":"大预言模型 客户端SDK开发\n- 客户端SDK开发、性能调优、架构优化  - （python） - 开源 \n      - 限速、服务端控制请求速率、全链路监控\n- 开源项目构建和文档管理\n  - 从0-1构建开源的客户端SDK，维护SDK项目"}                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | ["上海"]      | 毕业不限         |       5000 |            1 |         200 |         250 |             0 | 百度          | 上市           | 10000人以上    | 百度               | 北京                               |
|         896 |     33880 | 软件测试开发工程师（实习）      | {"requirements":"1、计算机相关专业，本科及以上学历\n2、有测试开发实习经验优先，有大模型调优、应用经验优先；\n3、具备扎实的计算机基础，熟练掌握至少一种编程语言如Java/JavaScript/Python/Shell等，具备良好的代码能力；\n4、了解各类软件测试方法和用例设计方法，掌握和应用单元测试、接口测试、系统测试、安全性测试、性能测试等测试手段 ；\n5、熟悉开源测试工具框架及相关扩展应用，熟悉网络协议和网络环境的应用，有实际某一项测试技术领域的应用和开发经验； \n6、思维活跃，有较强的逻辑思维能力，有较强的技术热情和学习欲望，责任心强，积极主动，具备良好的沟通能力和团队协作能力；\n7、有任一一项 移动端/PC端软件/Web端产品测试经验，有完整的整机产品交付经验优先；\n8、了解可靠性工程理论，了解概率统计理论基础相关知识者优先。\n\n\n","infos":"1、负责测试工具平台开发和维护优化； \n2、负责技术性专项测试工作（如性能、安全、功耗等）； \n3、负责提升整个研发团队的质量意识，落地开发测试流程最佳实践；\n4、负责通过其他类技术创新手段（如AI大模型应用）改善和提高整体研发质量和效率。"} | ["北京"]      | 毕业不限         |       5000 |            1 |         200 |         500 |            12 | 网易有道      | 上市           | 10000人以上    | 网易有道           | 南昌，重庆，昆明，北京，上海，东莞 |

## 🧩 图谱 Schema（核心）
- `(Company)-[:POSTED]->(Job)`
- `(Job)-[:LOCATED_IN]->(City)`
- `(Job)-[:REQUIRES_EDU]->(EducationLevel)`
- `(Job)-[:REQUIRES_SKILL]->(Skill)`
- `(Course)-[:HAS_KP]->(KP)` / `(KP)-[:PRE_REQ|POST_REQ|ASSOCIATED_WITH]->(KP)`

## 🧠 推荐逻辑（概览）
- **人岗匹配**：将学生自报技能标准化，与 `(:Job)-[:REQUIRES_SKILL]->(:Skill)` 交集/覆盖度排序。
- **缺口技能**：`Job` 目标岗位与学生技能的差集即为“待补足技能”。
- **课程补足**：匹配“覆盖缺口技能”的课程/知识点组合，采用贪心近似覆盖。
- **学习路径**：基于 KP 的 `PRE_REQ/POST_REQ` 拓扑，输出可执行的阶段化路径。
- **画像与建议（可选）**：调用 OpenAI（`llm_helper.py`）根据学生描述生成更结构化的规划建议。

## ⚙️ 运维与排错
- **Neo4j 连接失败**：检查 `bolt://host:7687` 可达、数据库名、认证开关（`NEO4J_NO_AUTH`）。
- **抓取节流**：`kg_pipeline.py` 内置 `tenacity` 指数退避与 `urllib3` 重试；可通过 `SLEEP_MIN/MAX` 微调间隔。
- **中文路径**：确保系统编码为 UTF-8；Windows 请使用 PowerShell（`chcp 65001`）。
- **密钥安全**：不要把 `OPENAI_API_KEY` 写入 `config.py`，用环境变量覆盖。

## 🧭 路线图
- [ ] `requirements.txt` / `poetry` 清单与版本钉住
- [ ] Docker 化（Neo4j + App 一键起）
- [ ] 课程导入的 CLI 封装与校验工具
- [ ] 推荐打分函数参数化（技能权重/地域/学历偏好）


