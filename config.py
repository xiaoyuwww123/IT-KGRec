# -*- coding: utf-8 -*-
# 配置集中管理：兼容现有 kg_pipeline 引用，新增 OpenAI 配置
import os

# —— Neo4j —— #
BOLT_URI  = os.getenv("NEO4J_BOLT_URI", "bolt://localhost:7687")
DATABASE  = os.getenv("NEO4J_DATABASE", "neo4j")
USER      = os.getenv("NEO4J_USER", "neo4j")
PASSWORD  = os.getenv("NEO4J_PASSWORD", "neo4j")
NO_AUTH   = os.getenv("NEO4J_NO_AUTH", "true").lower() in ("1","true","yes")

# —— kg_pipeline 兼容 —— #
EXPORT_DIR = os.getenv("EXPORT_DIR", "./export")
SLEEP_MIN  = float(os.getenv("SLEEP_MIN", "0.8"))
SLEEP_MAX  = float(os.getenv("SLEEP_MAX", "2.2"))

# —— 课程知识点 JSON 目录（你要新建 ./course 并放入 JSON）—— #
COURSE_JSON_DIR = "./course"

# —— OpenAI —— #
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o")  # 按你的要求用 4o

# 保障目录
try:
    os.makedirs(EXPORT_DIR, exist_ok=True)
    os.makedirs(COURSE_JSON_DIR, exist_ok=True)
except Exception:
    pass
