from typing import Dict, Any, List, Optional

def _get_client(api_key: str):
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("缺少 openai 库，请先安装：pip install openai>=1.51.0") from e
    if not api_key:
        raise RuntimeError("未提供 OpenAI API Key。")
    return OpenAI(api_key=api_key)

_SYSTEM = (
    "你是高校职业与学习路径规划顾问。结合用户输入的个人背景，给出："
    "1) 职业方向（3-5 条，说明匹配依据与岗位画像）；"
    "2) 学习路径初稿（阶段化 3-5 步，每步建议课程/知识点与目标能力）；"
    "3) 需要补充的信息清单（5-8 条问题）。风格需结构化、简洁、可执行。"
)

def stream_plan(api_key: str, model: str, user_input: str,
                history: Optional[List[Dict[str, str]]] = None):
    client = _get_client(api_key)
    messages = [{"role":"system","content":_SYSTEM}]
    if history: messages += history
    messages.append({"role":"user","content": user_input})
    try:
        with client.chat.completions.create(
            model=model, messages=messages, temperature=0.2, stream=True
        ) as stream:
            for event in stream:
                if hasattr(event.choices[0], "delta") and event.choices[0].delta \
                   and event.choices[0].delta.content:
                    yield event.choices[0].delta.content
    except Exception:
        # 失败时退化为非流式
        resp = client.chat.completions.create(
            model=model, messages=messages, temperature=0.2, stream=False
        )
        yield resp.choices[0].message.content

def extract_constraints(api_key: str, model: str, convo_text: str) -> Dict[str, Any]:
    """从自由文本中抽取用于图谱检索的结构化条件。"""
    client = _get_client(api_key)
    sys = (
        "将文本解析为 JSON: "
        "target_roles[str[]], skills_known[str[]], preferred_cities[str[]], degree[str], "
        "time_budget[str], learning_prefs[str[]], salary_expect[str], constraints[str[]], "
        "job_desc_text[str]。"
    )
    messages = [{"role":"system","content":sys}, {"role":"user","content":convo_text}]
    try:
        resp = client.chat.completions.create(
            model=model, messages=messages, temperature=0.1,
            response_format={"type":"json_object"}
        )
        import json as _json
        return _json.loads(resp.choices[0].message.content)
    except Exception:
        return {
            "target_roles": [], "skills_known": [], "preferred_cities": [], "degree":"",
            "time_budget":"", "learning_prefs": [], "salary_expect":"",
            "constraints": [], "job_desc_text": convo_text[:800]
        }