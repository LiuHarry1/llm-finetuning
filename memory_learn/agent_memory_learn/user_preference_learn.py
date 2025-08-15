import os
import json
import time
from openai import OpenAI

# -----------------------------
# 初始化 Qwen 客户端
# -----------------------------
client = OpenAI(
    # api_key=os.getenv("DASHSCOPE_API_KEY"),
    api_key="sk-f256c03643e9491fb1ebc278dd958c2d",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# -----------------------------
# 模拟数据库（用户画像 + 历史对话）
# -----------------------------
user_profiles = {}

# -----------------------------
# 读取用户画像
# -----------------------------
def get_user_profile(user_id):
    if user_id not in user_profiles:
        user_profiles[user_id] = {
            "业务范围": {},
            "答案风格": None,
            "问题类型偏好": {},
            "history": []  # 保存最近几轮对话
        }
    return user_profiles[user_id]

# -----------------------------
# 生成上下文摘要（取最近5条对话）
# -----------------------------
def summarize_context(history, max_rounds=5):
    recent = history[-max_rounds:]
    return "\n".join(recent)

# -----------------------------
# 偏好补全意图 + 上下文记忆
# -----------------------------
def complete_intent_with_context(user_profile, user_input):
    context = summarize_context(user_profile["history"])
    prompt = f"""
已知用户画像：
业务范围: {list(user_profile['业务范围'].keys())}
答案风格: {user_profile['答案风格']}
问题类型偏好: {list(user_profile['问题类型偏好'].keys())}

用户最近的对话历史：
{context}

用户的新问题：
{user_input}

请结合用户画像和最近对话补全完整查询意图，包括：
- 业务领域（默认使用画像或历史对话中的业务范围）
- 查询目标（指标/问题类型）
- 查询条件（时间范围、过滤条件）
- 输出格式（结合用户答案风格）

返回 JSON，例如：
{{
    "业务领域": "会员",
    "查询目标": "留存率",
    "时间范围": "最近30天",
    "输出格式": "表格形式"
}}
"""
    resp = client.chat.completions.create(
        model="qwen-plus",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    try:
        return json.loads(resp.choices[0].message.content)
    except:
        return {}

# -----------------------------
# 模拟业务 API
# -----------------------------
def query_business_data(intent):
    if intent.get("查询目标") == "留存率":
        return [
            {"日期": "2025-08-01", "留存率": "45%"},
            {"日期": "2025-08-02", "留存率": "47%"}
        ]
    elif intent.get("查询目标") == "转化率":
        return [
            {"日期": "2025-08-01", "转化率": "12%"},
            {"日期": "2025-08-02", "转化率": "13%"}
        ]
    else:
        return [{"提示": "暂无数据"}]

# -----------------------------
# 个性化答案生成
# -----------------------------
def format_answer(data, intent):
    if intent.get("输出格式") == "表格形式" and isinstance(data, list):
        headers = data[0].keys()
        table = "| " + " | ".join(headers) + " |\n"
        table += "|" + " | ".join(["---"] * len(headers)) + "|\n"
        for row in data:
            table += "| " + " | ".join(str(v) for v in row.values()) + " |\n"
        return table
    else:
        return json.dumps(data, ensure_ascii=False, indent=2)

# -----------------------------
# 更新用户偏好画像
# -----------------------------
def merge_preferences(old_prefs, intent):
    now_ts = time.time()
    # 业务范围
    if intent.get("业务领域"):
        old_scope = old_prefs.get("业务范围", {})
        old_scope[intent["业务领域"]] = {"last_seen": now_ts}
        old_prefs["业务范围"] = old_scope
    # 问题类型
    if intent.get("查询目标"):
        old_qtype = old_prefs.get("问题类型偏好", {})
        old_qtype[intent["查询目标"]] = {"last_seen": now_ts}
        old_prefs["问题类型偏好"] = old_qtype
    # 答案风格
    if intent.get("输出格式"):
        old_prefs["答案风格"] = intent["输出格式"]
    return old_prefs

# -----------------------------
# 主流程
# -----------------------------
def handle_user_message(user_id, user_input):
    profile = get_user_profile(user_id)
    # 结合偏好和多轮上下文补全意图
    intent = complete_intent_with_context(profile, user_input)
    # 调用业务 API
    data = query_business_data(intent)
    # 个性化输出
    answer = format_answer(data, intent)
    # 更新用户画像 + 对话历史
    updated_profile = merge_preferences(profile, intent)
    updated_profile["history"].append(user_input)
    user_profiles[user_id] = updated_profile
    return answer

# ======== 模拟多轮对话 ========
print(handle_user_message("u123", "帮我看一下最近的留存"))
print(handle_user_message("u123", "转化率也给我看看"))
print(handle_user_message("u123", "会员付费情况呢？"))
