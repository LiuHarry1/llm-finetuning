import os
import json
import time
from datetime import datetime, timedelta
from openai import OpenAI

# -----------------------------
# 初始化 Qwen 客户端
# -----------------------------
client = OpenAI(
    # api_key=os.getenv("DASHSCOPE_API_KEY"),
    api_key="sk-f256c03643e9491fb1ebc278dd958c2d",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 模拟数据库
user_profiles = {}  # {user_id: {...偏好...}}

# 偏好提取 Prompt
def build_preference_prompt(chat_history, user_input):
    return f"""
你是一个业务知识问答助手，现在要从对话中提取用户的偏好信息。
对话历史：
{chat_history}

用户最新的问题：
{user_input}

请提取以下三类偏好：
1. 业务范围（Domain Scope）：如“支付”、“会员”、“物流”、“营销”、“风控”。
2. 答案风格（Answer Style）：如“简短摘要”、“详细步骤说明”、“表格形式”、“图表”。
3. 问题类型偏好（Question Type Preference）：如“数据指标查询”、“业务流程解释”、“异常分析”、“操作指南”。

只返回 JSON，例如：
{{
  "业务范围": ["会员", "营销"],
  "答案风格": "表格形式",
  "问题类型偏好": ["数据指标查询", "异常分析"]
}}
"""

# 调用 Qwen 提取偏好
def extract_user_preferences(chat_history, user_input):
    prompt_text = build_preference_prompt(chat_history, user_input)
    response = client.chat.completions.create(
        model="qwen-plus",
        messages=[{"role": "user", "content": prompt_text}],
        temperature=0
    )
    raw_text = response.choices[0].message.content
    try:
        return json.loads(raw_text)
    except:
        return {
            "业务范围": [],
            "答案风格": None,
            "问题类型偏好": []
        }

# 偏好衰减（默认30天）
def decay_preferences(profile, days_threshold=30):
    cutoff = datetime.now() - timedelta(days=days_threshold)
    for key in ["业务范围", "问题类型偏好"]:
        profile[key] = {
            k: v for k, v in profile.get(key, {}).items()
            if datetime.fromtimestamp(v["last_seen"]) >= cutoff
        }
    return profile

# 合并偏好并更新 last_seen
def merge_preferences(old_prefs, new_prefs):
    now_ts = time.time()

    # 业务范围
    old_scope = old_prefs.get("业务范围", {})
    for item in new_prefs.get("业务范围", []):
        old_scope[item] = {"last_seen": now_ts}
    old_prefs["业务范围"] = old_scope

    # 问题类型偏好
    old_qtype = old_prefs.get("问题类型偏好", {})
    for item in new_prefs.get("问题类型偏好", []):
        old_qtype[item] = {"last_seen": now_ts}
    old_prefs["问题类型偏好"] = old_qtype

    # 答案风格（直接覆盖最新的）
    if new_prefs.get("答案风格"):
        old_prefs["答案风格"] = new_prefs["答案风格"]

    return old_prefs

# 处理真实对话
def handle_user_message(user_id, user_input):
    # 获取历史画像（如果不存在，初始化）
    current_profile = user_profiles.get(user_id, {
        "业务范围": {},
        "答案风格": None,
        "问题类型偏好": {},
        "history": []
    })

    # 取对话历史
    chat_history = "\n".join(current_profile["history"])

    # 提取偏好
    new_prefs = extract_user_preferences(chat_history, user_input)

    # 合并偏好
    updated_profile = merge_preferences(current_profile, new_prefs)

    # 偏好衰减
    updated_profile = decay_preferences(updated_profile, days_threshold=30)

    # 保存历史对话
    updated_profile["history"].append(user_input)

    # 存储
    user_profiles[user_id] = updated_profile

    # 输出
    print(f"用户画像更新: {updated_profile}")
    return f"根据你的习惯（业务范围: {list(updated_profile['业务范围'].keys())}）为你生成回答"

# ===== 模拟真实对话 =====
print(handle_user_message("u123", "帮我查一下上个月会员促销活动的转化率"))
print(handle_user_message("u123", "给我一张表格，显示最近三个月VIP留存率变化"))
print(handle_user_message("u123", "分析一下物流延迟率高的原因"))
