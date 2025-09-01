def llm_with_hitl(prompt):
    # Step 1: 初始推理
    step1 = f"LLM 根据 {prompt} 生成初稿..."
    human_feedback = yield step1  # 暂停，等待人类反馈

    if human_feedback == "修改":
        step2 = "人类要求修改 → 模型调整生成结果"
    else:
        step2 = "继续生成 → 模型展开更多细节"

    human_feedback = yield step2  # 再次暂停

    step3 = f"最终输出（结合人类反馈: {human_feedback}）"
    yield step3

gen = llm_with_hitl("写一首诗")
print(next(gen))                   # 模型生成初稿
print(gen.send("修改"))            # 人类说“修改”
print(gen.send("接受最终版本"))    # 人类确认 → 得到最终输出
