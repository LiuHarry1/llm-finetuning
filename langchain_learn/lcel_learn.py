from typing import Callable, Any

# 定义抽象 Runnable
class Runnable:
    def invoke(self, input: Any) -> Any:
        raise NotImplementedError

    # 重载 | 运算符，返回一个可组合的序列
    def __or__(self, other: "Runnable") -> "RunnableSequence":
        return RunnableSequence([self, other])


# 序列执行器
class RunnableSequence(Runnable):
    def __init__(self, steps):
        self.steps = steps

    def invoke(self, input: Any) -> Any:
        output = input
        for step in self.steps:
            output = step.invoke(output)
        return output

    # 支持继续用 | 组合
    def __or__(self, other: Runnable) -> "RunnableSequence":
        return RunnableSequence(self.steps + [other])


# 一个简单的 Prompt 模块
class Prompt(Runnable):
    def __init__(self, template: str):
        self.template = template

    def invoke(self, input: dict) -> str:
        return self.template.format(**input)


# 一个简单的 LLM 模拟器
class FakeLLM(Runnable):
    def invoke(self, input: str) -> str:
        return f"🤖 模型回答: {input}"


# 一个简单的 Parser
class SimpleParser(Runnable):
    def invoke(self, input: str) -> dict:
        return {"parsed_text": input.upper()}


# ---------------- 使用示例 ----------------
if __name__ == "__main__":
    # 定义组件
    prompt = Prompt("请回答这个问题: {question}")
    llm = FakeLLM()
    parser = SimpleParser()

    # 用 LCEL 风格组合
    chain = prompt | llm | parser

    # 调用链
    result = chain.invoke({"question": "什么是LCEL？"})
    print(result)
