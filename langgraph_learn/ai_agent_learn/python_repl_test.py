from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL # 修改成你文件的名字


def main():
    repl = PythonREPL()

    print("=== Test 1: sanitize_input ===")
    print(PythonREPL.sanitize_input("   python  print('hi')   "))  # 预期: print('hi')
    print(PythonREPL.sanitize_input("```python\nx=1```"))          # 预期: x=1

    print("\n=== Test 2: simple print ===")
    print(repl.run("print('hello world')"))  # 预期输出: hello world

    print("\n=== Test 3: variable persistence ===")
    repl.run("x = 42")
    print(repl.run("print(x)"))  # 预期输出: 42

    print("\n=== Test 4: exception handling ===")
    print(repl.run("1/0"))  # 预期包含 ZeroDivisionError

    print("\n=== Test 5: timeout ===")
    print(repl.run("while True: pass", timeout=1))  # 预期: Execution timed out

    print("\n=== Test 6: globals and locals persistence ===")
    repl.run("y = 99")
    print(repl.run("print(y)"))  # 预期输出: 99

def test_run():

    # 初始化 REPL
    repl = PythonREPL()

    # 定义要执行的 Python 代码
    code = """
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 2 * np.pi, 400)
y = np.sin(x ** 2)

plt.figure(figsize=(6, 4))
plt.plot(x, y, label="y = sin(x²)")
plt.title("Example Image via PythonREPL")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)

plt.savefig("repl_plot.png")
plt.close()
print("Image saved as repl_plot.png")
    """

    # 用 REPL 执行
    output = repl.run(code)
    print(output)


@tool
def python_repl(code: str) -> str:
    """
    Use this function to execute Python code and get the results.
    """
    repl = PythonREPL()
    try:
        print("Running the Python REPL tool")
        print(code)
        result = repl.run(code)
        print(result)
    except BaseException as e:
        return f"Failed to execute. Error: {e!r}"
    return f"Result of code execution: {result}"



if __name__ == "__main__":
    # main()
    # test_run()

    tools = [python_repl]
    tools_by_name = {tool.name: tool for tool in tools}
    print(tools_by_name)
