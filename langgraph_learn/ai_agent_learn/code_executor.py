from langchain_experimental.tools import PythonREPLTool

python_tool = PythonREPLTool()
def test1():


    code_to_run = "asdfasf```python print(sum([1, 2, 3, 4, 5])) ```dads"

    result = python_tool.run(code_to_run)

    print("执行代码:", code_to_run)
    print("返回结果:", result)

def test2():
    code_to_run = """
    import matplotlib.pyplot as plt

    x = [1, 2, 3, 4, 5]
    y = [i**2 for i in x]

    plt.plot(x, y)
    plt.title('示例曲线')
    plt.savefig('plot.png')  # 保存图片
    plt.close()
    'plot.png'  # 返回文件名
    """

    result = python_tool.run(code_to_run)
    print("图片文件路径:", result)

if __name__ == '__main__':
    test1()