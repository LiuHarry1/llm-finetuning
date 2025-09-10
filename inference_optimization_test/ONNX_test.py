import torch
import torch.nn as nn

def create_model():
    #3.1 定义 PyTorch 模型
    # 简单的全连接模型
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.fc1 = nn.Linear(10, 20)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(20, 2)

        def forward(self, x):
            return self.fc2(self.relu(self.fc1(x)))

    model = SimpleModel()
    result = model.eval()  # 导出前需要设为 eval 模式

    print(result)
    return model

def export_onnx(model):
    #3.2 导出为 ONNX
    dummy_input = torch.randn(1, 10)  # 模拟输入
    torch.onnx.export(
        model,
        dummy_input,
        "simple_model.onnx",
        input_names=["input"],
        output_names=["output"],
        opset_version=13  # ONNX opset 版本
    )
    print("ONNX 模型已导出！")

def inference_by_onnx():

    import onnxruntime as ort
    import numpy as np

    # 加载 ONNX 模型
    ort_session = ort.InferenceSession("simple_model.onnx")

    # 构造输入
    input_data = np.random.randn(1, 10).astype(np.float32)

    # 运行推理
    outputs = ort_session.run(None, {"input": input_data})
    print("输出结果:", outputs)

def quantize_by_onnx():
    from onnxruntime.quantization import quantize_dynamic, QuantType

    quantize_dynamic("simple_model.onnx", "simple_model_int8.onnx", weight_type=QuantType.QInt8)
    print("ONNX 模型已量化为 INT8！")


if __name__ == '__main__':
    model = create_model()
    export_onnx(model)
    inference_by_onnx()
    quantize_by_onnx()

