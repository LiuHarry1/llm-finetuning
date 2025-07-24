import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# 构造数据（sin 函数 + 噪声）
x = torch.linspace(0, 2 * np.pi, 100).unsqueeze(1)
y = torch.sin(x) + 0.1 * torch.randn_like(x)  # 添加噪声


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)


def train(weight_decay_value):
    model = SimpleNet()
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=weight_decay_value)
    loss_fn = nn.MSELoss()

    for epoch in range(500):
        #告诉模型：进入训练模式（比如启用 dropout、batchnorm 的训练行为）。
        model.train()
        #前向传播（预测）
        y_pred = model(x)
        # 计算损失
        loss = loss_fn(y_pred, y)
        # 反向传播准备
        #清空之前累积的梯度，防止干扰新一轮的计算。
        optimizer.zero_grad()
        #反向传播（求梯度）
        #根据损失反向传播，计算参数的梯度。
        loss.backward()
        # 根据梯度和优化器策略（含 weight decay）更新模型参数。
        optimizer.step()
    return model

model_no_decay = train(0.0)
model_with_decay = train(0.01)

# 绘图
plt.figure(figsize=(10, 5))
plt.scatter(x.numpy(), y.numpy(), label='Data', color='gray', alpha=0.5)

with torch.no_grad():
    y_pred1 = model_no_decay(x)
    y_pred2 = model_with_decay(x)

plt.plot(x.numpy(), y_pred1.numpy(), label='No Weight Decay', color='blue')
plt.plot(x.numpy(), y_pred2.numpy(), label='With Weight Decay', color='red')
plt.legend()
plt.title("Effect of Weight Decay on Fitting Sin Curve")
plt.show()

