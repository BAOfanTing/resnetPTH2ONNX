import onnx
from onnx import numpy_helper
import torch
from model import resnet34,resnet50
import torch.nn as nn

pth_path = './resNet50-bird.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
example = torch.randn(1,3, 448, 448).to(device)     # 1 3 224 224
print(example.dtype)

#加载模型
model = resnet50()     
in_channel = model.fc.in_features
model.fc = nn.Linear(in_channel, 15)  # 更改全连接层的输出特征数量为为自己训练的类别数
model.load_state_dict(torch.load(pth_path)) 
model = model.to(device)                            
model.eval()

# 导出模型
torch.onnx.export(model, example, r"resnet50-bird.onnx",input_names=["input"], output_names=["output"])     
model_onnx = onnx.load(r"resnet50-bird.onnx")                   # onnx加载保存的onnx模型

onnx.checker.check_model(model_onnx)                    # 检查模型是否有问题
print(onnx.helper.printable_graph(model_onnx.graph))    # 打印onnx网络