import numpy as np
import onnxruntime as ort
from PIL import Image
import torchvision.transforms as transforms
import onnx

# 图片预处理函数
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(256),              # 缩放图片使得短边为256
        transforms.CenterCrop(224),          # 中心裁剪得到224x224的图片
        transforms.ToTensor(),               # 将图片转换为Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 标准化
                             std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # 添加batch维度

# 加载ONNX模型,并且使用gpu进行推理
onnx_model_path = 'resnet34-bird-modified.onnx'  # 修改为修改后的模型名称
ort_session = ort.InferenceSession(onnx_model_path, providers=['CUDAExecutionProvider'])

# 加载并预处理图片
image_path = './丹顶鹤-001.jpg'
input_tensor = preprocess_image(image_path)

# ONNX Runtime推理
print(ort_session.get_inputs()[0].name)  # 'input'
input = {ort_session.get_inputs()[0].name: input_tensor.numpy()}
output = ort_session.run(None, input)

# 输出预测结果
# 找到最高分数的索引
predicted_index = np.argmax(output)
print(f"Predicted class index: {predicted_index}")            
