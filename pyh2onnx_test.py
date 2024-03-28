from PIL import Image
import numpy as np
import onnxruntime

#测试onnx模型
# Load the image
image_path = "./丹顶鹤-001.jpg"
image = Image.open(image_path)

# Preprocess the image
image = image.resize((224, 224))
image = np.array(image, dtype=np.float32)
image /= 255.0
image = np.expand_dims(image, axis=0)

# Load the ONNX model
session = onnxruntime.InferenceSession("resnet34-bird.onnx")

# Run the inference
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
output = session.run([output_name], {input_name: image})

# Get the predicted class
predicted_class = np.argmax(output)

# Print the predicted class
print("Predicted class:", predicted_class)                
