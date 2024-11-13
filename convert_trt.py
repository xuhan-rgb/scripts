import cv2
import tensorrt as trt
import onnx

import torch
from torch import nn
import numpy as np
import pycuda.driver as cudart

# onnx_path = "simple_model.onnx"
# calibration_data = np.random.randn(100, 3, 352, 640).astype(np.float32)  # 100个样本

# # 定义一个简单的全连接模型
# class SimpleModel(nn.Module):
#     def __init__(self):
#         super(SimpleModel, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
#         with torch.no_grad():
#             self.conv1.weight = torch.nn.Parameter(torch.ones(16, 3, 3, 3))  # 权重: (16, 3, 3, 3)
#             self.conv1.bias = torch.nn.Parameter(torch.zeros(16))  # 偏置: (16,)
    
#     def forward(self, x):
#         x = self.conv1(x)
#         x = torch.relu(x)
#         return x

# # 创建模型实例
# model = SimpleModel()

# # 创建一个示例输入
# x = torch.randn(1, 3, 352, 640)  # 批次大小为1，输入维度为3

# # 导出模型为ONNX格式
# torch.onnx.export(model, x, onnx_path, input_names=['input'], output_names=['output'])

# print(f"ONNX model saved to {onnx_path}")

# def load_onnx_model(onnx_path, logger):
#     """
#     加载 ONNX 模型并转换为 TensorRT 网络。
    
#     参数：
#     onnx_path (str): ONNX 模型文件的路径。
#     logger (trt.Logger): 用于记录日志的 TensorRT Logger 对象。
    
#     返回：
#     network (trt.INetworkDefinition): 转换后的 TensorRT 网络。
#     """
#     builder = trt.Builder(logger)
#     network = builder.create_network()
    

#     # 创建 ONNX 解析器
#     parser = trt.OnnxParser(network, logger)
#     # 加载 ONNX 模型并进行解析
#     with open(onnx_path, "rb") as f:
#         model_data = f.read()
#         if not parser.parse(model_data):
#             print("ONNX 模型解析失败:")
#             for error in parser.errors:
#                 print(error)
#             return None
#     print("ONNX 模型加载成功")
#     return network

# # 定义 INT8 校准器
# class Int8Calibrator(trt.IInt8Calibrator):
#     def __init__(self, calibration_data):
#         super().__init__()
#         self.data = calibration_data
#         self.index = 0
#         self.cache = None

#     def get_batch_size(self):
#         return 1

#     def get_batch(self, names):
#         if self.index < len(self.data):
#             batch = self.data[self.index]
#             self.index += 1
#             print("#########################")
#             return [batch.astype(np.float32)]  # 确保返回的是正确格式的 batch
#         print("**********")
#         return None

#     def read_calibration_cache(self):
#         return self.cache

#     def write_calibration_cache(self, cache):
#         self.cache = cache
#         print("Calibration cache has been saved.")

#     def get_algorithm(self):
#         return trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2  # 选择熵校准算法

# # 创建 TensorRT builder 和网络
# builder = trt.Builder(trt.Logger(trt.Logger.WARNING))
# network = load_onnx_model(onnx_path, trt.Logger(trt.Logger.WARNING))

# config = builder.create_builder_config()
# # 启用 INT8 模式并设置校准器
# # config.int8_mode = True
# config.int8_calibrator = Int8Calibrator(calibration_data)

# # 构建并序列化 TensorRT 引擎
# try:
#     serialized_engine = builder.build_serialized_network(network, config)
#     if serialized_engine is None:
#         raise Exception("Failed to build serialized engine")
#     engine_path = "int8_model.engine"
#     with open(engine_path, "wb") as f:
#         f.write(serialized_engine)
#     print(f"INT8 model engine saved to {engine_path}")
# except Exception as e:
#     print(f"Error during engine creation: {e}")

#################################测试代码##########
from tensorrt_inference import TensorRTModel
  # 使用TensorRTModel进行推理
# tensorrt_model = TensorRTModel(onnx_model_path='model/yolo11s_1class_1102.onnx')
tensorrt_model = TensorRTModel(onnx_model_path='model/depth_model.onnx')
# tensorrt_model = TensorRTModel(engine_model_path='model/depth_model.engine')

# 创建一个示例输入数据
img_data = cv2.imread('images/depth_image.jpg')
img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
img_data = cv2.resize(img_data, (640, 352))
img_data = img_data.transpose(2, 0, 1)
img_data = img_data.reshape(1, 3, 352, 640)/255.0
 
output = tensorrt_model.infer(img_data)
# Run inference
import time
start_time = time.time()
outputs = tensorrt_model.infer(img_data)
end_time = time.time()
print("Inference time:", (end_time - start_time)/100)
for output in outputs:
    print(output.shape)
# 打印推理结果
# print("Inference output:", output)

# 释放资源
tensorrt_model.release_resources()

################可视化###################
import os
import matplotlib as mpl
import matplotlib.cm as cm
# 创建彩色深度图
disp_np = outputs[0][0, 0]

vmax = np.percentile(disp_np, 95)
normalizer = mpl.colors.Normalize(vmin=disp_np.min(), vmax=vmax)
mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
colormapped_im = (mapper.to_rgba(disp_np)[:, :, :3] * 255).astype(np.uint8)
colormapped_im_bgr = cv2.cvtColor(colormapped_im, cv2.COLOR_RGB2BGR)

cv2.imwrite('images/depth_colormapped.jpg', colormapped_im_bgr) # 保存结果
# 判断是否在 SSH 环境中
is_ssh = "SSH_CLIENT" in os.environ or "SSH_TTY" in os.environ

# 显示图像（如果不是 SSH 终端）
if not is_ssh:
    cv2.imshow('colormapped_im_bgr', colormapped_im_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Detected SSH environment, skipping image display.")
    