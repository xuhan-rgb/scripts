import cv2
import numpy as np
import os
import time
import matplotlib as mpl
import matplotlib.cm as cm
from tensorrt_inference import TensorRTModel

# 加载模型
tensorrt_model = TensorRTModel(onnx_model_path='model/depth_model.onnx')

# 读取并预处理图像
img_data = cv2.imread('images/depth_image.jpg')
img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
img_data = cv2.resize(img_data, (640, 352))
img_data = img_data.transpose(2, 0, 1)
img_data = img_data.reshape(1, 3, 352, 640) / 255.0

# 推理
start_time = time.time()
outputs = tensorrt_model.infer(img_data)
end_time = time.time()
print("Inference time:", end_time - start_time)
for output in outputs:
    print(output.shape)

tensorrt_model.release_resources()

# 可视化深度图
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
    