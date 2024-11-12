import subprocess
import os
import importlib.util
import ctypes

# 检查 CUDA
try:
    cuda = ctypes.CDLL("libcudart.so")
    print("CUDA 库加载成功！")
    cuda_version = ctypes.c_int()
    cuda.cudaRuntimeGetVersion(ctypes.byref(cuda_version))
    print(f"CUDA 版本: {cuda_version.value // 1000}.{(cuda_version.value % 1000) // 10}")
except OSError as e:
    print("CUDA 库加载失败:", e)
# 检查 cuDNN
try:
    cudnn = ctypes.CDLL("libcudnn.so")
    print("cuDNN 库加载成功！")
    cudnn_version = ctypes.c_int()
    cudnn.cudnnGetVersion.restype = ctypes.c_size_t
    print("cuDNN 版本:", cudnn.cudnnGetVersion())
except OSError as e:
    print("cuDNN 库加载失败:", e)
    
def find_onnxruntime_dependencies():
    # 尝试导入 onnxruntime 或 onnxruntime-gpu
    try:
        onnxruntime_spec = importlib.util.find_spec("onnxruntime")
        if not onnxruntime_spec:
            print("ONNX Runtime is not installed in the current environment.")
            return

        # 获取 onnxruntime 库的路径
        onnxruntime_path = os.path.dirname(onnxruntime_spec.origin)
        onnx_lib_path = os.path.join(onnxruntime_path, "capi", "libonnxruntime_providers_cuda.so")

        if not os.path.exists(onnx_lib_path):
            print("ONNX Runtime GPU library not found.")
            return

        # 使用 ldd 命令获取依赖信息
        try:
            ldd_output = subprocess.check_output(["ldd", onnx_lib_path]).decode("utf-8")
        except subprocess.CalledProcessError as e:
            print("Error running ldd:", e)
            return

        # 查找 CUDA 和 cuDNN 版本信息
        cuda_version = None
        cudnn_version = None
        for line in ldd_output.splitlines():
            if "cuda" in line:
                cuda_version = line.split("=>")[-1].strip().split(" ")[0]
            elif "cudnn" in line:
                cudnn_version = line.split("=>")[-1].strip().split(" ")[0]

        # 输出结果
        def print_library_path(lib_name, lib_path):
            if os.path.islink(lib_path):
                real_path = os.path.realpath(lib_path)
                print(f"{lib_name} path: {lib_path} -> {real_path}")
            else:
                print(f"{lib_name} path: {lib_path}")
        
        if cuda_version != 'not':
            print_library_path("CUDA library", cuda_version)
        else:
            print("CUDA library not found in ONNX Runtime dependencies.")
            print("Filtered ldd output for CUDA debugging:")
            for line in ldd_output.splitlines():
                if "cuda" in line and "cudnn" not in line:
                    print(line)

        if cudnn_version != 'not':
            print_library_path("cuDNN library", cudnn_version)
        else:
            print("cuDNN library not found in ONNX Runtime dependencies.")
            print("Filtered ldd output for cuDNN debugging:")
            for line in ldd_output.splitlines():
                if "cudnn" in line:
                    print(line)

    except ImportError:
        print("ONNX Runtime is not installed.")

if __name__ == "__main__":
    find_onnxruntime_dependencies()

