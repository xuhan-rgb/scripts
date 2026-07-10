import ctypes
import importlib.util
import os
import subprocess


def print_cuda_runtime_version():
    try:
        cuda = ctypes.CDLL("libcudart.so")
        cuda.cudaRuntimeGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
        cuda.cudaRuntimeGetVersion.restype = ctypes.c_int
        version = ctypes.c_int()
        result = cuda.cudaRuntimeGetVersion(ctypes.byref(version))
        if result != 0:
            print(f"CUDA runtime version query failed with error code: {result}")
            return
        print("CUDA library loaded successfully!")
        print(
            "CUDA version: "
            f"{version.value // 1000}.{(version.value % 1000) // 10}"
        )
    except (AttributeError, OSError) as error:
        print("CUDA library could not be loaded:", error)


def print_cudnn_version():
    try:
        cudnn = ctypes.CDLL("libcudnn.so")
        cudnn.cudnnGetVersion.restype = ctypes.c_size_t
        print("cuDNN library loaded successfully!")
        print("cuDNN version:", cudnn.cudnnGetVersion())
    except (AttributeError, OSError) as error:
        print("cuDNN library could not be loaded:", error)


def find_ldd_dependency(ldd_output, library_name):
    for line in ldd_output.splitlines():
        if library_name not in line or "=>" not in line:
            continue
        return line.split("=>", 1)[1].strip().split(" ", 1)[0]
    return None


def print_dependency_result(lib_name, lib_path, ldd_output, marker):
    if lib_path and lib_path != "not":
        if os.path.islink(lib_path):
            print(f"{lib_name} path: {lib_path} -> {os.path.realpath(lib_path)}")
        else:
            print(f"{lib_name} path: {lib_path}")
        return

    print(f"{lib_name} not found in ONNX Runtime dependencies.")
    print(f"Filtered ldd output for {lib_name} debugging:")
    for line in ldd_output.splitlines():
        if marker in line:
            print(line)


def find_onnxruntime_dependencies():
    onnxruntime_spec = importlib.util.find_spec("onnxruntime")
    if not onnxruntime_spec:
        print("ONNX Runtime is not installed in the current environment.")
        return

    onnxruntime_path = os.path.dirname(onnxruntime_spec.origin)
    onnx_lib_path = os.path.join(
        onnxruntime_path, "capi", "libonnxruntime_providers_cuda.so"
    )
    if not os.path.exists(onnx_lib_path):
        print("ONNX Runtime GPU library not found.")
        return

    try:
        result = subprocess.run(
            ["ldd", onnx_lib_path], capture_output=True, check=False, text=True
        )
    except OSError as error:
        print("Could not run ldd:", error)
        return
    if result.returncode != 0:
        print("ldd failed:", result.stderr.strip())
        return

    cuda_path = find_ldd_dependency(result.stdout, "libcudart.so")
    cudnn_path = find_ldd_dependency(result.stdout, "libcudnn.so")
    print_dependency_result("CUDA library", cuda_path, result.stdout, "libcudart.so")
    print_dependency_result("cuDNN library", cudnn_path, result.stdout, "libcudnn.so")


if __name__ == "__main__":
    print_cuda_runtime_version()
    print_cudnn_version()
    find_onnxruntime_dependencies()
