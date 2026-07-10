# 工程审查 - 汇总

> 扫描日期：2026-07-10
> 扫描方向：汇总（运行正确性 + 资源管理 + 工程规范）
> 状态：✅ 已执行

## 汇总

| 编号 | 类别 | 审查项 | 等级 | 批复 | 状态 |
| --- | --- | --- | --- | --- | --- |
| R-M1 | 风险 | ONNX CUDA 检查可静默回退 CPU | 🟡 P1 | 修复 | ✅ 已修复 |
| R-M2 | 风险 | TensorRT I/O 顺序和 dtype 被硬编码 | 🟡 P1 | 修复 | ✅ 已修复 |
| R-M3 | 风险 | engine 构建/加载失败与重复释放缺少防护 | 🟡 P1 | 修复 | ✅ 已修复 |
| R-M4 | 风险 | `ldd` 未匹配依赖时诊断脚本可能崩溃 | 🟡 P1 | 修复 | ✅ 已修复 |
| O-M1 | 优化 | 深度示例导入即推理、失败路径未释放资源 | 🟡 P1 | 修复 | ✅ 已修复 |
| E-L1 | 工程规范 | README 返回类型错误且缺少可执行回归测试 | 🟢 P2 | 修复 | ✅ 已修复 |

> **统计**：✅ 已修复 6 项

## 风险项

### R-M1. ONNX CUDA 验证从 CPU 假成功改为失败退出

- **文件**：`test_onnx_env.py`
- **现状**：`InferenceSession` 请求 CUDA 后可回退到 CPU，原脚本仍以成功结束。
- **执行结果**：改用 `get_available_providers()` 预检，并在创建 session 后再次断言 CUDA Provider；当前机器缺少 ORT 所需的 `libcudnn.so.8` 时已验证返回退出码 `1`。

### R-M2. 按 TensorRT 元数据绑定 I/O 并采用真实 dtype

- **文件**：`tensorrt_inference.py`
- **现状**：原实现把第一个 I/O tensor 当输入，全部 tensor 都分配为 `float32`。
- **执行结果**：以 `get_tensor_mode()` 分类 I/O，读取 engine 声明的 dtype；明确拒绝多输入和动态 shape，避免错误绑定或未定义缓冲区。

### R-M3. 强化 engine 缓存和资源生命周期

- **文件**：`tensorrt_inference.py`
- **现状**：构建失败可留下空缓存，反序列化/执行失败不透明，重复释放可能二次 free。
- **执行结果**：增加构建、反序列化、执行和地址绑定检查；以临时文件原子发布缓存；缓存加载失败时重建；`release_resources()` 现为幂等。

### R-M4. 修正 ONNX Runtime 依赖解析

- **文件**：`get_onnx_dependencies.py`
- **现状**：未匹配到 CUDA/cuDNN 时会把 `None` 传给文件系统函数，且 CUDA 匹配过宽。
- **执行结果**：精确解析 `libcudart.so` 与 `libcudnn.so`，未匹配时输出诊断而不抛异常，并校验 CUDA Runtime API 返回码。

## 优化项

### O-M1. 使深度示例可导入、可诊断且可靠释放

- **文件**：`convert_trt.py`
- **现状**：导入模块即加载模型、推理和写文件；图片读取失败会跳过释放；输入隐式转为 `float64`。
- **执行结果**：迁入 `main()`，提供路径参数和显式 `--show`；校验模型/图片/输出契约；以 `try/finally` 释放资源；预处理固定为连续 `float32` NCHW 数据。

## 工程规范

### E-L1. 同步文档、忽略规则和回归测试

- **文件**：`README.md`、`.gitignore`、`tests/test_helpers.py`
- **现状**：README 把返回值写成单个数组，目录忽略规则会匹配任意嵌套路径，辅助逻辑无回归测试。
- **执行结果**：文档说明输出列表和模型契约；`build`、`model`、`images` 规则限定为项目根目录；新增 6 个无 GPU 测试。

## 剩余验证边界

- 当前 Python 环境未安装 `tensorrt`、`pycuda` 或 `torch`，因此未运行真实 engine 构建与 GPU 推理；代码已通过语法检查和无 GPU 测试。
- engine 缓存键包含 ONNX 内容和 TensorRT 版本，但不含 GPU 架构。缓存无法加载时会自动重建；换 GPU 后仍应首次运行时确认 engine 可用。
