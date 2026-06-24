# 第 12 章：GPU 与 CPU 代码生成后端

## 本章目标

本章解释 Scheduler 之后的代码生成：GPU 后端如何生成 Triton 代码，CPU 后端如何生成 C++ 代码，wrapper 在其中扮演什么角色。

## 背景知识

Inductor 的输出不只是一个 kernel。真实输出通常包括：

- Python wrapper 或 C++ wrapper。
- 输入检查。
- device guard。
- buffer 分配和释放。
- kernel 定义。
- 异步编译对象。
- kernel 调用。
- fallback 或 extern call。
- 返回结果。

## 核心概念

### Python wrapper

本环境中：

```text
torch/_inductor/codegen/wrapper.py
```

定义 `PythonWrapperCodegen`。它负责生成外层 Python 代码，例如：

- import 运行时依赖。
- 创建 `AsyncCompile`。
- 分配 buffer。
- 调用 kernel。
- 返回输出。

### GPU: Triton

GPU 侧核心文件：

```text
torch/_inductor/codegen/triton.py
```

重要类包括：

```text
TritonKernel
TritonScheduling
```

Triton kernel 常见结构：

```python
@triton.jit
def kernel(..., BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    ...
    tl.store(out_ptr + offsets, result, mask=mask)
```

真实生成代码会更复杂，包含 stride、mask、constexpr、heuristics、num_warps、dynamic shapes 等。

### CPU: C++

CPU 侧核心文件：

```text
torch/_inductor/codegen/cpp.py
```

重要类包括：

```text
CppScheduling
```

CPU 还涉及：

```text
torch/_inductor/cpp_builder.py
torch/_inductor/cpu_vec_isa.py
torch/_inductor/codegen/cpp_wrapper_cpu.py
torch/_inductor/codegen/cpp_gemm_template.py
```

CPU codegen 关心：

- OpenMP 或线程并行。
- SIMD 向量化。
- cache locality。
- dtype。
- GEMM template。
- fallback 到 ATen 或外部库。

### AsyncCompile

`torch/_inductor/async_compile.py` 中的 `AsyncCompile` 负责异步编译 Triton、C++、CUDA 等代码，减少阻塞并管理编译任务。

### CodeCache

`torch/_inductor/codecache.py` 提供：

```text
PyCodeCache
CppCodeCache
CUDACodeCache
```

生成代码最终会通过 code cache 写入、编译、加载和复用。

## 一个最小 PyTorch 示例

```python
import torch

def f(x, y):
    return torch.sin(x) + torch.cos(y)

device = "cuda" if torch.cuda.is_available() else "cpu"
x = torch.randn(1024, device=device)
y = torch.randn(1024, device=device)

compiled_f = torch.compile(f)
compiled_f(x, y)
```

查看输出代码：

```bash
TORCH_LOGS=output_code python backend_example.py
```

如果使用 GPU，通常会看到 Triton 相关代码；如果使用 CPU，通常会看到 C++ 或相关 wrapper/fallback 路径。

## 编译前后发生了什么

代码生成主线：

```text
GraphLowering.codegen
  -> init_wrapper_code
  -> scheduler.codegen
     -> TritonScheduling / CppScheduling / other backend
  -> wrapper_code.generate
  -> PyCodeCache.load
  -> compiled module callable
```

在 `graph.py` 中，`compile_to_module` 会调用 `codegen()` 或 `codegen_with_cpp_wrapper()`，然后通过 `PyCodeCache.load` 加载生成模块。

## TorchInductor 内部大致发生了什么

对于 GPU pointwise：

1. Scheduler 创建 fused node。
2. `TritonScheduling` 生成 `TritonKernel`。
3. kernel body 根据 IR loop body 打印成 Triton 语言。
4. wrapper 生成 `async_compile.triton(...)` 或相关加载逻辑。
5. runtime 调用 kernel。

对于 CPU pointwise：

1. Scheduler 创建 fused node。
2. `CppScheduling` 生成 C++ loop。
3. C++ code cache 编译成可加载扩展或绑定。
4. wrapper 调用 C++ kernel。

## 关键源码入口

```text
/usr/local/lib/python3.11/site-packages/torch/_inductor/codegen/wrapper.py
/usr/local/lib/python3.11/site-packages/torch/_inductor/codegen/triton.py
/usr/local/lib/python3.11/site-packages/torch/_inductor/codegen/cpp.py
/usr/local/lib/python3.11/site-packages/torch/_inductor/codegen/simd.py
/usr/local/lib/python3.11/site-packages/torch/_inductor/async_compile.py
/usr/local/lib/python3.11/site-packages/torch/_inductor/codecache.py
/usr/local/lib/python3.11/site-packages/torch/_inductor/runtime/triton_heuristics.py
```

建议搜索：

```bash
rg -n "class TritonKernel|class TritonScheduling" torch/_inductor/codegen/triton.py
rg -n "class CppScheduling" torch/_inductor/codegen/cpp.py
rg -n "class PythonWrapperCodegen" torch/_inductor/codegen/wrapper.py
```

## 常见误区

### Inductor 只生成 Triton

不是。Triton 是 GPU 上的重要路径，但 CPU、C++ wrapper、extern kernel、fallback、模板系统都属于 Inductor 工作范围。

### wrapper 是无关紧要的胶水

wrapper 负责 buffer 生命周期、输入检查、device guard 和 kernel 调用，是正确性和性能都很重要的一层。

### 生成代码越短越好

不一定。高性能代码常需要额外的 specialization、heuristics 和 runtime 包装。

## 小结

代码生成是 Inductor 从 IR 走向真实执行的最后阶段。GPU 侧常见目标是 Triton，CPU 侧常见目标是 C++，二者都由 wrapper 和 code cache 串起来。下一章整理缓存、运行时、调试和源码阅读路线。

## 思考题或练习

1. 在 CPU 和 CUDA 上分别运行同一个 pointwise 函数，比较生成代码。
2. 搜索 `AsyncCompile` 的 `triton` 和 `cpp` 方法，理解异步编译如何分发。
3. 查看 `codecache.py` 中 `PyCodeCache.load`，理解生成 Python 模块如何被加载。

## 本章需要人工核查的技术点

- CPU codegen 是否使用 OpenMP、向量化或外部库依赖平台和配置。
- GPU 代码可能因 CUDA、ROCm、Triton 版本不同而变化。

