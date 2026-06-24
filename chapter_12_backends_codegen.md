# 第 12 章：GPU 与 CPU 代码生成后端

## 本章目标

本章解释 Scheduler 之后的代码生成：GPU 后端如何生成 Triton 代码，CPU 后端如何生成 C++ 代码，wrapper 在其中扮演什么角色。

本章承接第 9 章的最后一步：

```text
Scheduler._codegen
  -> backend.codegen_node(node)
  -> backend.codegen_template(...)
  -> codegen_extern_call(...)
  -> wrapper_code.generate(...)
  -> PyCodeCache.load_by_key_path(...)
```

如果说第 8 章生成 IR，第 9 章决定怎么排，第 10/11 章解释不同 op 如何进入 IR，那么本章回答：这些调度后的节点如何变成可执行代码。

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

源码定位：

```text
/usr/local/lib/python3.11/site-packages/torch/_inductor/codegen/wrapper.py
  class PythonWrapperCodegen: 约 652 行
  write_header: 约 760 行
  generate: 约 1195 行
  codegen_allocation: 约 2347 行
  generate_kernel_call: 约 2127 行
```

wrapper 生成的代码通常会包含：

```text
from torch._inductor.async_compile import AsyncCompile
async_compile = AsyncCompile()
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu / empty_strided_cuda
reinterpret_tensor
extern_kernels
```

这说明 wrapper 不只是“调用 kernel”。它还负责把 Inductor 编译期的假设转换成运行时代码。

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

源码定位：

```text
/usr/local/lib/python3.11/site-packages/torch/_inductor/codegen/triton.py
  class TritonKernel: 约 1565 行
  class TritonScheduling: 约 4021 行
```

`TritonScheduling` 继承 SIMD 调度基类，负责把 Scheduler node 转换成 `TritonKernel`。`TritonKernel` 再把 loop body、indexing、mask、load/store、reduction 等打印成 Triton 语言。

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

源码定位：

```text
/usr/local/lib/python3.11/site-packages/torch/_inductor/codegen/cpp.py
  class CppScheduling: 约 4368 行
```

CPU 后端同样走 Scheduler 分派，只是 `get_backend(device)` 得到的是 C++ scheduling backend。它会生成 C++ loop、C++ template 调用或 extern/fallback 相关调用。

### AsyncCompile

`torch/_inductor/async_compile.py` 中的 `AsyncCompile` 负责异步编译 Triton、C++、CUDA 等代码，减少阻塞并管理编译任务。

源码定位：

```text
/usr/local/lib/python3.11/site-packages/torch/_inductor/async_compile.py
  class AsyncCompile: 约 191 行
  triton: 约 254 行
  cpp: 约 351 行
  cuda: 约 369 行
```

Triton 路径尤其复杂，因为编译开销高，源码里会使用 `CompiledTritonKernels` cache、进程池、`CachingAutotuner` 等机制。读者第一遍只要记住：wrapper 里出现的 `async_compile.triton(...)` 最终会走到这里。

### CodeCache

`torch/_inductor/codecache.py` 提供：

```text
PyCodeCache
CppCodeCache
CUDACodeCache
```

生成代码最终会通过 code cache 写入、编译、加载和复用。

`GraphLowering._compile_to_module` 会：

```text
wrapper_code, _ = self.codegen()
PyCodeCache.write(wrapper_code.value)
PyCodeCache.load_by_key_path(...)
return loaded_module
```

所以 `TORCH_LOGS=output_code` 中看到的 generated Python 文件，正是 code cache 写出的 wrapper module。

## 源码调用链解读

### 1. 从 `GraphLowering.codegen` 到 wrapper

第 9 章看到：

```text
GraphLowering.codegen
  -> self.init_wrapper_code()
  -> self._update_scheduler()
  -> self.wrapper_code.push_codegened_graph(self)
  -> self.scheduler.codegen()
  -> self.wrapper_code.generate(self.is_inference)
```

这说明 wrapper 是在 scheduler codegen 过程中逐步被填充的。kernel 定义、kernel 调用、buffer 分配等代码，不是最后一次性凭空生成，而是 Scheduler 遍历节点时不断写入 wrapper 的不同 buffer。

### 2. Scheduler 对节点类型分派到不同 codegen

第 9 章的 `_codegen` 可以和本章对应起来：

```text
node.is_template()
  -> backend.codegen_template(...)
  -> matmul/conv/flex attention 等模板路径

node.is_extern()
  -> codegen_extern_call(...)
  -> ATen/external library/fallback 路径

SchedulerNode / FusedSchedulerNode
  -> backend.codegen_node(node)
  -> pointwise/reduction/fused loop 路径
```

因此读 generated code 时，要先判断某段代码属于哪类节点。不要看到不是 Triton kernel 就以为没编译；它可能是 extern/template 或 C++ 路径。

### 3. GPU pointwise 的典型路径

```text
FusedSchedulerNode
  -> TritonScheduling.codegen_node
  -> SIMDScheduling 生成 node schedule
  -> TritonKernel
  -> codegen_body / codegen_kernel
  -> wrapper 中注册 async_compile.triton(...)
  -> 运行时 kernel call
```

关键对象：

```text
TritonScheduling:
  负责把调度节点变成 Triton kernel 生成任务。

TritonKernel:
  负责具体的 Triton 代码文本，包括 program id、mask、load/store。

AsyncCompile:
  负责编译 Triton 源码并缓存编译结果。
```

### 4. CPU pointwise 的典型路径

```text
FusedSchedulerNode
  -> CppScheduling.codegen_node
  -> C++ loop / vectorized code
  -> wrapper 调用 async_compile.cpp(...)
  -> CppCodeCache 编译加载
```

CPU 代码生成更关心：

- SIMD ISA。
- 线程并行。
- 函数参数数量限制。
- C++ 编译器和 ABI。
- Python binding 或 C++ wrapper。

### 5. `compile_to_module` 是最后的装载点

在第 7 章 `_InProcessFxCompile` 中，最终执行：

```text
compiled_fn = graph.compile_to_module().call
```

而 `compile_to_module` 内部会生成 wrapper code，再通过 `PyCodeCache` 写文件并加载模块。这个 `.call` 就是 Dynamo 后续缓存和调用的 compiled callable。

从调用链看：

```text
GraphLowering.compile_to_module
  -> GraphLowering.codegen
  -> Scheduler.codegen
  -> PythonWrapperCodegen.generate
  -> PyCodeCache.write
  -> PyCodeCache.load_by_key_path
  -> module.call
```

这也是为什么查看 generated code 对调试极其重要：它就是 compiled function 的真实运行入口。

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

更贴近源码：

```text
GraphLowering._compile_to_module
  -> self.codegen()
     -> init_wrapper_code
     -> Scheduler(self.operations)
     -> scheduler.codegen()
        -> backend.codegen_node / codegen_template / codegen_extern_call
     -> wrapper_code.generate()
  -> PyCodeCache.write(wrapper_code.value)
  -> PyCodeCache.load_by_key_path(...)
  -> mod.call
```

在 `graph.py` 中，`compile_to_module` 会调用 `codegen()` 或 `codegen_with_cpp_wrapper()`，然后通过 `PyCodeCache.load` 加载生成模块。

## TorchInductor 内部大致发生了什么

对于 GPU pointwise：

1. Scheduler 创建 fused node。
2. `TritonScheduling` 生成 `TritonKernel`。
3. kernel body 根据 IR loop body 打印成 Triton 语言。
4. wrapper 生成 `async_compile.triton(...)` 或相关加载逻辑。
5. runtime 调用 kernel。

如果你在 output code 中看到：

```text
async_compile = AsyncCompile()
...
triton_poi_fused_...
...
async_compile.wait(globals())
```

通常说明 wrapper 中包含 Triton kernel 的异步编译和等待逻辑。实际格式会随版本变化。

对于 CPU pointwise：

1. Scheduler 创建 fused node。
2. `CppScheduling` 生成 C++ loop。
3. C++ code cache 编译成可加载扩展或绑定。
4. wrapper 调用 C++ kernel。

如果你在 output code 中看到 C++ source 片段、`async_compile.cpp`、`CppCodeCache` 相关路径，说明走的是 CPU codegen 或 C++ helper/template。

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

和前面章节的连接：

```text
第 7 章:
  compile_fx.py 最终调用 graph.compile_to_module().call

第 8 章:
  GraphLowering 生成 operations

第 9 章:
  Scheduler 融合并分派到 backend

第 10 章:
  pointwise/reduction lowering 决定普通 fused kernel 的 IR

第 11 章:
  matmul/conv template 和 extern choice 进入 Scheduler

本章:
  backend + wrapper + async_compile + codecache 生成并加载可执行代码
```

## 阅读 generated code 的顺序

打开 `TORCH_LOGS=output_code` 后，不建议从第一行逐行读。可以按这个顺序：

1. 找 `call(args)` 函数，看运行时入口。
2. 找输入 assert，例如 `assert_size_stride`，看 shape/stride 假设。
3. 找 buffer 分配，例如 `empty_strided_cuda` 或 `empty_strided_cpu`。
4. 找 kernel 调用顺序。
5. 找 kernel 定义或 async compile 注册。
6. 对照 debug graph 看 kernel 来自哪些 FX/IR 节点。

这样能把源码的 `Scheduler.codegen` 和实际输出代码对应起来。

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
