# 写作方案

## 书名

推荐书名：

**《走进 TorchInductor：从 `torch.compile` 到深度学习编译器》**

备选书名：

1. 《TorchInductor 入门与实现：理解 PyTorch 2.x 编译加速》
2. 《从 Eager 到 Compiler：TorchInductor 技术导读》
3. 《PyTorch 编译器之路：TorchDynamo、FX、AOTAutograd 与 TorchInductor》

## 目标读者画像

本书面向已经会使用 PyTorch，但还没有系统理解 PyTorch 编译栈的读者。

读者大致具备：

- 熟悉 Python 基础语法。
- 会写基本 PyTorch 训练或推理代码。
- 理解 Tensor、Module、autograd、GPU 加速等基本概念。
- 可能听说过 `torch.compile`，但不知道它内部如何工作。
- 可能不了解编译器、IR、图优化、kernel fusion、Triton、CUDA kernel 生成。
- 希望通过源码、例子和调试方法，真正理解 TorchInductor 的核心机制。

本书不假设读者有编译器背景。每个底层概念都应先用直觉解释，再逐步过渡到 PyTorch 内部实现。

## 全书主线

```text
普通 PyTorch eager 程序
        ↓
torch.compile
        ↓
TorchDynamo 捕获 Python 程序
        ↓
FX Graph 表示计算
        ↓
AOTAutograd 拆分前向/反向
        ↓
TorchInductor 接收图
        ↓
图级优化、调度、融合、布局处理
        ↓
生成 Triton / C++ / Python wrapper
        ↓
编译、缓存、执行
        ↓
调试正确性和性能问题
```

## 初版目录

### 第 1 章：为什么 PyTorch 需要 TorchInductor？

核心问题：

- PyTorch eager mode 好在哪里？慢在哪里？
- 深度学习框架为什么需要编译器？
- `torch.compile` 解决的到底是什么问题？
- TorchInductor 在 PyTorch 2.x 中扮演什么角色？

建议代码例子：

```python
import torch

def f(x, y):
    return torch.relu(x + y) * 2

x = torch.randn(1024, 1024, device="cuda")
y = torch.randn(1024, 1024, device="cuda")

compiled_f = torch.compile(f)

print(f(x, y))
print(compiled_f(x, y))
```

推荐源码路径：

- `torch/__init__.py`
- `torch/_dynamo/`
- `torch/_inductor/`
- `torch/_inductor/compile_fx.py`

### 第 2 章：从 Eager Mode 到 Graph Mode

核心问题：

- eager mode 是如何逐行执行 PyTorch 程序的？
- 什么是“图”？
- 为什么图表示更适合优化？
- PyTorch 程序为什么难以直接变成静态图？

建议代码例子：

```python
import torch

def f(x):
    a = x + 1
    b = torch.relu(a)
    c = b * 2
    return c
```

推荐源码路径：

- `torch/fx/`
- `torch/fx/graph.py`
- `torch/fx/graph_module.py`
- `torch/fx/proxy.py`

### 第 3 章：`torch.compile` 的整体执行链路

核心问题：

- `torch.compile` 是一个编译器吗？
- 它和 TorchDynamo、AOTAutograd、TorchInductor 如何协作？
- 一次 compiled function 调用大致经历哪些阶段？
- graph break 是什么？

建议代码例子：

```python
import torch

def f(x):
    if x.sum() > 0:
        return x * 2
    else:
        return x - 2

compiled_f = torch.compile(f)
x = torch.randn(4)
print(compiled_f(x))
```

推荐源码路径：

- `torch/_dynamo/`
- `torch/_dynamo/eval_frame.py`
- `torch/_dynamo/convert_frame.py`
- `torch/_dynamo/output_graph.py`
- `torch/_inductor/compile_fx.py`

### 第 4 章：TorchDynamo：捕获 Python 程序

核心问题：

- TorchDynamo 负责什么？
- 它为什么从 Python frame / bytecode 层工作？
- 什么是 guard？
- 为什么输入 shape、dtype、device 会影响重新编译？

建议代码例子：

```python
import torch
import torch._dynamo as dynamo

def f(x, y):
    return torch.sin(x + y)

explanation = dynamo.explain(f)(torch.randn(8), torch.randn(8))
print(explanation)
```

推荐源码路径：

- `torch/_dynamo/eval_frame.py`
- `torch/_dynamo/convert_frame.py`
- `torch/_dynamo/guards.py`
- `torch/_dynamo/variables/`
- `torch/_dynamo/symbolic_convert.py`

### 第 5 章：FX Graph：PyTorch 计算的中间表示

核心问题：

- FX Graph 是什么？
- Graph、Node、GraphModule 分别是什么？
- TorchInductor 为什么不直接处理 Python 源码？
- 如何打印和理解 FX Graph？

建议代码例子：

```python
import torch
from torch import fx

class M(torch.nn.Module):
    def forward(self, x):
        return torch.relu(x + 1)

m = M()
gm = fx.symbolic_trace(m)
print(gm.graph)
gm.graph.print_tabular()
```

推荐源码路径：

- `torch/fx/graph.py`
- `torch/fx/node.py`
- `torch/fx/graph_module.py`
- `torch/fx/interpreter.py`
- `torch/fx/passes/`

### 第 6 章：AOTAutograd：把训练图拆开

核心问题：

- 推理和训练的编译有什么不同？
- autograd 为什么让编译更复杂？
- AOTAutograd 做了什么？
- 前向图、反向图、joint graph 是什么？

建议代码例子：

```python
import torch

def loss_fn(x, w):
    y = x @ w
    return y.relu().sum()

x = torch.randn(16, 32, requires_grad=True)
w = torch.randn(32, 64, requires_grad=True)

compiled_loss = torch.compile(loss_fn)
loss = compiled_loss(x, w)
loss.backward()
```

推荐源码路径：

- `torch/_functorch/aot_autograd.py`
- `torch/_functorch/partitioners.py`
- `torch/_dynamo/backends/`
- `torch/_inductor/compile_fx.py`

### 第 7 章：TorchInductor 的入口：从 FX Graph 到可执行代码

核心问题：

- TorchInductor 接收到的输入是什么？
- 它如何从 FX Graph 开始编译？
- 编译过程有哪些主要阶段？
- generated code 通常长什么样？

建议代码例子：

```python
import torch

def f(x, y):
    return torch.relu(x + y)

compiled_f = torch.compile(f, backend="inductor")

x = torch.randn(128, 128, device="cuda")
y = torch.randn(128, 128, device="cuda")
compiled_f(x, y)
```

推荐源码路径：

- `torch/_inductor/compile_fx.py`
- `torch/_inductor/graph.py`
- `torch/_inductor/scheduler.py`
- `torch/_inductor/codegen/`
- `torch/_inductor/config.py`

### 第 8 章：IR、Buffer 与 Scheduler：TorchInductor 如何理解计算

核心问题：

- TorchInductor 内部 IR 是什么？
- Tensor 计算如何变成 buffer、loop 和 schedule？
- Scheduler 的职责是什么？
- 为什么调度决定 fusion 和代码生成效果？

建议代码例子：

```python
import torch

def f(x):
    a = x + 1
    b = torch.relu(a)
    c = b * 2
    return c
```

推荐源码路径：

- `torch/_inductor/ir.py`
- `torch/_inductor/scheduler.py`
- `torch/_inductor/dependencies.py`
- `torch/_inductor/lowering.py`
- `torch/_inductor/select_algorithm.py`

### 第 9 章：Pointwise、Reduction 与 Fusion

核心问题：

- 什么是 pointwise 操作？
- 什么是 reduction 操作？
- 为什么 fusion 可以减少显存读写？
- 哪些操作容易融合，哪些不容易？
- fusion 的限制来自哪里？

建议代码例子：

```python
def f(x, y):
    return torch.sigmoid(x + y) * 3

def g(x):
    return torch.sum(torch.relu(x), dim=1)

def h(x):
    return torch.softmax(x, dim=-1)
```

推荐源码路径：

- `torch/_inductor/lowering.py`
- `torch/_inductor/scheduler.py`
- `torch/_inductor/ir.py`
- `torch/_inductor/codegen/triton.py`
- `torch/_inductor/codegen/cpp.py`

### 第 10 章：矩阵乘法、外部库与算法选择

核心问题：

- matmul 为什么通常不简单手写 pointwise kernel？
- TorchInductor 如何处理 `mm`、`bmm`、`addmm`？
- 什么时候调用外部库，什么时候生成 Triton kernel？
- autotuning 是什么？

建议代码例子：

```python
import torch

def f(x, w, b):
    return torch.relu(x @ w + b)

x = torch.randn(128, 256, device="cuda")
w = torch.randn(256, 512, device="cuda")
b = torch.randn(512, device="cuda")

compiled_f = torch.compile(f)
compiled_f(x, w, b)
```

推荐源码路径：

- `torch/_inductor/kernel/`
- `torch/_inductor/select_algorithm.py`
- `torch/_inductor/codegen/triton.py`
- `torch/_inductor/codegen/cpp.py`
- `torch/_inductor/lowering.py`

### 第 11 章：Layout、Stride 与 Memory Planning

核心问题：

- Tensor layout 为什么重要？
- contiguous、stride、view、transpose 会给编译带来什么问题？
- TorchInductor 如何处理中间 buffer？
- memory planning 的目标是什么？
- 为什么有时编译后反而出现额外 copy？

建议代码例子：

```python
import torch

def f(x):
    y = x.transpose(0, 1)
    z = y.contiguous()
    return z * 2

def g(x):
    return x[:, ::2] + 1
```

推荐源码路径：

- `torch/_inductor/ir.py`
- `torch/_inductor/lowering.py`
- `torch/_inductor/scheduler.py`
- `torch/_inductor/memory.py`
- `torch/_inductor/codegen/`

### 第 12 章：GPU 后端：Triton 代码生成入门

核心问题：

- Triton 是什么？
- 为什么 TorchInductor 常用 Triton 生成 GPU kernel？
- 一个 Triton kernel 大致长什么样？
- TorchInductor 生成的 Triton 代码如何阅读？
- block size、num warps、mask、program id 是什么？

建议代码例子：

```python
import torch

def f(x, y):
    return torch.sin(x) + torch.cos(y)

x = torch.randn(1024, device="cuda")
y = torch.randn(1024, device="cuda")

compiled_f = torch.compile(f)
compiled_f(x, y)
```

推荐源码路径：

- `torch/_inductor/codegen/triton.py`
- `torch/_inductor/runtime/triton_heuristics.py`
- `torch/_inductor/runtime/triton_helpers.py`
- Triton 官方文档和 `triton.language`

### 第 13 章：CPU 后端、调试方法与源码阅读路线

核心问题：

- TorchInductor 的 CPU 后端大致如何工作？
- 如何查看生成代码？
- 如何定位 graph break？
- 如何分析重新编译？
- 如何调试性能问题？
- 初学者应该如何阅读 TorchInductor 源码？

建议代码例子：

```python
import torch

def f(x):
    return torch.relu(x + 1) * 2

x = torch.randn(1024, 1024)
compiled_f = torch.compile(f)
compiled_f(x)
```

推荐源码路径：

- `torch/_inductor/codegen/cpp.py`
- `torch/_inductor/cpp_builder.py`
- `torch/_inductor/runtime/`
- `torch/_inductor/config.py`
- `torch/_dynamo/config.py`
- `torch/_logging/`

## 附录规划

### 附录 A：常用调试环境变量与日志选项

候选内容：

- `TORCH_LOGS`
- `TORCH_COMPILE_DEBUG`
- `TORCHDYNAMO_VERBOSE`
- `torch._dynamo.explain`
- `torch._dynamo.reset`
- generated code 目录
- cache 目录

这些选项随 PyTorch 版本可能变化，正式写正文前要核查本地源码和官方文档。

### 附录 B：术语表

建议包含：

- eager mode
- graph
- FX
- guard
- graph break
- IR
- lowering
- fusion
- scheduler
- buffer
- stride
- layout
- reduction
- autotuning
- Triton kernel
- CUDA kernel
- dynamic shape

### 附录 C：源码阅读地图

```text
torch.compile 用户入口
↓
torch/_dynamo/
↓
torch/fx/
↓
torch/_functorch/aot_autograd.py
↓
torch/_inductor/compile_fx.py
↓
torch/_inductor/graph.py
↓
torch/_inductor/ir.py
↓
torch/_inductor/lowering.py
↓
torch/_inductor/scheduler.py
↓
torch/_inductor/codegen/
↓
torch/_inductor/runtime/
```

## 写作顺序建议

1. 先让读者知道为什么需要编译。
2. 再解释 `torch.compile` 是入口，不是全部。
3. 接着讲 Dynamo 捕获程序，FX 表示图。
4. 然后讲 AOTAutograd 处理训练。
5. 再进入 TorchInductor 主体。
6. 最后讲具体优化、后端和调试。

## 需要版本核查的重点

正式写正文前，建议核查当前安装的 PyTorch 版本和源码，因为以下内容变化较快：

- `torch.compile` 参数和默认 backend 行为。
- `TORCH_LOGS` 支持的日志项。
- `TORCH_COMPILE_DEBUG` 生成目录结构。
- TorchInductor 内部文件路径。
- Triton codegen 相关类名和函数名。
- CPU 后端是否使用 C++、ATen、OpenMP、向量化或外部库路径。
- matmul、conv、attention 等算子的 lowering 和选择策略。
- dynamic shape 支持范围。
- cache、autotune、benchmark 相关配置。

