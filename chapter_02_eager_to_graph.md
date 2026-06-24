# 第 2 章：从 Eager Mode 到 Graph Mode

![从 eager 到编译链路](./assets/compile_chain.svg)

## 本章目标

本章解释为什么 Inductor 的故事必须从“图”开始。TorchInductor 本身并不直接理解任意 Python 程序，它理解的是前面组件整理后的图。读完本章，你应该能把下面三件事区分开：

- eager mode：逐行执行 Python 和 Tensor 操作。
- graph mode：把一段 Tensor 计算表示为节点和边。
- compiler backend：基于图做优化和代码生成。

## 背景知识

PyTorch eager mode 的核心优势是即时性：

```python
import torch

x = torch.randn(4)
y = x + 1
z = torch.relu(y)
print(z)
```

这段代码中，`x + 1` 和 `torch.relu(y)` 都立即运行。对研究者来说，这很自然；对编译器来说，这却意味着每个操作都像“单独来敲门”，编译器很难提前知道下一个操作是什么。

图模式换了一种看法：先不急着执行每一步，而是把操作记录下来：

```text
x -> add -> relu -> output
```

有了整段计算，优化器才有机会跨节点做事情，比如融合、删除冗余、选择布局、规划内存。

## 核心概念

### 节点和边

计算图里，节点通常表示输入、输出或操作；边表示数据依赖。

```text
placeholder x
    |
 call_function aten.add
    |
 call_function aten.relu
    |
 output
```

对 Inductor 来说，图不是画给人看的流程图，而是一种可遍历、可分析、可变换的数据结构。

### 图优化为什么有用

考虑：

```python
def f(x, y):
    return torch.relu(x + y) * 2
```

如果只看 `x + y`，它只是一次加法。如果看到完整图，编译器会发现 `add -> relu -> mul` 都是逐元素操作，可以尝试生成一个 fused kernel。

![fusion 减少中间内存访问](./assets/fusion_memory.svg)

图优化常见目标：

- 减少 kernel launch。
- 减少中间 Tensor 写回和读出。
- 把形状、stride、dtype 等信息传给代码生成器。
- 把复杂算子拆解成后端更容易处理的原语。
- 对矩阵乘法、卷积、attention 等模式选择专门实现。

### PyTorch 的难点

PyTorch 程序不是简单的静态图语言。用户可以写：

```python
def f(x):
    if x.sum() > 0:
        return x * 2
    return x - 2
```

这里的分支依赖 Tensor 运行时值。对于编译器来说，这类 Python 控制流很难一次性捕获成纯静态图。PyTorch 2.x 的策略不是要求用户放弃 Python，而是由 TorchDynamo 尽可能捕获可编译片段，不能捕获的地方发生 graph break 或回退。

## 一个最小 PyTorch 示例

```python
import torch
from torch import fx

class M(torch.nn.Module):
    def forward(self, x):
        a = x + 1
        b = torch.relu(a)
        return b * 2

m = M()
gm = fx.symbolic_trace(m)
print(gm.graph)
gm.graph.print_tabular()
```

这不是 `torch.compile` 的真实捕获路径，但它适合入门观察 FX Graph。你会看到类似：

```text
placeholder x
call_function operator.add
call_function torch.relu
call_function operator.mul
output
```

## 编译前后发生了什么

在 eager 下，每个操作立即执行；在 graph mode 下，系统先获得一段计算的结构。注意“获得图”和“执行图”是两件事：

```text
捕获图：把 Python/Tensor 操作变成 GraphModule
优化图：对图做 rewrite、decomposition、fusion 等
执行图：运行生成的 Python/C++/Triton 代码
```

Inductor 主要参与后两步，但它拿到的输入来自前面的图捕获和 autograd 处理。

## TorchInductor 内部大致发生了什么

Inductor 不负责解释 Python bytecode，也不负责把任意 Python 控制流变成图。它接收 FX `GraphModule` 和 `example_inputs`。这些输入提供两类信息：

- 图结构：有哪些算子，依赖关系是什么。
- 示例输入：shape、dtype、device、stride、是否需要梯度等。

随后 Inductor 会进入 `compile_fx.py`，并在更深处用 `GraphLowering` 解释 FX 节点，将它们降低成 Inductor IR。

## 关键源码入口

本环境 PyTorch 版本：`2.7.1`。

建议阅读：

```text
/usr/local/lib/python3.11/site-packages/torch/fx/graph.py
/usr/local/lib/python3.11/site-packages/torch/fx/node.py
/usr/local/lib/python3.11/site-packages/torch/fx/graph_module.py
/usr/local/lib/python3.11/site-packages/torch/_inductor/compile_fx.py
```

FX 侧重点：

- `Graph` 管理节点列表。
- `Node` 表示单个操作。
- `GraphModule` 把图包装成可调用的 `nn.Module` 风格对象。

## 常见误区

### FX Graph 不等于完整 Python AST

FX Graph 表示的是被捕获到的 Tensor 计算，不是 Python 源代码的完整语法树。

### 图不是越大越好

大图给优化更多上下文，但也会提高编译成本。`fullgraph=True` 对调试很有用，但不一定适合所有真实程序。

### graph break 不总是错误

默认 `fullgraph=False` 时，Dynamo 可以把函数拆成多个可编译区域，中间穿插 eager 执行。它可能影响性能，但并不一定导致程序无法运行。

## 小结

Inductor 是图编译器后端。要理解它，先要接受一个基本事实：它优化的对象不是你肉眼看到的 Python 源码，而是前面系统生成的 FX 图及其元信息。下一章会进入真实入口：`torch.compile` 如何把用户函数交给 Dynamo 和 Inductor。

## 思考题或练习

1. 用 `fx.symbolic_trace` 追踪一个只包含 Tensor 运算的 `nn.Module`。
2. 在 `forward` 中加入 `print(x)`，观察 FX 追踪是否还能按预期工作。
3. 思考 `x.view(...)`、`x.transpose(...)` 这类 view 操作在图里为什么很重要。

## 本章需要人工核查的技术点

- `fx.symbolic_trace` 的输出不等同于 Dynamo 捕获的图，正文后续应继续区分。
- 不同 PyTorch 版本中 FX `Node` 的元信息字段可能变化。

