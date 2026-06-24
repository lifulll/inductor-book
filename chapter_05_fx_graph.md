# 第 5 章：FX Graph：Inductor 接收的计算语言

## 本章目标

本章介绍 Inductor 接收的核心输入：`torch.fx.GraphModule`。读完后你应该能看懂一张简单 FX 图，并知道哪些字段会影响 Inductor 后续 lowering。

## 背景知识

FX Graph 是 PyTorch 内部常用的图表示。它由三类对象组成：

- `Graph`：节点容器。
- `Node`：单个输入、操作或输出。
- `GraphModule`：把 `Graph` 包成可调用模块。

Inductor 的 `compile_fx(gm, example_inputs)` 中，`gm` 就是 `GraphModule`。

## 核心概念

### Node 的常见 op 类型

常见 `node.op`：

- `placeholder`：函数输入。
- `call_function`：调用函数或 ATen op。
- `call_method`：调用对象方法。
- `call_module`：调用子模块。
- `get_attr`：读取模块属性。
- `output`：函数输出。

Dynamo/AOTAutograd 交给 Inductor 的图通常更接近 ATen/prim 级别，而不是用户写的高层 Python 结构。

### `node.target`

`target` 表示具体调用目标。例如：

```text
aten.add.Tensor
aten.relu.default
aten.mm.default
```

Inductor 的 lowering registry 会根据 target 找到对应 lowering。

### `node.meta`

`node.meta` 存放元信息。对编译器很重要的元信息包括：

- fake tensor 或 example value。
- tensor shape。
- dtype。
- device。
- stride。
- traceback/provenance 信息。

这些信息可能由 Dynamo、FakeTensor、AOTAutograd、shape propagation 等阶段填入。

## 一个最小 PyTorch 示例

```python
import torch
from torch import fx

class M(torch.nn.Module):
    def forward(self, x, y):
        return torch.relu(x + y)

gm = fx.symbolic_trace(M())
print(gm.graph)
gm.graph.print_tabular()
```

输出中重点看：

```text
placeholder x
placeholder y
call_function add
call_function relu
output
```

真实 `torch.compile` 中的图可能使用 `torch.ops.aten.*` 形式。不要把 `fx.symbolic_trace` 的教学输出完全等同于 Dynamo 输出。

## 编译前后发生了什么

FX Graph 进入 Inductor 前，可能已经经历：

- Dynamo 捕获。
- decomposition，把复杂 op 拆成更基础 op。
- AOTAutograd 前后向拆分。
- functionalization，把 mutation/view 相关行为改写成更利于编译的形式。
- fake tensor propagation，填充 shape/device/dtype 信息。

## TorchInductor 内部大致发生了什么

在 Inductor 侧，`GraphLowering` 继承自 `torch.fx.Interpreter`。这意味着它会像解释器一样遍历 FX 节点：

```text
placeholder -> call_function -> call_function -> output
```

但它不是为了立即算出 Tensor 值，而是为了把节点转换成 Inductor IR，例如 `TensorBox`、`ComputedBuffer`、`Pointwise`、`Reduction` 等。

![lowering 到 scheduler](./assets/lowering_scheduler.svg)

## 关键源码入口

```text
/usr/local/lib/python3.11/site-packages/torch/fx/graph.py
/usr/local/lib/python3.11/site-packages/torch/fx/node.py
/usr/local/lib/python3.11/site-packages/torch/fx/graph_module.py
/usr/local/lib/python3.11/site-packages/torch/fx/interpreter.py
/usr/local/lib/python3.11/site-packages/torch/_inductor/graph.py
```

重点关注 `torch/_inductor/graph.py`：

- `class GraphLowering(torch.fx.Interpreter)`
- `run`
- `call_function`
- `output`
- `codegen`
- `compile_to_module`

## 常见误区

### FX 图越接近用户代码越好

对教学来说是；对编译器来说不一定。Inductor 更喜欢规范化、低层、可分析的图。

### 所有 FX 节点都能 lowering

不是。没有 lowering 的 op 可能走 fallback、extern kernel，或者导致编译失败。

### `node.meta` 只是调试信息

不是。shape、dtype、device 等 meta 信息会影响编译决策。

## 小结

FX Graph 是 Dynamo 和 Inductor 之间的重要接口。Inductor 不是随意扫描 Python，而是解释 FX 节点，把它们 lowering 成自己的 IR。下一章讨论训练场景中一个重要中间层：AOTAutograd。

## 思考题或练习

1. 打印一个 `nn.Linear + relu` 模型的 FX graph。
2. 对比 `fx.symbolic_trace` 和 `torch._dynamo.explain` 看到的图结构差异。
3. 思考为什么 `node.meta["val"]` 这类 fake value 对编译很有用。

## 本章需要人工核查的技术点

- Dynamo 交给 Inductor 的节点 target 形式可能随 decomposition 和版本变化。
- `node.meta` 中具体字段不是稳定公共 API。

