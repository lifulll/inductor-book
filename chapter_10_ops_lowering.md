# 第 10 章：Pointwise、Reduction、Softmax 与常见 Lowering

## 本章目标

本章从算子类型角度解释 Inductor 如何处理常见计算：pointwise、reduction、softmax、view 和 fallback。重点不是背 API，而是建立“看到一个 PyTorch op，知道去哪里查 lowering”的能力。

## 背景知识

Inductor 通过 lowering 把 FX node target 转成 IR。核心文件：

```text
torch/_inductor/lowering.py
```

此外，一些重要算子族分散在专门文件中，例如 matmul 在 `kernel/mm.py`，卷积在 `kernel/conv.py`。

## 核心概念

### Lowering registry

`lowering.py` 中大量使用 `register_lowering(...)`。它把 ATen op 映射到 Python lowering 函数。

当 GraphLowering 遇到 `aten.relu.default` 这类 target，会查找对应 lowering，生成 Inductor IR。

### Pointwise

Pointwise 是 Inductor 最容易优化的一类：

```python
def f(x, y):
    return torch.sigmoid(x + y) * 3
```

其核心特征是每个输出元素可以独立计算：

```text
out[i] = sigmoid(x[i] + y[i]) * 3
```

通常适合 fusion。

### Reduction

Reduction 把一个或多个维度归约掉：

```python
def g(x):
    return torch.sum(torch.relu(x), dim=1)
```

Reduction 需要处理：

- reduction 维度。
- 初始值。
- combine 函数。
- 并行归约策略。
- 输出 shape。

在 GPU 上，reduction 的代码生成比 pointwise 更复杂，涉及 block、mask、局部归约和跨 program 组合等。

### Softmax

Softmax 通常不是单纯 pointwise：

```text
softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
```

它包含 max reduction、exp、sum reduction、除法。Inductor 可能使用专门 lowering 或融合模式。源码中可搜索 `softmax` 和 `prepare_softmax_online`。

### View、reshape、transpose

这些 op 很多时候不产生真实计算，而是改变 shape/stride 解释。Inductor 需要判断：

- 能否作为 view 保留。
- 是否必须 contiguous。
- 是否需要 copy。
- 后续 kernel 能否直接按 stride 访问。

## 一个最小 PyTorch 示例

```python
import torch

def pointwise(x, y):
    return torch.sigmoid(x + y) * 3

def reduction(x):
    return torch.sum(torch.relu(x), dim=1)

def softmax_case(x):
    return torch.softmax(x, dim=-1)

device = "cuda" if torch.cuda.is_available() else "cpu"
x = torch.randn(128, 256, device=device)
y = torch.randn(128, 256, device=device)

torch.compile(pointwise)(x, y)
torch.compile(reduction)(x)
torch.compile(softmax_case)(x)
```

配合：

```bash
TORCH_LOGS=output_code python ops.py
```

观察生成代码差异。

## 编译前后发生了什么

以 pointwise 为例：

```text
aten.add -> Pointwise
aten.sigmoid -> Pointwise
aten.mul -> Pointwise
Scheduler fusion
Triton/C++ codegen
```

以 reduction 为例：

```text
aten.relu -> Pointwise
aten.sum -> Reduction
Scheduler 判断是否融合 producer
后端生成 reduction kernel
```

## TorchInductor 内部大致发生了什么

`lowering.py` 不只是简单映射。它经常要：

- 做类型提升。
- 处理 broadcasting。
- 创建 layout。
- 选择 fallback。
- 插入 view/copy。
- 调用 `ir.Pointwise` 或 `ir.Reduction`。
- 对复杂 op 调用 decomposition 或专用 helper。

对于缺少 lowering 的 op，可能出现 fallback 到 ATen/external kernel 的路径。fallback 能保持正确性，但通常减少 fusion 机会。

## 关键源码入口

```text
/usr/local/lib/python3.11/site-packages/torch/_inductor/lowering.py
/usr/local/lib/python3.11/site-packages/torch/_inductor/ir.py
/usr/local/lib/python3.11/site-packages/torch/_inductor/ops_handler.py
/usr/local/lib/python3.11/site-packages/torch/_inductor/index_propagation.py
/usr/local/lib/python3.11/site-packages/torch/_inductor/codegen/triton.py
/usr/local/lib/python3.11/site-packages/torch/_inductor/codegen/cpp.py
```

建议搜索：

```bash
rg -n "register_lowering.*relu|register_lowering.*sum|softmax" torch/_inductor/lowering.py
rg -n "class Pointwise|class Reduction" torch/_inductor/ir.py
```

## 常见误区

### 所有 ATen op 都在 `lowering.py`

不是。大型算子族可能在 `torch/_inductor/kernel/` 或其他专门文件里注册 lowering。

### Reduction 只是多一个 for 循环

不是。高性能 reduction 涉及并行策略、内存访问模式和数值细节。

### view 不花时间所以不重要

view 可能不产生 kernel，但会改变 stride 和索引表达，对后续 kernel 性能非常重要。

## 小结

lowering 是 Inductor 从“算子图”走向“循环和 buffer”的关键翻译层。Pointwise 通常最容易融合；reduction 和 softmax 更复杂；view/layout 虽然不总是执行计算，却经常决定性能边界。

## 思考题或练习

1. 搜索 `aten.sum` 的 lowering，找到它最终如何构造 reduction。
2. 写一个 `transpose -> relu` 函数，观察生成代码是否直接按 stride 访问。
3. 对比 `softmax` 和手写 `exp/sum/div` 的生成代码。

## 本章需要人工核查的技术点

- softmax lowering 和在线 softmax 优化路径随版本变化。
- fallback 的具体实现和性能影响需要结合生成代码判断。

