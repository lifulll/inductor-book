# 第 10 章：Pointwise、Reduction、Softmax 与常见 Lowering

## 本章目标

本章从算子类型角度解释 Inductor 如何处理常见计算：pointwise、reduction、softmax、view 和 fallback。重点不是背 API，而是建立“看到一个 PyTorch op，知道去哪里查 lowering”的能力。

本章从调用链上看，是第 8 章里这一句的展开：

```text
GraphLowering.call_function
  -> selected_lowering = lowerings[target]
  -> selected_lowering(*args, **kwargs)
```

第 8 章解释了 `GraphLowering` 如何调用 lowering；本章解释 lowering 函数内部通常做什么。

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

源码定位：

```text
/usr/local/lib/python3.11/site-packages/torch/_inductor/lowering.py
  register_lowering: 约 477 行
  make_pointwise: 约 574 行
  make_reduction: 约 5785 行
```

`register_lowering` 本身是装饰器入口。它会把一个或多个 ATen overload 注册进全局 `lowerings` 表。`GraphLowering.call_function` 查的就是这张表。

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

`make_pointwise` 的源码读法：

```text
输入 TensorBox
  -> promote_constants
  -> inputs[i].make_loader()
  -> ranges = inputs[0].get_size()
  -> 构造 inner_fn(index)
  -> inner_fn 里按 index load 输入
  -> 调用真正的标量计算 fn
  -> 返回 Pointwise 风格 IR
```

最重要的是 `make_loader()` 和 `inner_fn(index)`。它们说明 Inductor 并不是马上把整个 Tensor 算出来，而是构造“给我一个逻辑 index，我知道怎么计算这个元素”的函数。Scheduler 和 codegen 后面会把这个函数变成循环或 Triton program。

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

`make_reduction` 的源码读法：

```text
_make_reduction_inner
  -> 计算 kept sizes 和 reduced sizes
  -> 构造 loader
  -> 返回 device / dtype / ranges / reduction_ranges

Reduction.create(...)
  -> 创建 Reduction IR
  -> 某些情况下 result.realize()
```

和 pointwise 相比，reduction 多了 `reduction_ranges`。这会在后端 codegen 中变成 reduction 维度的循环、block reduction 或 cooperative reduction。

### Softmax

Softmax 通常不是单纯 pointwise：

```text
softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
```

它包含 max reduction、exp、sum reduction、除法。Inductor 可能使用专门 lowering 或融合模式。源码中可搜索 `softmax` 和 `prepare_softmax_online`。

在源码里查 softmax 时，建议不要只搜 `aten.softmax`。还要搜：

```text
_softmax
softmax
prepare_softmax_online
```

因为 AOT/decomposition/post-grad pass 可能已经把高层 softmax 改写成更基础的 reduction + pointwise 组合，或者使用 Inductor 自己的 primitive。

### View、reshape、transpose

这些 op 很多时候不产生真实计算，而是改变 shape/stride 解释。Inductor 需要判断：

- 能否作为 view 保留。
- 是否必须 contiguous。
- 是否需要 copy。
- 后续 kernel 能否直接按 stride 访问。

本章也要和第 7 章联系起来：`_InProcessFxCompile.codegen_and_compile` 在进入 GraphLowering 前会调用 `view_to_reshape(gm)`。源码注释说明，这主要和 layout optimization 有关：当 contiguous tensor 被布局优化成 channels-last 等格式时，原本合法的 view 可能不再合法，因此先把 view 转为 reshape 更稳妥。

## 源码调用链解读

### 1. 从 `aten.add.Tensor` 到 pointwise IR

对于：

```python
def f(x, y):
    return x + y
```

源码链路是：

```text
FX node target = aten.add.Tensor
  -> GraphLowering.call_function
  -> lowerings[aten.add.Tensor]
  -> make_pointwise(...)
  -> TensorBox(Pointwise(...))
```

如果后面紧跟 `relu`、`mul`，这些 lowering 会继续返回可组合的 pointwise 表达。最终 Scheduler 可能把它们融合。

### 2. broadcasting 在 lowering 阶段处理

`lowering.py` 中有 `broadcast_symbolic_shapes` 和类型提升逻辑。比如：

```python
x.shape == [128, 256]
b.shape == [256]
y = x + b
```

Inductor 需要把 `b` 的访问表达成 broadcast 后的索引。这个工作不应该留到 Triton 字符串拼接阶段才临时处理，而是在 lowering/IR 阶段就表达清楚。

### 3. reduction 经常会主动 realize

`make_reduction` 中创建 `Reduction` 后，某些情况下会调用 `result.realize()`。对初学者来说，可以先这样理解：

```text
Pointwise 链:
  更容易保持懒表达，等待融合。

Reduction:
  经常形成一个更明确的调度边界，因为它改变 iteration space。
```

这也是为什么 `relu(x).sum(dim=1)` 可能融合 producer，但不一定像纯 pointwise 链那样完全自然。

### 4. fallback 是 lowering 的一部分，不是 codegen 失败后的补丁

`GraphLowering.call_function` 在发现 target 不在 `lowerings` 中时，可能创建 fallback。fallback 最终也会以 IR 节点形式进入 Scheduler，例如 extern kernel 节点。

因此调试性能时，如果看到生成代码里有 extern call，要往前追：

```text
这个 op 是不是没有 lowering？
是不是走了 make_fallback？
是不是 layout constraint 导致额外 contiguous/copy？
```

### 5. 大算子不一定在 `lowering.py`

`lowering.py` 是普通算子的重要入口，但 matmul、bmm、conv 等会在 `torch/_inductor/kernel/` 下注册 lowering。例如第 11 章要看的：

```text
torch/_inductor/kernel/mm.py
  @register_lowering(aten.mm)
  def tuned_mm(...)
```

所以“查 lowering”的正确动作不是只打开 `lowering.py`，而是：

```bash
rg -n "register_lowering\\(aten\\.目标算子" torch/_inductor
```

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

更贴近源码：

```text
GraphLowering.call_function(aten.add.Tensor)
  -> lowerings[aten.add.Tensor]
  -> make_pointwise inner
  -> TensorBox.create(...)
  -> GraphLowering.operations 增加或延迟相关 IR
  -> Scheduler fusion
  -> backend.codegen_node
```

以 reduction 为例：

```text
aten.relu -> Pointwise
aten.sum -> Reduction
Scheduler 判断是否融合 producer
后端生成 reduction kernel
```

更贴近源码：

```text
GraphLowering.call_function(aten.sum...)
  -> lowerings[aten.sum...]
  -> make_reduction("sum")
  -> Reduction.create(...)
  -> maybe realize()
  -> Scheduler 根据 dependency 和 group 决定融合边界
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

源码阅读时可以按三个问题拆 lowering 函数：

```text
1. 输入规范化了吗？
   dtype、broadcast、axis、keepdim、layout。

2. 生成什么 IR？
   Pointwise、Reduction、View、ExternKernel、TemplateBuffer。

3. 是否强制 realize 或 fallback？
   这会影响第 9 章的 fusion 和第 12 章的 codegen。
```

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

和上下章的连接：

```text
第 8 章:
  GraphLowering.call_function 查 lowerings[target]

本章:
  lowering.py 把常见 ATen op 变成 Pointwise / Reduction / View / fallback

第 9 章:
  Scheduler 读取 lowering 产生的 operations 并融合

第 11 章:
  matmul/conv 等大算子走 kernel/mm.py、kernel/conv.py 和 select_algorithm.py
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
