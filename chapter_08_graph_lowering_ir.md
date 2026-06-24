# 第 8 章：GraphLowering、IR、Buffer 与 Layout

![GraphLowering 与 IR](./assets/lowering_scheduler.svg)

## 本章目标

本章进入 Inductor 的核心内部：`GraphLowering` 如何把 FX Graph 转换成 Inductor IR。读完后你应该知道 `TensorBox`、`StorageBox`、`ComputedBuffer`、`Pointwise`、`Reduction`、`Layout` 这些名字大概代表什么。

本章承接第 7 章的这一行源码主线：

```text
_InProcessFxCompile.codegen_and_compile
  -> graph = GraphLowering(...)
  -> graph.run(*example_inputs)
```

`graph.run(...)` 是本章的入口。它会像 FX Interpreter 一样遍历 `GraphModule`，但每个节点的执行结果不是 eager Tensor，而是 Inductor IR。

## 背景知识

FX Graph 适合表达 PyTorch 算子级计算，但代码生成还需要更低层的信息：

- 输出 buffer 应该多大？
- stride 是什么？
- 这个值是 view 还是实际分配？
- 这个操作是 pointwise loop，还是 reduction loop？
- 中间结果是否必须 materialize？
- 输入输出在哪个 device 上？

这些信息由 Inductor IR 表达。

## 核心概念

### `GraphLowering`

本环境中 `torch/_inductor/graph.py` 定义：

```python
class GraphLowering(torch.fx.Interpreter):
```

它继承 FX Interpreter，说明它会解释 FX 节点。关键方法包括：

- `run`
- `call_function`
- `output`
- `codegen`
- `compile_to_module`

但它解释节点时，产物不是 eager Tensor，而是 Inductor IR 对象。

源码定位：

```text
/usr/local/lib/python3.11/site-packages/torch/_inductor/graph.py
  class GraphLowering: 约 266 行
  run: 约 875 行
  call_function: 约 1114 行
  output: 约 1264 行
  codegen: 约 1994 行
```

读 `GraphLowering` 时要抓住一个继承关系：它继承 `torch.fx.Interpreter`。所以 `placeholder`、`call_function`、`output` 这些 FX 节点，会分派到对应方法。

### `TensorBox` 与 `StorageBox`

`torch/_inductor/ir.py` 中有：

```text
class TensorBox
class StorageBox
```

可以先这样理解：

- `TensorBox` 是 Inductor 对“一个 tensor 值”的包装。
- `StorageBox` 管理这个值背后的存储表达。
- `TensorBox.realize()` 往往意味着把懒表达 materialize 成实际 buffer。

这种 box 设计让 Inductor 可以延迟决定是否真的分配中间结果。

一个简化的 mental model：

```text
TensorBox
  持有 StorageBox
    持有真正的 IR data
      可能是 Pointwise
      可能是 Reduction
      可能是 ComputedBuffer
      可能是 View
      可能是 ExternKernel
```

这套包装不是为了复杂而复杂。它允许 Inductor 先把计算表达为“可以被读取的值”，再在 Scheduler 阶段决定是否 fusion、何时 materialize。

### `ComputedBuffer`

`ComputedBuffer` 表示通过某段计算产生的 buffer。它通常会关联：

- layout。
- data，也就是 loop 表达。
- name。

如果 pointwise 链能融合，中间节点可能不会变成多个独立 `ComputedBuffer` 写回内存。

### `Pointwise` 与 `Reduction`

`ir.py` 里有：

```text
class Pointwise(Loops)
class Reduction(Loops)
```

它们都属于 loop 风格 IR。区别是：

- Pointwise：每个输出元素基本独立计算。
- Reduction：多个输入元素归约到较少输出元素，例如 sum、max。

### `Layout`

layout 描述 tensor 的形状、stride、device、dtype 等。常见类包括：

```text
FixedLayout
FlexibleLayout
```

layout 不是附属细节，它会影响索引表达、内存连续性、fusion 可行性和后端选择。

在源码里，layout 对象经常跟这些问题绑定：

- `get_size()`：输出 shape。
- `get_stride()`：输出 stride。
- `get_device()`：选择 CPU/GPU 后端。
- `get_dtype()`：决定类型提升和代码生成类型。
- `make_indexer()`：把逻辑索引映射到物理内存偏移。

当你看到某个 kernel 生成了很复杂的索引表达，通常可以回到对应 IR 的 layout 查原因。

## 源码调用链解读

### 1. `graph.run` 不是执行 Tensor，而是执行 lowering

第 7 章看到 `_InProcessFxCompile` 调用：

```text
graph.run(*example_inputs)
```

这个调用最终来自 FX Interpreter。解释执行到一个 `call_function` 节点时，会进入 `GraphLowering.call_function`。

### 2. `call_function` 查 `lowerings`

2.7.1 中 `GraphLowering.call_function` 的核心逻辑可以概括为：

```text
if target not in lowerings:
    尝试 fallback 或报 MissingOperator
else:
    selected_lowering = lowerings[target]
    out = selected_lowering(*args, **kwargs)
```

这里的 `target` 通常是 `torch.ops.aten.*` 的某个 overload，例如：

```text
aten.add.Tensor
aten.relu.default
aten.sum.dim_IntList
aten.mm.default
```

这就是第 8 章和第 10、11 章的连接点：普通 pointwise/reduction 去 `lowering.py`，matmul/conv 等大算子可能去 `kernel/` 目录里的专门 lowering。

### 3. missing lowering 时发生什么

`call_function` 中如果找不到 lowering，会按情况处理：

- 如果 op 在 fallback allow list 中，创建 fallback。
- 如果开启 implicit fallback，尝试保守 fallback。
- 如果存在 decomposition 但没有应用，会抛 `MissingOperatorWithDecomp`。
- 否则抛 `MissingOperatorWithoutDecomp`。

这对调试非常重要。看到“unsupported operator”时，不要直接去 codegen 里找；先看这个 op 有没有进入 `lowerings`，有没有 decomposition，或者是否被 fallback。

### 4. `output` 会强制实现一部分值

`GraphLowering.output` 会把解释器得到的结果整理成输出列表，并调用类似：

```text
ir.ExternKernel.realize_input(x)
```

这意味着输出边界通常会把懒表达变成某种可返回的实体。中间 pointwise 可以保持懒，但函数输出最终需要有明确的 buffer/view/常量表达。

### 5. `GraphLowering` 收集 Scheduler 的输入

`GraphLowering` 在 lowering 过程中不断填充：

```text
self.operations
self.buffers
self.graph_inputs
self.graph_outputs
self.constants
```

第 9 章的 Scheduler 会消费 `self.operations`：

```text
Scheduler(self.operations)
```

所以可以把第 8 章理解成“生产调度素材”的阶段。

## 一个最小 PyTorch 示例

```python
import torch

def f(x):
    y = x + 1
    z = torch.relu(y)
    return z * 2

x = torch.randn(1024)
compiled_f = torch.compile(f)
compiled_f(x)
```

对这个例子，Inductor 可以把 `add -> relu -> mul` 表达成一个 pointwise loop。

## 编译前后发生了什么

从 FX 到 IR 的过程可以理解成：

```text
FX node: aten.add.Tensor
  -> lowering.py 查找 lowering
  -> 创建 Pointwise / TensorBox 等 IR

FX node: aten.relu.default
  -> lowering
  -> 继续包裹 pointwise 表达

FX output
  -> GraphLowering.output
  -> 标记 graph_outputs
```

GraphLowering 在解释过程中维护：

- `graph_inputs`
- `graph_outputs`
- `buffers`
- `operations`
- `constants`
- `device_types`
- `sizevars`

这些字段在后续 scheduler 和 wrapper codegen 中会继续使用。

更贴近源码的对象流是：

```text
FX placeholder
  -> GraphLowering.placeholder
  -> InputBuffer / TensorBox

FX call_function target
  -> GraphLowering.call_function
  -> lowerings[target]
  -> TensorBox / IRNode / ExternKernel

FX output
  -> GraphLowering.output
  -> graph_outputs

GraphLowering.operations
  -> Scheduler(self.operations)
```

## TorchInductor 内部大致发生了什么

对 pointwise 链，Inductor 倾向于保持懒表达：

```text
TensorBox(add)
  -> TensorBox(relu(add))
  -> TensorBox(mul(relu(add), 2))
```

只要没有必须实现的边界，Scheduler 可以把它们合成一个 kernel。必须 realize 的情况包括但不限于：

- 值被多个不兼容 consumer 使用。
- mutation 或 alias 语义需要保守处理。
- 后端限制要求 materialize。
- 输出需要稳定 layout。

源码阅读时，建议把 `realize()` 当作一个“边界信号”：

```text
还没 realize:
  这个值可能只是一个可内联的表达，可以继续 fusion。

已经 realize:
  这个值大概率已经需要一个具名 buffer，后续调度会把它当作依赖节点。
```

这不是绝对规则，但对初学者读源码很有帮助。

## 例子：`relu(x + 1) * 2` 如何经过 GraphLowering

假设 FX 图已经是 ATen 风格：

```text
placeholder x
call_function aten.add.Tensor
call_function aten.relu.default
call_function aten.mul.Tensor
output
```

GraphLowering 的处理顺序是：

```text
placeholder x
  -> 创建输入 TensorBox

aten.add.Tensor
  -> lowerings[aten.add.Tensor]
  -> make_pointwise 风格 lowering
  -> 返回 TensorBox(add expression)

aten.relu.default
  -> lowerings[aten.relu.default]
  -> 继续包装 pointwise expression

aten.mul.Tensor
  -> lowerings[aten.mul.Tensor]
  -> 继续包装 pointwise expression

output
  -> 标记 graph_outputs
  -> 必要时 realize
```

到这一步还没有真正生成 Triton 或 C++。只是形成了 Inductor 能调度的 IR。下一章的 Scheduler 会决定这些表达是否融合为一个 kernel。

## 关键源码入口

```text
/usr/local/lib/python3.11/site-packages/torch/_inductor/graph.py
/usr/local/lib/python3.11/site-packages/torch/_inductor/ir.py
/usr/local/lib/python3.11/site-packages/torch/_inductor/lowering.py
/usr/local/lib/python3.11/site-packages/torch/_inductor/virtualized.py
/usr/local/lib/python3.11/site-packages/torch/_inductor/sizevars.py
```

和上下章的连接：

```text
上一章 compile_fx.py:
  GraphLowering(...) 和 graph.run(...)

本章 graph.py / ir.py:
  FX node -> lowerings[target] -> IR

下一章 scheduler.py:
  Scheduler(self.operations) -> fusion -> backend.codegen_node(...)
```

建议搜索：

```bash
rg -n "class GraphLowering|def call_function|def output" torch/_inductor/graph.py
rg -n "class TensorBox|class Pointwise|class Reduction|class ComputedBuffer" torch/_inductor/ir.py
```

## 常见误区

### IR 是另一种 FX Graph

不是。FX 是算子图；Inductor IR 更接近“如何生成循环和 buffer”的表示。

### layout 只是性能优化

layout 也关系正确性。stride、view、alias 和输出格式都需要正确处理。

### realize 一定不好

不一定。realize 可能增加内存写回，但有时能简化依赖、支持复用、满足后端要求。

## 小结

GraphLowering 是 Inductor 从图到内部世界的入口。它把 FX 节点 lowering 成 IR，并积累 buffer、operation、layout、shape 等信息。下一章看 Scheduler 如何使用这些 IR 决定 fusion 和代码生成顺序。

## 思考题或练习

1. 阅读 `ir.py` 中 `Pointwise` 和 `Reduction` 的类定义，比较它们字段差异。
2. 用 `x.transpose(0, 1)` 的例子思考 view 如何影响 layout。
3. 搜索 `realize()` 的调用点，观察哪些情况会强制 materialize。

## 本章需要人工核查的技术点

- IR 类字段经常调整，本章解释以概念为主。
- `TensorBox.realize()` 的触发条件复杂，应针对具体版本和具体图核查。
