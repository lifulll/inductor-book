# 第 9 章：Scheduler、依赖分析与 Fusion

![fusion 示意](./assets/fusion_memory.svg)

## 本章目标

本章解释 Inductor 如何从 IR 进入调度阶段：哪些节点能融合、按什么顺序生成代码、为什么有些看似简单的操作不能融合。

## 背景知识

GraphLowering 产生 IR 后，Inductor 还没有生成代码。它需要先决定：

- 哪些 buffer 依赖哪些输入？
- 哪些计算可以放进同一个 kernel？
- 哪些必须先执行？
- 生成几个 kernel？
- 每个 kernel 的 iteration space 是什么？

这就是 Scheduler 的职责。

## 核心概念

### Scheduler 节点

本环境 `torch/_inductor/scheduler.py` 中有：

```text
class Scheduler
class SchedulerNode
class FusedSchedulerNode
class NopKernelSchedulerNode
class OutputNode
```

`SchedulerNode` 通常包装一个 IR operation；`FusedSchedulerNode` 表示多个节点融合后的调度单元。

### 依赖分析

Scheduler 需要知道每个节点读写哪些 buffer。相关逻辑位于：

```text
torch/_inductor/dependencies.py
```

依赖分析回答：

- producer 是谁？
- consumer 是谁？
- 是否存在写后读、读后写、别名或 mutation 风险？
- 两个节点交换顺序是否安全？

### Vertical fusion 与 horizontal fusion

可以用直觉区分：

- vertical fusion：生产者和消费者融合，例如 `add -> relu`。
- horizontal fusion：多个独立但形状相近的计算合并，例如两个并列 pointwise 输出。

源码中 `Scheduler` 有 `can_fuse_vertical`、`can_fuse_horizontal` 等判断。

### fusion 的收益估计

源码中存在 `speedup_by_fusion` 等逻辑。融合不是越多越好。融合可能带来：

- 更少 kernel launch。
- 更少中间内存访问。
- 更大 kernel，寄存器压力增加。
- 并行度变差。
- 代码生成复杂度上升。

## 一个最小 PyTorch 示例

```python
import torch

def f(x, y):
    a = x + y
    b = torch.relu(a)
    c = torch.sigmoid(b)
    return c * 3

x = torch.randn(1024, 1024, device="cuda")
y = torch.randn(1024, 1024, device="cuda")
torch.compile(f)(x, y)
```

这是典型 vertical fusion 场景。

一个更复杂的例子：

```python
def g(x, y):
    return (x + 1, y + 1)
```

这里可能涉及 horizontal fusion，具体取决于 shape、device、调度策略和版本。

## 编译前后发生了什么

Scheduler 大致流程：

```text
IR operations/buffers
  -> create SchedulerNode
  -> 计算 read/write dependency
  -> 尝试 fusion
  -> 排序
  -> 调用 backend scheduling.codegen()
```

GraphLowering 的 `codegen` 中会调用：

```text
self._update_scheduler()
self.scheduler.codegen()
self.wrapper_code.generate(...)
```

这说明 wrapper codegen 和 kernel codegen 是调度之后的事情。

## TorchInductor 内部大致发生了什么

对于：

```python
return torch.relu(x + y) * 2
```

可能形成一个融合节点。这个节点包含原本多个 pointwise 计算，在 GPU 上交给 `TritonScheduling` 生成 Triton kernel；在 CPU 上交给 `CppScheduling` 生成 C++ loop 或调用合适后端。

不能融合的常见原因：

- device 不同。
- dtype/layout 不兼容。
- reduction 和 pointwise 的 iteration space 不适合合并。
- mutation 或 alias 语义风险。
- 外部 kernel 边界，例如某些 matmul/conv/template。
- fusion 后估计收益不好。

## 关键源码入口

```text
/usr/local/lib/python3.11/site-packages/torch/_inductor/scheduler.py
/usr/local/lib/python3.11/site-packages/torch/_inductor/dependencies.py
/usr/local/lib/python3.11/site-packages/torch/_inductor/codegen/simd.py
/usr/local/lib/python3.11/site-packages/torch/_inductor/codegen/triton.py
/usr/local/lib/python3.11/site-packages/torch/_inductor/codegen/cpp.py
```

建议搜索：

```bash
rg -n "class Scheduler|def fuse_nodes|def can_fuse|def codegen" torch/_inductor/scheduler.py
rg -n "class ReadWrites|class MemoryDep" torch/_inductor/dependencies.py
```

## 常见误区

### 能融合就一定融合

不是。Scheduler 会考虑正确性和收益。

### fusion 只对 GPU 有意义

不是。CPU 上减少中间内存访问和循环开销同样有意义，只是收益模型和代码生成不同。

### fusion 后一定只有一个输出

不一定。融合 kernel 可以产生多个输出，但这会增加复杂性。

## 小结

Scheduler 是 Inductor 的决策中心之一。它把 IR 操作变成可生成代码的调度单元，并决定 fusion。下一章聚焦几个常见算子族：pointwise、reduction、softmax 等如何 lowering。

## 思考题或练习

1. 用 `TORCH_COMPILE_DEBUG=1` 查看 debug 目录中的 pre/post fusion 信息。
2. 尝试写一个返回两个 pointwise 输出的函数，观察生成 kernel 数量。
3. 搜索 `can_fuse_vertical`，阅读其中的限制条件。

## 本章需要人工核查的技术点

- fusion 规则和收益模型随版本变化很快。
- `INDUCTOR_POST_FUSION_SVG=1` 需要 Graphviz 支持，生成图行为依赖环境。

