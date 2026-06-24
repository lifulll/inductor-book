# 第 6 章：AOTAutograd：训练图如何进入 Inductor

## 本章目标

本章解释训练场景为什么比推理更复杂，以及 AOTAutograd 如何把前向和反向计算整理成可交给 Inductor 编译的图。

## 背景知识

推理只需要前向：

```python
y = model(x)
```

训练需要：

```python
loss = model(x).sum()
loss.backward()
```

反向图不是用户显式写出来的 Python 代码，而是 autograd 根据前向操作动态构造的。要编译训练，系统需要提前拿到 backward 的计算结构。

## 核心概念

### AOT 的意思

AOT 是 ahead-of-time。这里不是说整个程序像 C++ 一样预编译，而是说 autograd 的前向和反向图在运行前被追踪、拆分、编译。

### joint graph

AOTAutograd 可以先构造同时包含前向与反向关系的 joint graph，再通过 partitioner 分成 forward graph 和 backward graph。

### saved tensors

反向计算通常需要前向中间值。例如 ReLU backward 需要知道前向哪些位置大于 0。AOTAutograd 需要决定哪些中间值保存下来，哪些可以重算。

## 一个最小 PyTorch 示例

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

print(x.grad.shape, w.grad.shape)
```

如果启用日志：

```bash
TORCH_LOGS=aot_graphs python train_example.py
```

可以尝试查看 AOTAutograd 相关图。日志项可用：

```bash
TORCH_LOGS=help python -c "import torch"
```

核查当前版本支持项。

## 编译前后发生了什么

训练路径高度简化如下：

```text
Dynamo 捕获用户 forward
  -> AOTAutograd 追踪 autograd 语义
  -> 得到 forward graph 和 backward graph
  -> 分别或延迟交给 Inductor compile_fx_inner
  -> 运行时 forward 保存 backward 所需状态
  -> backward 被调用时执行编译后的 backward
```

在本环境 2.7.1 中，`torch/_inductor/compile_fx.py` 的 `compile_fx` 文档明确写着：虽然它位于 `torch._inductor`，但它负责调用 AOTAutograd，最终回调 `inner_compile` 执行实际编译。

## TorchInductor 内部大致发生了什么

Inductor 对 forward graph 和 backward graph 的编译机制相似：都从 FX GraphModule 进入 `compile_fx_inner`，再进入 GraphLowering、Scheduler 和 Codegen。

区别是 backward graph 常有更多：

- gradient 输入。
- saved tensors。
- view/mutation 语义处理。
- memory planning 压力。
- recomputation 与保存中间值的权衡。

## 关键源码入口

```text
/usr/local/lib/python3.11/site-packages/torch/_functorch/aot_autograd.py
/usr/local/lib/python3.11/site-packages/torch/_functorch/partitioners.py
/usr/local/lib/python3.11/site-packages/torch/_functorch/_aot_autograd/
/usr/local/lib/python3.11/site-packages/torch/_inductor/compile_fx.py
```

阅读建议：

- 第一遍只理解 `aot_autograd.py` 的角色，不要陷入所有 wrapper。
- `partitioners.py` 是理解 forward/backward 拆分和保存策略的关键。
- 回到 `compile_fx.py` 看 Inductor 如何作为 AOTAutograd 的 compiler callback。

## 常见误区

### AOTAutograd 只和 functorch 有关，和 Inductor 无关

它确实位于 functorch 相关目录，但在 `torch.compile(..., backend="inductor")` 训练路径上，它是进入 Inductor 前的重要阶段。

### backward 一定在第一次 forward 时全部编译

具体策略可能随版本和配置变化。源码中存在 lazy backward 编译相关逻辑，阅读时要以当前版本为准。

### saved tensor 越少越好

不一定。少保存会增加重算，可能降低性能。

## 小结

推理编译主要处理前向图；训练编译还要处理 autograd。AOTAutograd 把动态 autograd 过程提前整理成图，再交给 Inductor。下一章进入 Inductor 的正式入口 `compile_fx.py`。

## 思考题或练习

1. 编译一个只做推理的函数和一个会 `backward` 的 loss function，对比日志。
2. 用 `TORCH_LOGS=aot_graphs` 查看 forward/backward 图。
3. 思考为什么 backward 图里可能出现前向没有显式写出的操作。

## 本章需要人工核查的技术点

- AOTAutograd 的内部包装层在 PyTorch 版本间变化较快。
- lazy backward compile、autograd cache、partitioner 细节应基于当前源码复核。

