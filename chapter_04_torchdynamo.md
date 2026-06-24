# 第 4 章：TorchDynamo：从 Python Frame 到 FX Graph

## 本章目标

本章解释 Dynamo 在调用链中的位置。Dynamo 的任务不是生成 Triton kernel，而是拦截 Python 执行，捕获可编译的 Tensor 计算区域，并把它们交给 backend。

## 背景知识

PyTorch 用户写的是普通 Python。普通 Python 有循环、分支、异常、上下文管理器、对象属性、列表字典等。要从这里提取 Tensor 计算图，需要比 `fx.symbolic_trace` 更强的机制。

Dynamo 的入口在 `torch._dynamo.optimize(...)`。上一章看到，`torch.compile` 最后会返回：

```python
torch._dynamo.optimize(
    backend=backend,
    nopython=fullgraph,
    dynamic=dynamic,
    disable=disable,
)(model)
```

## 核心概念

### Frame 和 bytecode

Dynamo 工作在 Python frame / bytecode 层。它观察 Python 函数执行，识别 Tensor 相关操作，并构造 FX Graph。

这样做的好处是：用户不需要改写模型，Dynamo 可以在运行时看到真实调用路径。

代价是：Python 太灵活，Dynamo 必须决定哪些东西能安全捕获，哪些需要 guard，哪些必须 graph break。

### Guard

guard 是“这份编译结果还能复用吗”的条件。常见 guard 包括：

- 输入 tensor 的 dtype。
- device。
- shape 或 symbolic shape 关系。
- stride/layout。
- 某些 Python 全局变量或对象身份。

当后续调用不满足 guard，Dynamo 可能重新编译，直到达到 recompile limit 后回退 eager。

### Graph break

graph break 是 Dynamo 无法继续把当前 Python 执行纳入同一张图时的断点。默认情况下，它会把前面捕获到的图交给 backend，然后在断点处回到 Python/eager，再尝试捕获后续区域。

## 一个最小 PyTorch 示例

```python
import torch
import torch._dynamo as dynamo

def f(x, y):
    z = x + y
    return torch.relu(z)

x = torch.randn(8)
y = torch.randn(8)

explanation = dynamo.explain(f)(x, y)
print(explanation)
```

`dynamo.explain` 是理解 graph、guard、graph break 的入门工具。输出格式随版本可能变化，但通常会包含图数量、break 原因、op 统计和 guard 信息。

## 编译前后发生了什么

一次调用大致是：

```text
compiled_f(x, y)
  -> Dynamo 拦截 frame
  -> 解释/转换 Python bytecode
  -> 记录 Tensor 计算
  -> 创建 FX GraphModule
  -> 调用 backend(gm, example_inputs)
  -> 得到 compiled callable
  -> 执行 compiled callable
```

如果后续输入满足 guard，则直接使用缓存的 compiled callable。

## TorchInductor 内部大致发生了什么

严格说，本章还没有进入 Inductor 内部。Dynamo 把 `gm` 和 `example_inputs` 交给 backend 时，Inductor 才真正开始工作。

但 Dynamo 的输出会深刻影响 Inductor：

- 图越完整，Inductor 越有优化空间。
- graph break 越多，fusion 越容易被打断。
- guard 越容易失败，编译成本越难摊薄。
- shape 信息越复杂，Inductor 的 symbolic shape 处理越重要。

## 关键源码入口

本环境推荐路径：

```text
/usr/local/lib/python3.11/site-packages/torch/_dynamo/eval_frame.py
/usr/local/lib/python3.11/site-packages/torch/_dynamo/convert_frame.py
/usr/local/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py
/usr/local/lib/python3.11/site-packages/torch/_dynamo/output_graph.py
/usr/local/lib/python3.11/site-packages/torch/_dynamo/guards.py
```

阅读建议：

- 先看 `eval_frame.py`，理解 Dynamo 如何接管 frame evaluation。
- 再看 `convert_frame.py`，理解 frame 如何进入转换流程。
- `symbolic_convert.py` 很长，不适合第一遍逐行读，建议带着 graph break 例子搜索相关报错。
- `output_graph.py` 帮你理解 FX Graph 如何被积累和输出。

## 常见误区

### Dynamo 是 Inductor 的一部分

不是。Dynamo 是前端捕获机制，Inductor 是 backend。两者协作紧密，但职责不同。

### guard failure 是 bug

不一定。输入 shape 或 Python 状态变化时，guard failure 是保持正确性的机制。

### graph break 必然导致错误

默认模式下，graph break 可能只是性能问题。`fullgraph=True` 时，它才会变成强约束。

## 小结

Dynamo 是 `torch.compile` 里的“前端捕获器”。它把灵活的 Python/PyTorch 程序分解为可编译的 FX 图片段，并用 guard 保证缓存代码的正确复用。下一章我们会聚焦 Dynamo 交给 Inductor 的语言：FX Graph。

## 思考题或练习

1. 用 `TORCH_LOGS=graph_breaks` 运行包含 Python `print` 的函数。
2. 用 `TORCH_LOGS=recompiles` 运行不同 shape 输入，观察重新编译原因。
3. 对同一函数分别设置 `fullgraph=False` 和 `fullgraph=True`，比较行为。

## 本章需要人工核查的技术点

- `torch._dynamo.explain` 的输出结构随版本可能变化。
- recompile limit 的默认值应以 `torch/_dynamo/config.py` 为准。

