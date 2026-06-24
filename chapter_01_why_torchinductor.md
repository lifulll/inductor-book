# 第 1 章：为什么 PyTorch 需要 TorchInductor？

## 本章目标

这一章先不急着进入源码细节。我们先回答一个更基础的问题：如果 PyTorch 的 eager mode 已经这么好用，为什么还需要 `torch.compile`，又为什么需要 TorchInductor？

读完本章，你应该能理解：

- eager mode 为什么适合研究和调试。
- eager mode 在性能上可能遇到什么瓶颈。
- 编译器能从一段 PyTorch 程序里看到什么优化机会。
- `torch.compile`、TorchDynamo 和 TorchInductor 的大致关系。
- 为什么第一次运行 compiled function 通常更慢，而后续运行可能更快。

## 背景知识

PyTorch 最重要的体验之一是 eager mode，也就是“写一行，执行一行”。

例如：

```python
import torch

x = torch.randn(4)
y = x + 1
z = torch.relu(y)
print(z)
```

这段代码的执行方式非常直观：

1. 创建一个 Tensor。
2. 执行加法。
3. 执行 ReLU。
4. 打印结果。

每一步都会立刻发生。你可以随时 `print`，随时打断点，随时查看 Tensor 的值。这种模式非常适合教学、实验和模型开发。

但对性能优化来说，eager mode 有一个天然限制：它通常只能看到“当前这一小步”，很难提前知道后面还会发生什么。

如果程序是：

```python
def f(x, y):
    a = x + y
    b = torch.relu(a)
    c = b * 2
    return c
```

从数学上看，这只是一个很简单的计算链：

```text
x, y -> add -> relu -> multiply -> output
```

但是在 eager mode 下，PyTorch 可能会分别调度多个底层操作：

```text
kernel 1: x + y
kernel 2: relu(a)
kernel 3: b * 2
```

在 GPU 上，每次 kernel launch 都有调度成本。更重要的是，中间结果 `a` 和 `b` 可能要写入显存，再被下一个 kernel 读出来。对于很多 pointwise 操作来说，真正慢的常常不是计算本身，而是内存读写和调度开销。

这就是编译器开始有用的地方。

## 核心概念

### Eager mode

Eager mode 是 PyTorch 默认执行方式。它的特点是：

- Python 代码逐行执行。
- 每个 Tensor 操作通常会立即触发实际计算。
- 调试体验好。
- 对动态 Python 代码非常友好。

它的代价是：

- 很难跨多个操作做全局优化。
- 可能产生较多 kernel launch。
- 中间 Tensor 读写可能较多。
- Python 调度开销在小算子或碎片化模型中可能变得明显。

### Graph mode

Graph mode 的思路是：先把一段计算捕获成“图”，再对整张图做优化。

图可以理解成一张计算流程图：

```text
placeholder x
placeholder y
     ↓
   add
     ↓
   relu
     ↓
 multiply
     ↓
 output
```

一旦有了图，编译器就能提出问题：

- 这些操作能不能融合成一个 kernel？
- 中间结果是否真的需要写入内存？
- 有没有不必要的 copy？
- 输入 shape 是否稳定，可以生成专门优化过的代码？
- CPU 和 GPU 应该走不同的代码生成路径吗？

### `torch.compile`

`torch.compile` 是 PyTorch 2.x 提供给用户的编译入口。它不是单独一个小工具，而是一个把多个组件串起来的入口。

一个高度简化的链路是：

```text
用户 Python 函数
        ↓
torch.compile
        ↓
TorchDynamo 捕获 Python 执行
        ↓
FX Graph 表示计算
        ↓
AOTAutograd 处理训练图，视场景而定
        ↓
TorchInductor 优化图并生成可执行代码
```

这条链路里，TorchInductor 主要负责后半段：接收图，做优化，生成代码，编译代码，然后执行。

### TorchInductor

TorchInductor 可以先粗略理解成 PyTorch 2.x 默认编译后端之一。它的目标不是改变用户写 PyTorch 的方式，而是在尽量保持 PyTorch 易用性的前提下，让可优化的计算跑得更快。

它关心的问题包括：

- 如何把 FX Graph 降低到内部 IR。
- 如何决定哪些操作可以 fusion。
- 如何为 pointwise、reduction、matmul 等模式生成合适代码。
- 如何处理 layout、stride、buffer 和中间内存。
- 如何在 GPU 上生成 Triton kernel。
- 如何在 CPU 上生成或调用合适的底层实现。
- 如何缓存编译结果，避免重复编译。

## 一个最小 PyTorch 示例

下面是一个最小例子：

```python
import time
import torch

def f(x, y):
    return torch.relu(x + y) * 2

device = "cuda" if torch.cuda.is_available() else "cpu"
x = torch.randn(1024, 1024, device=device)
y = torch.randn(1024, 1024, device=device)

compiled_f = torch.compile(f)

# eager
out_eager = f(x, y)

# 第一次 compiled 调用通常包含编译开销
out_compiled = compiled_f(x, y)

print(torch.allclose(out_eager, out_compiled))
```

这个例子很短，但它已经包含 TorchInductor 最适合入门的一类优化机会：

```text
add -> relu -> multiply
```

这些操作都是 pointwise 操作。所谓 pointwise，就是输出中的每个元素基本可以独立计算：

```text
out[i] = relu(x[i] + y[i]) * 2
```

这种模式很适合 fusion。编译器可以尝试把多个小操作合并成一个更大的 kernel：

```text
原始执行：
  kernel 1: add
  kernel 2: relu
  kernel 3: multiply

可能的编译后执行：
  kernel 1: add + relu + multiply
```

注意这里说的是“可能”。真实行为会受到 PyTorch 版本、设备、输入 shape、dtype、配置项和后端能力影响。正式分析时应查看当前版本生成的代码。

## 编译前后发生了什么

先看 eager mode。

```python
out = torch.relu(x + y) * 2
```

从用户视角看，这是一行代码。从底层执行视角看，它包含多个 Tensor 操作：

1. `x + y`
2. `torch.relu(...)`
3. `... * 2`

在 eager mode 下，这些操作倾向于按顺序立即执行。

再看 compiled mode。

```python
compiled_f = torch.compile(f)
out = compiled_f(x, y)
```

第一次调用 `compiled_f(x, y)` 时，PyTorch 可能会做一批额外工作：

1. 观察这次 Python 函数执行。
2. 尝试捕获 Tensor 相关计算。
3. 生成 FX Graph。
4. 把图交给编译后端。
5. TorchInductor 分析并优化图。
6. 生成底层代码。
7. 编译并缓存生成结果。
8. 执行编译后的代码。

所以第一次运行 compiled function 不一定快，甚至通常会更慢，因为它包含编译开销。

真正重要的是后续调用。如果输入满足之前编译结果的 guard 条件，PyTorch 可以复用已经生成的代码：

```text
第一次调用：
  捕获 + 编译 + 执行

后续调用：
  检查 guard + 执行缓存代码
```

这也是为什么 benchmark `torch.compile` 时，通常需要区分：

- compile time
- warmup time
- steady-state execution time

只测第一次调用，很容易得出错误结论。

## TorchInductor 内部大致发生了什么

仍然以这个函数为例：

```python
def f(x, y):
    return torch.relu(x + y) * 2
```

TorchInductor 不会直接读你写的 Python 源码。通常情况下，它接收到的是前面组件准备好的图表示。

一个简化后的图可能像这样：

```text
x: placeholder
y: placeholder
add = x + y
relu = relu(add)
mul = relu * 2
return mul
```

接下来 TorchInductor 会把这张图转换成更适合代码生成的内部表示，并尝试做优化。对这个例子来说，最直观的优化是融合：

```text
for i in range(numel):
    out[i] = max(x[i] + y[i], 0) * 2
```

如果目标是 GPU，TorchInductor 可能生成 Triton 代码。Triton 可以把这种逐元素计算表达成 GPU kernel。

一个极度简化的 Triton 风格伪代码如下：

```python
@triton.jit
def kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    out = tl.maximum(x + y, 0.0) * 2.0

    tl.store(out_ptr + offsets, out, mask=mask)
```

这不是 TorchInductor 对上面例子必然生成的真实代码，只是帮助理解的教学版本。真实代码会包含更多参数、布局处理、类型处理、启发式配置和运行时包装逻辑。

如果目标是 CPU，TorchInductor 可能走 C++ 代码生成、ATen 调用或其他 CPU 后端路径。具体实现随 PyTorch 版本、平台和配置变化较大，正文后续章节会专门讨论。

## 关键源码入口

本章只建议先建立地图，不建议一开始钻进全部细节。

可以先关注这些路径：

```text
torch/__init__.py
torch/_dynamo/
torch/_inductor/
torch/_inductor/compile_fx.py
```

大致阅读顺序：

1. 从公开 API 认识 `torch.compile` 的入口。
2. 进入 `torch/_dynamo/`，理解为什么需要捕获 Python 程序。
3. 进入 `torch/_inductor/compile_fx.py`，理解 TorchInductor 如何作为编译后端接收 FX Graph。
4. 之后再看 `graph.py`、`ir.py`、`scheduler.py` 和 `codegen/`。

不要试图第一天就读完整个 `torch/_inductor/`。更好的方式是带着一个极小例子，打开 debug 输出，然后沿着生成结果反推源码路径。

## 常见误区

### 误区一：`torch.compile` 会让所有程序都更快

不一定。

`torch.compile` 对很多模型有效，但它不是魔法。以下情况可能收益有限，甚至变慢：

- 模型太小，编译开销无法摊薄。
- 程序里 Python 动态控制流太多，频繁 graph break。
- 输入 shape 变化太频繁，导致重复编译。
- 操作主要由高度优化的大算子组成，额外 fusion 空间有限。
- 后端暂时不支持某些模式，只能 fallback。

### 误区二：第一次调用速度代表编译后性能

第一次调用通常包含编译成本。评估性能时要区分编译时间和稳定运行时间。

### 误区三：TorchInductor 等于 Triton

不等于。

Triton 是 TorchInductor 在 GPU 后端经常使用的重要代码生成目标，但 TorchInductor 本身还包括图处理、IR、调度、fusion、layout、内存规划、运行时包装和缓存等许多部分。

### 误区四：Graph mode 要求用户放弃 PyTorch 动态特性

PyTorch 2.x 的目标不是强迫用户改写成完全静态的图语言，而是在普通 Python/PyTorch 程序中尽可能捕获可优化的部分。不能捕获的部分可能触发 graph break 或 fallback。

### 误区五：生成代码就是最终真相

生成代码非常重要，但它只是某次输入、某个配置、某个版本下的结果。TorchInductor 的策略可能因为 shape、dtype、device、环境变量、版本变化而改变。

## 小结

TorchInductor 的出现，是 PyTorch 在易用性和性能之间继续向前走的一步。

Eager mode 给了 PyTorch 极好的交互体验，但它不总是能看到跨操作优化机会。`torch.compile` 试图在不改变用户主要编程方式的前提下，把一段 PyTorch 程序捕获成图，再交给后端优化。

TorchInductor 就是这条链路中的核心编译后端之一。它接收图，理解计算，做调度和融合，生成 CPU 或 GPU 可执行代码，并通过缓存减少后续调用成本。

本章只建立直觉。下一章会进一步讨论 eager mode 和 graph mode 的区别：为什么“先看见整段计算”会给优化打开空间。

## 思考题或练习

1. 写一个包含 3 个 pointwise 操作的函数，例如 `sin -> add -> relu`，分别用 eager 和 `torch.compile` 执行，确认输出是否一致。
2. 思考为什么 `x + y`、`relu`、`* 2` 这类操作适合 fusion。
3. 尝试把输入 Tensor 从 `1024 x 1024` 改成很小的 `4 x 4`，思考为什么 compiled 版本不一定更快。
4. 如果你有 GPU，尝试多次运行同一个 compiled function，观察第一次和后续调用的耗时差异。

一个简单计时脚本如下：

```python
import time
import torch

def f(x, y):
    return torch.relu(x + y) * 2

device = "cuda" if torch.cuda.is_available() else "cpu"
x = torch.randn(1024, 1024, device=device)
y = torch.randn(1024, 1024, device=device)

compiled_f = torch.compile(f)

def sync():
    if device == "cuda":
        torch.cuda.synchronize()

for name, fn in [("eager", f), ("compiled", compiled_f)]:
    sync()
    start = time.perf_counter()
    for _ in range(10):
        out = fn(x, y)
    sync()
    end = time.perf_counter()
    print(name, end - start)
```

这个脚本只是入门观察，不是严谨 benchmark。更严谨的测试需要 warmup、更多迭代、隔离编译时间，并使用专门的 benchmark 工具。

## 本章需要人工核查的技术点

写作或出版前建议核查：

- 当前 PyTorch 版本中 `torch.compile` 的默认 backend 是否仍以 Inductor 为主。
- 当前版本中 `torch.compile` 对 CPU、CUDA、不同 dtype 的默认行为。
- `torch/_inductor/compile_fx.py` 是否仍是最适合初学者跟踪的 TorchInductor 入口之一。
- 当前版本生成 Triton 代码的 debug 查看方式。
- 当前版本是否推荐使用 `TORCH_COMPILE_DEBUG=1`、`TORCH_LOGS="+inductor"` 等方式查看编译细节。

