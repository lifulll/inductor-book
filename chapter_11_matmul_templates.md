# 第 11 章：Matmul、Conv、模板与算法选择

## 本章目标

本章解释为什么矩阵乘法和卷积不能简单套用 pointwise fusion 思路，以及 Inductor 如何在外部库、Triton template、C++ template 和 autotune 之间选择。

本章承接第 10 章的结论：不是所有 lowering 都在 `lowering.py` 里。矩阵乘法、批量矩阵乘法和卷积这类重算子，通常有专门文件负责 lowering 和算法选择。

调用链可以先记成：

```text
GraphLowering.call_function(aten.mm)
  -> lowerings[aten.mm]
  -> torch/_inductor/kernel/mm.py::tuned_mm
  -> 构造 choices
  -> autotune_select_algorithm
  -> ExternKernel 或 TemplateBuffer
  -> Scheduler
  -> backend.codegen_template / codegen_extern_call
```

## 背景知识

矩阵乘法：

```python
C = A @ B
```

每个输出元素都需要一个 dot product。它是计算密集型操作，性能高度依赖 tiling、memory hierarchy、tensor core、线程块形状、数据类型和硬件。

因此 Inductor 不会把 matmul 当成普通 pointwise loop 来处理。

## 核心概念

### `tuned_mm` 和 `tuned_addmm`

本环境中：

```text
torch/_inductor/kernel/mm.py
```

包含：

```text
def tuned_mm(...)
def tuned_addmm(...)
```

并通过 `register_lowering(aten.mm)`、`register_lowering(aten.addmm)` 注册。

源码定位：

```text
/usr/local/lib/python3.11/site-packages/torch/_inductor/kernel/mm.py
  tuned_mm: 约 425 行
  tuned_addmm: 约 593 行
```

`tuned_mm` 第一眼看起来像普通 lowering，但它和第 10 章的 pointwise lowering 很不一样：它不是直接创建一个简单 `Pointwise`，而是构造一组候选实现 `choices`。

### `select_algorithm.py`

`torch/_inductor/select_algorithm.py` 负责算法选择和 autotune 相关逻辑。它会组织候选实现，benchmark 或基于启发式选择。

候选可能包括：

- ATen/external kernel。
- Triton template。
- C++ GEMM template。
- 平台特定后端。

源码定位：

```text
/usr/local/lib/python3.11/site-packages/torch/_inductor/select_algorithm.py
  AlgorithmSelectorCache.__call__: 约 1669 行
  autotune_select_algorithm: 约 2342 行
```

`AlgorithmSelectorCache.__call__` 的主线是：

```text
过滤 choices
如果没有 choice，报 NoValidChoicesError
如果只有一个非 CUDA template choice，直接 output_node
否则 precompile choices
benchmark/autotune choices
选择 timing 最小的 choice
返回 selected_choice.output_node()
```

所以 matmul lowering 的产物不是“某个固定 kernel”，而是“候选集合 + 选择逻辑”。

### Epilogue fusion

很多模型中不是单独 matmul，而是：

```python
torch.relu(x @ w + b)
```

`+ b` 和 `relu` 是 matmul 的 epilogue。把 epilogue 融入 matmul kernel 可以减少额外读写。是否能融合取决于配置、模板能力和具体图。

从调用链看，epilogue fusion 跨越两层：

```text
kernel/mm.py:
  候选 template 需要声明自己能不能接 epilogue/prologue。

scheduler.py:
  template node 和周围 pointwise node 是否能融合，要看 fusion 规则和 template 能力。
```

这就是为什么 `relu(x @ w + b)` 不一定总能变成一个 matmul template kernel。它不仅依赖数学结构，还依赖当前 choices、后端、dtype、layout 和配置。

### Autotune

Autotune 的核心思想：对多个候选 kernel 或参数配置实际测量，选择最快的。相关配置可能由 `mode="max-autotune"` 或 options 启用。

源码中 autotune 还分成两个阶段：

```text
precompile:
  先把候选 kernel 编译出来，尽量并行处理。

benchmark/autotune:
  用示例输入或生成输入测量候选时间。
```

这也解释了为什么 `mode="max-autotune"` 第一次运行可能明显更慢：它做的工作比普通 codegen 多很多。

## 源码调用链解读

### 1. `tuned_mm` 先整理问题规模

`tuned_mm(mat1, mat2, layout=None)` 会通过 `mm_args` 得到：

```text
m, n, k
layout
mat1
mat2
```

这一步把高层 `A @ B` 转成 GEMM 领域最熟悉的问题规模：

```text
[m, k] @ [k, n] -> [m, n]
```

后续所有候选配置都围绕 `m/n/k`、dtype、device、layout 展开。

### 2. choices 是算法候选列表

`tuned_mm` 会根据配置和平台追加候选：

```text
ATen / extern mm
Triton template
persistent TMA Triton template
CUTLASS template
CK template
CppGemmTemplate
external_matmul
```

不是每个环境都有所有候选。例如 CPU 不会走普通 Triton CUDA template；某些 GPU/dtype/shape 不满足 template 条件；某些外部库只在特定构建中存在。

### 3. fallback 不是失败，而是候选之一

如果 `should_fallback_to_aten(choices)` 成立，`tuned_mm` 会返回 ATen mm 的 output node。这个路径通常牺牲部分 fusion 空间，但保持正确性和覆盖面。

这点和第 10 章的 fallback 一样：fallback 是编译策略的一部分，不是异常之后临时补救。

### 4. `autotune_select_algorithm` 返回 IR 节点

`autotune_select_algorithm` 最终会返回：

```text
choices[0].output_node()
或 selected_key.output_node()
或 MultiTemplateBuffer 包装后的 TensorBox
```

也就是说，算法选择结束后，仍然回到 Inductor IR 世界。Scheduler 后面看到的是一个 extern/template/multitemplate buffer，而不是一段已经运行完的结果。

### 5. template 节点在 Scheduler 阶段继续参与融合

第 9 章看到 `Scheduler._codegen` 有：

```text
if node.is_template():
    backend.codegen_template(template_node, epilogue, prologue)
```

这说明 template 并不是普通 pointwise fused node。Scheduler 会把它作为特殊节点处理，并尝试把 prologue/epilogue 与 template 一起交给后端。

## 从 `relu(x @ w + b)` 看完整链路

假设图中出现：

```text
aten.mm
aten.add
aten.relu
```

理想化链路如下：

```text
aten.mm
  -> tuned_mm
  -> choices: extern_mm / triton_mm / cpp_gemm ...
  -> autotune_select_algorithm
  -> template or extern IR

aten.add, aten.relu
  -> pointwise lowering
  -> TensorBox / Pointwise IR

Scheduler
  -> 判断 add/relu 是否能作为 epilogue
  -> template node + epilogue fusion

Codegen
  -> backend.codegen_template 或 codegen_extern_call
```

真实情况可能退化成：

```text
extern matmul kernel
pointwise add/relu fused kernel
```

也可能是：

```text
matmul template with epilogue
```

要判断真实路径，最可靠的方法是看 `TORCH_LOGS=output_code` 和 debug graph。

## 一个最小 PyTorch 示例

```python
import torch

def f(x, w, b):
    return torch.relu(x @ w + b)

device = "cuda" if torch.cuda.is_available() else "cpu"
x = torch.randn(128, 256, device=device)
w = torch.randn(256, 512, device=device)
b = torch.randn(512, device=device)

compiled_f = torch.compile(f, mode="max-autotune")
compiled_f(x, w, b)
```

调试：

```bash
TORCH_LOGS=output_code python mm.py
```

可进一步尝试：

```bash
TORCH_LOGS=+inductor,output_code python mm.py
```

日志量会显著增加。

## 编译前后发生了什么

简化流程：

```text
FX: aten.mm / aten.addmm
  -> kernel/mm.py lowering
  -> 构造候选算法
  -> select_algorithm 选择
  -> 可能生成 template buffer
  -> Scheduler 处理 epilogue/fusion 边界
  -> Codegen 输出调用
```

更贴近源码：

```text
GraphLowering.call_function(aten.mm)
  -> lowerings[aten.mm]
  -> tuned_mm
     -> mm_args
     -> choices.append(...)
     -> should_fallback_to_aten?
     -> autotune_select_algorithm
        -> AlgorithmSelectorCache.__call__
        -> precompile choices
        -> benchmark choices
        -> selected_choice.output_node()
  -> Scheduler sees TemplateBuffer / ExternKernel-like node
  -> Scheduler._codegen
     -> codegen_template or codegen_extern_call
```

如果选择外部库，生成代码中可能是 extern call；如果选择 Triton/C++ template，可能看到生成 kernel。

## TorchInductor 内部大致发生了什么

以 `x @ w + b` 为例：

1. Dynamo/AOT 生成包含 `aten.mm` 或 `aten.addmm` 的图。
2. Inductor lowering 进入 `kernel/mm.py`。
3. 根据 dtype、device、shape、layout 构建候选。
4. `select_algorithm.py` 负责选择或 autotune。
5. 后续 pointwise epilogue 可能融合进 template，也可能成为独立 kernel。

读源码时有一个好用的定位方法：

```bash
TORCH_LOGS=+inductor python mm.py
```

然后搜索日志里的：

```text
Tuned aten.mm
Tuned aten.addmm
Max autotune selects from ...
selected choice
```

这些日志可以反推出 `tuned_mm` 构造了多少 choices，以及 `AlgorithmSelectorCache` 最后选了谁。

卷积路径类似但更复杂。本环境中：

```text
torch/_inductor/kernel/conv.py
```

注册 `aten.convolution` lowering。卷积可能走外部库、Triton、布局优化或其他后端。

## 关键源码入口

```text
/usr/local/lib/python3.11/site-packages/torch/_inductor/kernel/mm.py
/usr/local/lib/python3.11/site-packages/torch/_inductor/kernel/bmm.py
/usr/local/lib/python3.11/site-packages/torch/_inductor/kernel/conv.py
/usr/local/lib/python3.11/site-packages/torch/_inductor/select_algorithm.py
/usr/local/lib/python3.11/site-packages/torch/_inductor/codegen/common.py
/usr/local/lib/python3.11/site-packages/torch/_inductor/codegen/cpp_gemm_template.py
/usr/local/lib/python3.11/site-packages/torch/_inductor/codegen/triton.py
```

和上下章的连接：

```text
第 10 章:
  普通 op 的 lowering 多在 lowering.py，pointwise/reduction 直接构造 loop IR。

本章:
  matmul/conv lowering 构造候选实现，并通过 select_algorithm 选择。

第 9 章:
  Scheduler 看到 template/extern 节点后决定 fusion 和 codegen 分派。

第 12 章:
  template/extern/fused node 最终交给 Triton/C++/wrapper/codecache。
```

建议搜索：

```bash
rg -n "def tuned_mm|def tuned_addmm" torch/_inductor/kernel/mm.py
rg -n "class TritonTemplate|ChoiceCaller|autotune" torch/_inductor/select_algorithm.py
```

## 常见误区

### matmul fusion 就是把三层 for 循环和 relu 写一起

高性能 matmul 远比三层循环复杂。tile、vectorization、tensor core、cache blocking 都很关键。

### `max-autotune` 总是适合

它可能提高稳定运行性能，但会增加编译/调优时间。

### CPU 和 GPU matmul 策略一样

不一样。CPU 侧可能涉及 C++ GEMM template、向量化、线程并行和外部库；GPU 侧可能涉及 Triton、CUDA、cuBLAS 等路径。

## 小结

matmul 和 conv 是 Inductor 中最重的算子族之一。它们通过专门 lowering、模板和算法选择系统处理。理解它们时，不要只看 `lowering.py`，还要看 `kernel/` 和 `select_algorithm.py`。

## 思考题或练习

1. 对 `x @ w`、`x @ w + b`、`relu(x @ w + b)` 分别查看生成代码。
2. 尝试 `mode="default"` 和 `mode="max-autotune"`，比较编译时间。
3. 搜索 `epilogue_fusion` 配置，理解它和 matmul template 的关系。

## 本章需要人工核查的技术点

- matmul/conv 的候选实现高度依赖硬件、dtype、shape 和配置。
- 外部库选择和 template 支持范围应以当前源码和运行日志为准。
