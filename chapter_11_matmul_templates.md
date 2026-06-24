# 第 11 章：Matmul、Conv、模板与算法选择

## 本章目标

本章解释为什么矩阵乘法和卷积不能简单套用 pointwise fusion 思路，以及 Inductor 如何在外部库、Triton template、C++ template 和 autotune 之间选择。

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

### `select_algorithm.py`

`torch/_inductor/select_algorithm.py` 负责算法选择和 autotune 相关逻辑。它会组织候选实现，benchmark 或基于启发式选择。

候选可能包括：

- ATen/external kernel。
- Triton template。
- C++ GEMM template。
- 平台特定后端。

### Epilogue fusion

很多模型中不是单独 matmul，而是：

```python
torch.relu(x @ w + b)
```

`+ b` 和 `relu` 是 matmul 的 epilogue。把 epilogue 融入 matmul kernel 可以减少额外读写。是否能融合取决于配置、模板能力和具体图。

### Autotune

Autotune 的核心思想：对多个候选 kernel 或参数配置实际测量，选择最快的。相关配置可能由 `mode="max-autotune"` 或 options 启用。

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

如果选择外部库，生成代码中可能是 extern call；如果选择 Triton/C++ template，可能看到生成 kernel。

## TorchInductor 内部大致发生了什么

以 `x @ w + b` 为例：

1. Dynamo/AOT 生成包含 `aten.mm` 或 `aten.addmm` 的图。
2. Inductor lowering 进入 `kernel/mm.py`。
3. 根据 dtype、device、shape、layout 构建候选。
4. `select_algorithm.py` 负责选择或 autotune。
5. 后续 pointwise epilogue 可能融合进 template，也可能成为独立 kernel。

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

