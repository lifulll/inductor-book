# 走进 TorchInductor：从 `torch.compile` 到深度学习编译器

这是一本面向初学者的 TorchInductor 技术书初版草稿。

目标读者已经熟悉 Python 和基本 PyTorch 使用，但不要求具备编译器、CUDA、Triton 或深度学习框架底层实现经验。

## 当前文件

- [00_writing_plan.md](./00_writing_plan.md)：全书写作方案与详细目录
- [SUMMARY.md](./SUMMARY.md)：章节索引
- [chapter_01_why_torchinductor.md](./chapter_01_why_torchinductor.md)：第 1 章
- [chapter_02_eager_to_graph.md](./chapter_02_eager_to_graph.md)：第 2 章
- [chapter_03_torch_compile_entry.md](./chapter_03_torch_compile_entry.md)：第 3 章
- [chapter_04_torchdynamo.md](./chapter_04_torchdynamo.md)：第 4 章
- [chapter_05_fx_graph.md](./chapter_05_fx_graph.md)：第 5 章
- [chapter_06_aot_autograd.md](./chapter_06_aot_autograd.md)：第 6 章
- [chapter_07_compile_fx.md](./chapter_07_compile_fx.md)：第 7 章
- [chapter_08_graph_lowering_ir.md](./chapter_08_graph_lowering_ir.md)：第 8 章
- [chapter_09_scheduler_fusion.md](./chapter_09_scheduler_fusion.md)：第 9 章
- [chapter_10_ops_lowering.md](./chapter_10_ops_lowering.md)：第 10 章
- [chapter_11_matmul_templates.md](./chapter_11_matmul_templates.md)：第 11 章
- [chapter_12_backends_codegen.md](./chapter_12_backends_codegen.md)：第 12 章
- [chapter_13_debugging_source_map.md](./chapter_13_debugging_source_map.md)：第 13 章
- [assets/](./assets/)：正文配图 SVG

## 参考环境

本初版参考当前环境中的 PyTorch 源码：

```text
PyTorch: 2.7.1
源码根目录: /usr/local/lib/python3.11/site-packages/torch/
```

全书主线按以下调用链组织：

```text
torch.compile
  -> torch._dynamo.optimize
  -> FX GraphModule
  -> AOTAutograd
  -> torch._inductor.compile_fx
  -> GraphLowering
  -> Inductor IR
  -> Scheduler/Fusion
  -> Triton/C++ Codegen
  -> CodeCache/Runtime
```
