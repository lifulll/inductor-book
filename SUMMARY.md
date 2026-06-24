# 目录

- [写作方案](./00_writing_plan.md)
- [第 1 章：为什么 PyTorch 需要 TorchInductor？](./chapter_01_why_torchinductor.md)
- [第 2 章：从 Eager Mode 到 Graph Mode](./chapter_02_eager_to_graph.md)
- [第 3 章：`torch.compile` 的入口与参数](./chapter_03_torch_compile_entry.md)
- [第 4 章：TorchDynamo：从 Python Frame 到 FX Graph](./chapter_04_torchdynamo.md)
- [第 5 章：FX Graph：Inductor 接收的计算语言](./chapter_05_fx_graph.md)
- [第 6 章：AOTAutograd：训练图如何进入 Inductor](./chapter_06_aot_autograd.md)
- [第 7 章：`compile_fx.py`：Inductor 编译入口](./chapter_07_compile_fx.md)
- [第 8 章：GraphLowering、IR、Buffer 与 Layout](./chapter_08_graph_lowering_ir.md)
- [第 9 章：Scheduler、依赖分析与 Fusion](./chapter_09_scheduler_fusion.md)
- [第 10 章：Pointwise、Reduction、Softmax 与常见 Lowering](./chapter_10_ops_lowering.md)
- [第 11 章：Matmul、Conv、模板与算法选择](./chapter_11_matmul_templates.md)
- [第 12 章：GPU 与 CPU 代码生成后端](./chapter_12_backends_codegen.md)
- [第 13 章：缓存、运行时、调试与源码阅读路线](./chapter_13_debugging_source_map.md)
- 附录 A：常用调试环境变量与日志选项
- 附录 B：术语表
- 附录 C：源码阅读地图
