# 已知限制与适用边界（更新至 2026-11-03）

## 当前支持范围（2026-11-03）

`opt_gqa_cuda` 支持 MHA/GQA/MQA prefill，Hq 必须为 Hkv 的正整数倍，
D=8..256 且为 8 的倍数，BF16/FP16。完整契约和测试见 `coverage_20261101.md`。
后续已实现连续/分页 KV cache 与增量 decode，详见 `decode_paged_20261101.md`。
decode 独立支持 D=1..256 的整数；只做 inference，不支持 cache backward、
跨 stream、共享前缀页/COW、滑动窗口或自动扩容。普通 cache API 拒绝外部 capture；PreparedDecode 支持固定活动请求的 GPU 计算路径 Graph，CPU 管理不在捕获范围。接口、真实权重单层验证与算子性能见 `prepared_decode_20261102.md`。原生 GQA prefill 尚无新增正式性能结论；
decode 旧阶段相对冻结 cache v1 的 28 配置性能取证见 `decode_perf_20261102.md`。最新 Prepared Graph trusted 模式仍有 14–15/28 配置慢于 dense FA core，不构成普遍加速结论；单层数值对齐不等于完整模型质量验证。

新增实验性 `examples/full_qwen2.py` 集成完整 Qwen2.5-0.5B base，支持完整 GPU 前向 Graph，包含 24 层 MLP 与 LM head；CPU 页管理和 token staging 仍在图外。仅 batch=1、固定容量、单 stream、teacher-forced，未验证动态 batching、padding、调度或服务。FA2 dense 对照使用新建单请求的连续物理页视图，不能用于任意碎片化 cache；默认允许 FA2 自动 split-KV。模型内部有限 Q/K/V 为显式假设。

最新支持显式 `deterministic_merge=True`、`route_precision='fp32'`、`stable_routes=True`，以及 `dense_layers`/`prefill_mode` 实验性模型策略。warp 内排序固定合并次序；FP32 均值与完全同分时优先更小 chunk 的规则构成新策略，可能改变上游路由。默认行为保持兼容，旧版选块逐位一致声明不推广到这些新策略。Qwen2.5-0.5B/1.5B 当前重复 prefill 实验不再漂移，不保证跨平台、矩阵形状变化、量化或输入扰动后的路由一致性。

新增独立 train/valid 校准的混合策略与 1.5B 完整模型测试。0.5B/1.5B held-out prefill PPL 分别恢复至12.86/9.78，dense为12.40/9.19；部分测试指标仍未达到校准时的ΔNLL≤0.05目标。2K校准策略的质量预算未泛化到长上下文。0.5B约32K可运行；1.5B仅确认至16K，约32K在7GiB allocator/8GiB设备监控限制下OOM。MLP分块候选因真实模型数值验证失败被撤回。48行长上下文比较不构成普遍快于dense Graph的证据，详见 `stable_quality_20261103.md`。

前一完整模型阶段质量仅评估 WikiText-2 tokenized test 的前 4×2048-token 块（8188 个目标）、128 个 continuation 目标与六个 synthetic passkey。该阶段正式 36 行延迟没有证明真正稀疏路径优于所测 dense Graph。相对 HF eager 的约 4.3–4.6× 是 dense Graph/集成收益，不是稀疏算法或完整请求收益。packed QKV 仅为 opt-in，数值差异接近短用例门槛；默认旧 sparse prefill 不保证跨运行逐位复现。早期 Triton dense 完整模型对照未过数值门槛，未宣称已修复。详见 `full_model_20261103.md`，其中保留负面结果和适用边界；新阶段的512-target continuation及长上下文诊断单列。

## 历史 MHA 基线支持范围（方案 2.4，56eedcd）

- 单张 A40（CC 8.6）；packed varlen `[T,H,D]`；MHA（Q/K/V head 数相同）；causal、dropout=0
- head_dim 64/128；BF16 与 FP16（FP16 验证于 F03S/F07M/F12L 三代表配置）
- K=1 走官方 fallback（全量 causal FA），行为逐位一致，不混入稀疏加速统计
- 明确不支持：原生 GQA/MQA、decode KV cache、分页 Attention、多 GPU、端到端模型质量、CUDA Graph 全图捕获。不支持输入显式报错（dtype/布局/空序列/非法 cu_seqlens 有测试覆盖）；子算子槽位容量溢出显式报错（CUDA/Triton 两后端均有测试）

## 性能结论的边界

1. **收益只在前向**。反向与官方逐行一致，耗时比实测 0.997–1.006（噪声内）。任何"前反向合计"数字都是前向收益被反向稀释后的结果。
2. **历史合并优化的主贡献为 CUDA 实现**（`src/moba_opt/ops/merge.cu` + `bindings.cpp`，nvcc 12.9、sm_86、无 fast-math、带设备绑定 `CUDAGuard`）；合并的 Triton 版（`merge_triton.py`）为验证期参考实现。新增直接分页 decode 使用独立的 Triton kernel（`paged_decode.py`），其相对自有 cache v1 的结果见单独性能报告，不与历史合并数字混用。
3. **收益幅度随形状变化**（统计口径：轮中位数的跨轮中位数，AB/BA 配对；修复后 harness 的正式取证运行）：CUDA 后端 fwd+bwd **1.062–1.129×**（12 核心 + 11 补充 + 冷元数据共 24 行，几何平均 **1.0875×**），fwd **1.123–1.455×**。H=1/小 T 配置收益最小；块越小、top-k 越大、head 越多，收益越大。
4. **历史加速数字不推广到性能矩阵之外的形状**：H=16/32、D≠64/128、更长序列不在该历史性能矩阵内；CH128 的更高收益不外推到其他 C。后续新增的形状正确性测试不构成这些形状的性能证据。
5. 原子建图的票据顺序不确定 → lane 级累加次序逐运行可变。口径：**保持数学语义、输出在明确容差内对齐**（FP32 合成用例重复运行实测在 atol 1e-6 / rtol 1e-5 内）；不提供逐位或逐 ulp 保证。官方 `index_add_` 原子序同样不确定。**选块集合与官方逐位一致**（离散路由，有测试）；这与输出数值接近是两个不同声明。
6. **显存**（修复测量泄漏后按方案 8.5 独立进程分别测量）：峰值显存与官方**持平**（C01 37/37、C08 479/478、C12 2010/2011 MiB，±1 MiB 噪声级）。不宣传省显存。
7. 冷元数据场景（每次新长度对象）：C08 上优化仍成立（CUDA 后端 fwd+bwd 1.088），但两侧都要付 metadata 重建与额外 host 同步成本。
8. V1 仪器开销实测 0.985–0.997×（record_function 无 profiler 时），诊断数字不作为性能结论。
9. **一次未复现的测试失败（如实记录）**：首次双卡全量套件运行时，两个 device-guard 测试失败，特征为跨卡拷贝后的输入张量（含 int64 mapping）在目标卡上呈全零——容量检查**正确拒绝**了在损坏输入上计算（安全机制按设计工作）。该现象在随后 4 次运行（隔离 ×1、组合 ×2、完整套件 ×2）均未复现；首次失败时宿主机负载很高。根因未定位（疑似该进程状态下的跨卡拷贝路径），不构成对合并 kernel 的否定——kernel 只在正确输入上被验证。若再次出现将沿传输路径排查。

## 正确性声明的准确口径

- 选块集合：与官方**逐位一致**（离散路由，路由代码逐行相同）。
- 完整前向输出：对官方在 BF16 舍入量级内对齐；另有 FP64 固定路由独立参考交叉验证（**仅输出**）。
- 完整梯度（dQ/dK/dV）：固定路由下**与官方实现对照**（多 seed 随机 VJP，官方口径）；合并原语 VJP 公式经 FP64 有限差分校验。不提供"独立 FP64 梯度验证"的说法。
- k=1 fallback 与官方调用同一函数，输出逐位一致。

## 统计口径

- 主指标：同进程同输入 AB/BA 配对、5 轮 ×30 计时样本 + 20 预热、**每轮中位数再取跨轮中位数**、配对 speedup；P99 不作为指标。终端打印与 `scripts/summarize_results.py` 同一口径。
- 暖元数据（同一不可变 cu_seqlens 对象）；冷元数据单列。
- kernel/stage 级数字仅用于解释机制，不作为完整 Attention 加速结论。
