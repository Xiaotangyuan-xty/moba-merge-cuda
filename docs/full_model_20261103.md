# Qwen2.5-0.5B 完整模型集成、质量与 Graph 实验

后续已完成确定顺序合并与混合策略实验，见 [稳定性与长上下文报告](stable_quality_20261103.md)。下文保留本阶段原始配置、数值问题和结论。

日期：2026-11-03。模型为 **Qwen/Qwen2.5-0.5B base**，没有稀疏训练或微调。单 A40、BF16、batch=1、固定请求容量、teacher-forced decode。实现位于 `examples/full_qwen2.py`，属于实验性集成。

本阶段完成完整模型 eager、Attention core Graph、完整 GPU 前向 Graph，以及可选 packed QKV。**主要有效收益来自完整模型 Graph。当前实验不支持“MoBA 在该模型上无损加速”的结论。** 使用 FA2 自动 split-KV 的 dense Graph 后，真正稀疏的配置均没有超过所测 dense Graph；直接替换稀疏 Attention 还会显著损害质量。

## 模型与数据固定

- 模型 revision：`060db6499f32faf8b98477b0a26969ef7d8b9987`。
- 权重 SHA256：`88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342`，988097824 bytes。
- 24 层，hidden=896，Hq=14/Hkv=2，D=64，intermediate=4864，词表 151936，RoPE theta=1e6，关闭 sliding window，权重不修改。
- 语料为 [PyTorch examples 固定提交中的 WikiText-2 tokenized test](https://raw.githubusercontent.com/pytorch/examples/d5678bc8ac0cdd79dbd5e44d4130271018bcec4e/word_language_model/data/wikitext-2/test.txt)，**不是 raw 版本，也不是完整测试集评估**。
- 语料 SHA256：`d790b833ef8cf03a90db7bf1271b7520b83c45ce07ba3c1a9699df81e239eca0`。
- torch 2.4.0+cu121、Transformers 4.49.0、flash-attn 2.6.3。复用现有环境，未安装新增依赖。
- 正式实验代码提交 `8a488700e7b8f4e04a1533eea73c168976db1680`；实验均为 clean checkout。最终文档提交可以晚于实验代码提交。

## 集成与正确性

完整前向包含 embedding、RoPE、24 层 RMSNorm/QKV/Attention/o_proj/residual/MLP、最终 norm 和 LM head。每层使用本项目的 KV cache 和 prepared decode。Graph 捕获这条 GPU 前向路径；CPU 页容量检查、增量页表计划、token 输入写入与 CPU 长度提交仍在 Graph 外。

Dense 对照在同一套 cache 管理下调用 `flash_attn_with_kvcache`。仅在新建的 batch=1 请求、物理页按逻辑顺序分配时使用其连续视图；初始化和页增量都有相应检查。这不构成“FA2 控制路径支持任意碎片化分页”的声明。真正稀疏的 decode 仍直接读取页表。

主正确性对照为原生 HF FA2，另外保留原生 SDPA 与 FA2 的差异。预设阈值为 relative logits L2≤0.02、KL≤0.002、maxabs≤0.5，要求 finite，不因失败放宽阈值。

- 真实模型：128-token prefix +12 个固定 continuation token。默认 dense eager、dense Graph 和全部块必选的 MoBA 快路径，在该用例中与 HF FA2 logits 逐位一致。不能把此局部结果外推为所有输入逐位一致。
- packed QKV 候选在同一用例通过阈值，但最大 relative L2=0.01869、KL≈0.00199990，已接近门槛。它是 opt-in 候选，未成为默认实现。
- 真正稀疏的完整模型 Graph：从同一份逐层 prefill KV 起点出发，prefix=1023、topk=4、连续 20 步，跨页和 chunk 边界。普通及 packed 两组的 eager/Graph logits 和选块逐位一致，长度一致，capture 不推进可见长度。
- 两层随机模型补充检查 eager/Attention Graph/完整 Graph、packed 与非 packed，以及容量错误、非法 token、capture 状态恢复、跨页和跨 chunk 的输出/选块/KV 一致性。

最终 `pytest tests/`：**531 passed、2 skipped**（双卡测试超出单卡授权范围），用时 36.99 秒，测试提交与正式实验相同。

失败记录也属于结论边界：早期 Triton dense decode 在真实模型首个新 token 上 relative L2=0.02525、KL=0.002586，未通过门槛。因此完整模型默认 dense 控制改为 FA2；没有宣称该 Triton 路径已修复。另一次将两次独立 sparse prefill 直接对比的实验，relative L2=0.02718、KL=0.004667。独立 sparse prefill 的数值和后续离散路由可能随运行变化，尚未证明根因；它不能直接用于隔离 Graph 误差。共享 KV 的 Graph 测试只证明执行路径等价，不提供稀疏模型跨运行逐位复现保证。

## 质量：不做稀疏适配的代价

预先固定前 4 个连续 2048-token 块，无重叠，不按结果选样本。每块预测 2047 个目标，总计 8188 个目标。先汇总 NLL，再除以总 token 数并取 exp；不平均块 PPL。独立计算每块末尾 32 个 token 的 cache continuation loss，总计 128 个目标，**两种分数不能混合**。

正式主轮数据如下，复测结果与全部逐块记录在 `docs/data/full_model_20261103_*.json` 及汇总中保留。

| 方法 | 8188 targets PPL | 相对 HF FA2 ΔNLL | 128 continuation targets PPL |
|---|---:|---:|---:|
| HF SDPA | 12.4027 | +0.00008 | 7.6172 |
| HF FA2 | 12.4017 | 0 | 7.6484 |
| 自有 dense | 12.4017 | 0 | 7.6484 |
| MoBA topk=2 | 63.0021 | +1.62534 | 47.7952 |
| MoBA topk=4 | 24.5611 | +0.68333 | 23.4610 |
| MoBA topk=8 | 14.7214 | +0.17147 | 11.4226 |
| MoBA topk=4 packed | 24.6300 | +0.68613 | 23.6939 |

同一代码的第二个独立进程复测：topk=4 的 PPL 为 24.5689，topk=8 为 14.6801，packed k4 为 24.6375；dense 与 k2 分数相同。较小的 128-target continuation 样本波动更明显：k4 两轮为 23.4610/24.2432，packed k4 为 23.6939/22.6193。因此不能把 packed 与普通 k4 的细小质量差异解读为改进，也不提供跨运行逐位一致的稀疏质量保证。初轮固定单 split 的完整质量记录也在 historical 数据中保留。

稀疏方法从 prefill 开始替换 Attention，因此这些质量损失同时包含 prefill 和 decode 的影响，不能单独归因于 decode 或 CUDA 合并实现。当前测试不能证明稀疏训练后的质量，也不能代表其他模型或语料。

补充六个固定 synthetic passkey 用例：512/2048/4096 token ×针位置 25%/75%。记录正确答案 NLL、16-token greedy 输出及首个四位数字是否正确；保留全部用例。它是小样本诊断，不是通用 QA benchmark，也不测吞吐或生成时延。原生 dense 与自有 dense 均 6/6，topk=2/4/8 分别为 0/6、3/6、6/6；不能用个别成功样例替代质量评估。

## 计时与公平对照

相同模型、相同 prefix、相同 teacher-forced token，候选与原生 HF FA2 AB/BA 交替。每行预热 5 步，3 轮×10 对样本，先取每轮中位数，再取跨轮中位数；speedup 取轮配对比值的中位数。两个独立进程，共 3 长度×6 方法×2 轮运行 =36 行。全部原始配对样本和退化行公开。

计时包含完整模型 GPU 前向、CPU cache 管理、token 写入和末端同步；不含采样、tokenization、网络、排队或 continuous batching。每次 step 返回全部词表 logits。Graph 使用固定容量而非固定当前有效长度。

必须区分三种比较：

1. 自有 dense Graph / HF eager FA2：包括 Graph、模型封装和 cache 管理差异，不能称为 MoBA 算法收益。
2. Sparse Graph / dense Graph：用于判断稀疏在已优化执行路径上是否仍有价值。本实验未发现优势。
3. packed / 非 packed：只解释 QKV 合并投影候选，仍需计入质量变化与额外权重副本。

**纠正 dense 基线**：初轮固定 `num_splits=1`，8192 长度 dense Graph 约 9.13 ms；允许 FA2 自动 split-KV 后约 4.38 ms。使用固定单 split 会夸大稀疏的优势，因此正式结论使用 `num_splits=0`。初轮全部 36 行作为历史 CSV 保留，不混入正式加速统计。

当前正式结果中，dense Graph 约 4.13/4.17/4.38 ms（256/2048/8192）；HF eager 约 19 ms。稀疏 Graph 在 2048/8192 下，topk=2/4/8 约 4.41/4.79/5.45 ms。两轮完整数据、配对 speedup 和退化行见 `full_model_20261103_latency_all_rows.csv`。

下表为两进程各自行中位数的均值，供比较配置；原始配对加速比仍以 CSV 每行为准。不同候选与 dense Graph 之间未做直接配对，不对不到 1% 的差异作显著性结论。

| 方法 | prefix=256 ms | prefix=2048 ms | prefix=8192 ms |
|---|---:|---:|---:|
| 自有 dense eager | 21.628 | 21.582 | 21.341 |
| 自有 dense Graph | 4.130 | 4.171 | 4.378 |
| MoBA k2 Graph | 4.419 | 4.412 | 4.416 |
| MoBA k4 Graph | 4.130† | 4.789 | 4.797 |
| MoBA k8 Graph | 4.131† | 5.453 | 5.470 |
| MoBA k4 Graph packed | 3.923† | 4.588 | 4.588 |

† 全部块必选，走 dense FA2。自有 dense eager 在全部六行均慢于 HF eager（0.878–0.884×），没有删除这些退化结果。自有 dense Graph 对 HF eager 为 4.276–4.606×；这是三个长度、两个进程的范围，不外推到其他模型/硬件。

256-token 的 topk=4/8 在容量范围内全部块必选，明确记录 `semantic_dense_fastpath=true`；这些行走 dense FA2，不能算稀疏加速。topk=1 同样保留上游全量 causal fallback 语义。

初始化、prefill、prepare/capture 和首次 decode 分开记录。每行只有一次准备成本观测，可能包含 JIT、库初始化或冷启动，只作诊断；不是经过稳健计时的 TTFT。汇总中的 capture/request break-even 是以稳态节省估算的摊销点，未通过完整变长生成请求验证，不能宣称实际服务延迟改善相同倍数。对真实端到端请求，当前自有 prefill 也存在更高开销。

Dense Graph 的 prepare/capture 单次观测约 127–165 ms；将 init、prefill 差额与 capture 一并计入，估算需约 10–14 个 decode step 才摊平对 HF eager 的额外准备成本。这个估算不包含首次 decode 偏离稳态的额外项，没有测量真实请求重用 Graph 的策略。

## Profiling 与融合决策

prefix=1024，warmup 4 步，profile 3 步。CUDA Runtime launch 事件包括 kernel 与 Graph launch；不等同于全部 driver API，也不用于正式延迟统计。每步平均如下：

| 方法 | CUDA kernel 数 | Runtime launch 事件数 | kernel duration 合计 ms |
|---|---:|---:|---:|
| MoBA k4 eager | 1125 | 981 | 4.991 |
| Attention core Graph | 1173 | 1029 | 5.015 |
| 完整 GPU 前向 Graph | 1127 | 4 | 4.677 |
| 完整 Graph +packed QKV | 1031 | 4 | 4.466 |

完整 Graph 每步剩 1 个 `cudaGraphLaunch` 和 3 个 `cudaLaunchKernel`，CPU 页管理仍在图外。仅包 Attention 的 Graph 没有减少全模型 host launch 事件，因而本阶段扩展到 embedding、norm、投影、MLP 与 LM head。Graph 主要消除 dispatch 间隙，没有把一千多个 GPU kernel 融合成一个 kernel。

packed QKV 拼接原始 Q/K/V 权重和 bias，用一次线性投影替代三次投影，三种输出仍按原顺序切分；每步少 96 个 kernel，GPU duration 小幅下降。这属于框架线性层合并，**不是新的手写融合 CUDA kernel**。保留为可选候选，因为存在额外权重副本及 BF16 数值变化，稀疏完整模型最终仍慢于 dense Graph。

trace 中剩余较大 GPU 时间在模型 GEMM/GEMV、Attention、逐元素/归约与 top-k；下一阶段应以质量预算和完整请求成本为约束，再决定是否优化 norm/RoPE/cache 写入，或使用经过稀疏适配的模型。当前证据不支持直接把 MoBA 设为真实模型默认 Attention。

## 复现入口

任何 GPU 命令均需当次授权和共享卡检查。示例中的路径、GPU UUID、预算需在私有 `scripts/env.local.sh` 中按实际环境设置；模型和语料路径由调用方提供。

```bash
source scripts/env.sh
# CPU 下载入口：固定 revision、串行、校验；按需运行，不重新下载已有权重。
python -m scripts.download_full_model_assets --help

# GPU：先检查显存和进程，再经 guard 启动；每次 tag 和 out 必须唯一。
python scripts/run_shared_gpu_batch.py --tag full_validate_01 -- \
  python -m scripts.evaluate_full_model --model "$MODEL_DIR" --corpus "$CORPUS_FILE" \
  --stage validate --out results/full_validate_01.json
# 其他 stage：profile / quality / latency / retrieval；latency 用两个独立进程。

# CPU：从公开的完整样本重新汇总质量和全部 36 行计时。
python -m scripts.summarize_full_model \
  --quality docs/data/full_model_20261103_quality.json \
  --latency docs/data/full_model_20261103_latency_a.json docs/data/full_model_20261103_latency_b.json \
  --out-dir results/full_model_summary_repro
```

公开数据不含机器私有路径、GPU UUID、其他用户进程信息、权重或语料原文。私有原始日志、资源账本、失败与中断记录仍保留。共享卡所有 GPU 任务串行运行，监测到外部占用时只停止本项目子进程，不切换未授权卡。本批含早期失败的进程累计墙钟 473.84 秒，监控到的设备显存峰值 3033 MiB；低于 1200 秒/8192 MiB 批次上限。CPU 线程不超过 2，未操作其他用户进程，结束后 GPU1 显存为 0 MiB、无本项目进程。
