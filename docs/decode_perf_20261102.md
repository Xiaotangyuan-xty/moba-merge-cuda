# Decode 路由与 host 开销优化

日期：2026-11-02。代码在 Mac 修改、提交并上传 GitHub，服务器仅同步已提交代码并验证。本文的性能基线是本项目冻结的 cached decode v1，不是 Moonshot prefill，也不是完整模型推理。

## 实现与决策

1. **按请求组织路由。** v1 先构造“所有请求历史块 × 所有 query × Hq”的 score，再屏蔽其他请求。现在每个 query 只读取所属序列的块均值，8 个候选块共用一个 Triton program，topk 沿连续的最后一维计算。等长 batch 每条请求一个 query 时，score 数量从约 B²×Hq×历史块数降为 B×Hq×历史块数；不复制全长 KV。
2. **复用已验证输入。** 公开入口保留严格 shape/device/dtype/finite/alias 检查；组合接口只验证 Q/K/V 各一次，再调用内部已验证路径。反例测试确认 NaN/Inf 仍在发布追加状态前报错。
3. **减少 metadata 上传。** 页表保留二维转换路径；长度、token IDs/位置和刚完成的 chunk IDs 打包上传。append 使用两次 metadata H2D；组合接口直接复用追加阶段的 GPU query IDs/位置。独立 attend 的 query metadata 合为一次 H2D。

仍保留 v1 的返回槽位宽度与 -1 padding、topk=1 dense fallback、绝对位置因果与请求隔离。浮点 tie 的非稳定选块边界不变，不承诺所有输入与 v1 逐位相同。

一次中间实现把整张页表也展开成 Python 列表，造成大 B/L 的追加接口回退；完整矩阵发现这一问题后，最终实现保留二维页表转换，仅打包小向量。此前版本的原始记录保留，不混入最终结果。

## 测量方法

- 冻结基线：`add0a49fdccfca52e5873fbdad0893e96777cf89`。`benchmarks/_decode_v1.py` 只调整 import，kernel 原样存于 `_decode_v1_kernels.py`，已与该提交核对；不是用完整 prefill 重算模拟 decode。
- 矩阵固定在 `configs/decode_perf.json`：18 组 BF16 GQA（B=1/4/16，L=2048/8192/32768，D=64/128，Hq/Hkv=32/8），6 组 BF16 MQA（D=128，B=1/4/16，L=2048/32768，32/1），4 组 FP16 GQA。共 28 组；page=16、chunk=128、topk=4。
- 两个独立进程，seed=401/402；每个实现 5 次预热，3 轮×10 配对样本，样本及轮次交替 AB/BA。两侧同 Q/K/V、初始缓存和追加次数，严格校验均开启；初始化、随机生成、JIT 和 profiler 不计入正式延迟。
- 分别测 `attend` 与 `append_and_attend`。前者缓存不变；后者每次真实追加一批 token，两侧推进相同绝对位置。固定的新 Q/K/V 在计时样本间复用，避免随机生成成本混入。最后再验证两侧选块、输出与 LSE。
- 主指标为同步 CPU wall latency；先算各轮两侧延迟中位数之比，再取跨轮中位数。CUDA event 是 stream 区间，包含 host 调度造成的空隙，不称为纯 kernel 时间。几何平均包含全部配置。
- 数学等价的 attention 子阶段对照：相同固定选块，GPU gather 后调用 `flash_attn_func`；计入 gather，排除共同路由。矩阵中选块均完整，已省去 padding 排序。它是一个工程参考，未声称是最优稀疏 FA 实现。
- dense `flash_attn_with_kvcache` 单列，读取全部历史 token，直接视图复用物理连续的等长 fixture；不额外重建整份缓存。它与 MoBA 的 token 集合不同，不作为同语义 kernel 加速比。
- 测量用单 A40、共享服务器、nice=10、线程上限 2；没有锁 GPU 时钟或改变他人任务。页表路径参与计时，但性能 fixture 未模拟碎片化页与动态请求调度。

## 最终结果

最终实现/统一回归/两轮测量代码：`be0f30a68d0ee225f3b21b0b2e95a99173fcb5f5`。后续仅补充文档与统计表。

| 独立进程 | attend 相对 v1 | attend 几何平均 | append_and_attend 相对 v1 | append 几何平均 |
| --- | --- | --- | --- | --- |
| seed 401 | 1.025–4.553× | 1.320× | 1.466–2.281× | 1.647× |
| seed 402 | 1.017–4.580× | 1.322× | 1.482–2.381× | 1.675× |

两轮全部 56 行等权几何平均分别为 **1.321× / 1.661×**。两个完整矩阵的点估计都未出现相对 v1 的退化行，但最小 attend 收益只有约 1.7%，不能据此声称每个配置都有统计显著收益。

**dense FA 在两轮中都快于当前 attend 的 22/28 个配置。** 因此不能把相对自有 v1 的改善写成“普遍优于 FlashAttention”或完整模型推理加速。尤其小 batch/MQA 的性能仍需要改善。

完整 56 行见 [全部配置 CSV](data/decode_perf_20261102_all_rows.csv)，机器可读汇总见 [summary JSON](data/decode_perf_20261102_summary.json)。下表仅用于阅读，取 seed 401 的代表行；延迟单位 ms，速度比按逐轮统计，可能与两个聚合延迟直接相除略有差别。

| ID / B / L / D / KV头 | v1 attend | 当前 attend | 相对 v1 | append 相对 v1 | dense FA（不同选块集合） |
| --- | --- | --- | --- | --- | --- |
| P01 / 1 / 2K / 64 / 8 | 0.529 | 0.512 | 1.037× | 1.582× | 0.068 |
| P06 / 1 / 32K / 128 / 8 | 0.552 | 0.529 | 1.044× | 1.550× | 0.293 |
| P10 / 4 / 8K / 128 / 8 | 0.580 | 0.559 | 1.036× | 1.534× | 0.288 |
| P18 / 16 / 32K / 128 / 8 | 2.987 | 0.746 | 4.006× | 2.274× | 3.728 |
| M06 / 16 / 32K / 128 / 1 | 2.949 | 0.693 | 4.259× | 2.281× | 0.506 |
| F04 / 16 / 32K / 128 / 8，FP16 | 2.970 | 0.704 | 4.218× | 2.273× | 3.728 |

Profiler 诊断中，P18 路由 kernel 三次平均由 **2263.38 µs 降到 24.85 µs**；P01 仅由 2.23 µs 降到 1.67 µs。它解释了收益随 B/L 变化的原因，不把局部 kernel 比值用作接口或模型加速比。Profiler 的 CPU scope 包含等待及仪器开销，不将其占比当成精确的无仪器阶段占比。

## 验收与资源记录

- 最终完整回归 **485 通过、2 跳过**（487 collected），pytest 73.58 秒；两个跳过项仅为既有双卡设备绑定，本批只授权单 GPU1。唯一 warning 为既有诊断路径的 `index_reduce` beta API 提示。67 个被测源码/配置文件的 SHA256 已与 Mac 核对一致。
- 最终日志：`logs/decode_perf_20261102/compact_full_suite.log`；JUnit：`results/decode_perf_20261102/compact_full_suite.xml`；正式样本：`compact_r1.jsonl` / `compact_r2.jsonl`；诊断：`compact_profiles.jsonl` 和独立的 `compact_profiles_traces/`。
- 本批 18 次 GPU 进程运行，累计墙钟 **887.28 秒（14 分 47 秒）**，在获准的 1,200 秒内。包括中间候选、编译等待、回归与废弃预试；中间版本不合并进最终性能统计。
- **首次预试因输入构造超线被守护脚本终止自身进程组**：1 秒采样记录 9,053 MiB，超过本批 8,192 MiB 监控线。改为分块填充 fixture，并把 PyTorch allocator 限至最多 6 GiB 后重试。其后成功任务的采样峰值最高 4,595 MiB，最终两轮矩阵均为 4,463 MiB。采样值不是精确瞬时峰值，软件监控不等于硬性显存隔离。
- 每次任务前检查显存与进程并绑定物理 GPU1；没有他人进入目标卡的冲突记录，没有使用其他 GPU 或改动他人进程。最终 GPU1 为 0 MiB、无本批 GPU 进程。
- 复用现有 Torch/FlashAttention/Triton 环境，没有新增依赖或模型下载。未运行 compute-sanitizer，未做完整模型质量/吞吐测量。

## 正确性与复现

原有独立 FP64 输出/LSE、原始 KV 精确读回、路由排名、因果和请求隔离测试继续保留。新增多 seed BF16/FP16 的 ragged 请求路由对照、非有限输入拒绝，以及 metadata 上传/索引复用检查。统计工具的 CPU 测试验证缺样本报错、逐轮比值口径和退化行保留。

复现前按 README 取得当批 GPU 许可，配置私有 `scripts/env.local.sh` 中的物理 GPU1 UUID、批次 ID、时间和显存限额。每次使用新 tag/输出路径，保留累计预算记录。

```bash
source scripts/env.sh
python scripts/run_shared_gpu_batch.py --tag decode_recheck_401 -- \
  python -m benchmarks.benchmark_decode --seed 401 \
  --out "results/${MOBA_BATCH_ID}/decode_recheck_401.jsonl"
python scripts/run_shared_gpu_batch.py --tag decode_recheck_402 -- \
  python -m benchmarks.benchmark_decode --seed 402 \
  --out "results/${MOBA_BATCH_ID}/decode_recheck_402.jsonl"
# CPU 汇总；缺少任何配置/样本会报错，不丢弃失败或退化行。
python scripts/summarize_decode.py \
  "results/${MOBA_BATCH_ID}/decode_recheck_401.jsonl" \
  "results/${MOBA_BATCH_ID}/decode_recheck_402.jsonl" \
  --out-dir "results/${MOBA_BATCH_ID}/summary_recheck"
```

原始样本、资源记录和 trace 留在忽略的 `results/`，机器配置留在忽略的 `env.local.sh`；公开统计表不包含设备 UUID、服务器私有路径或其他人的进程信息。

## 适用边界与后续

以上只覆盖本项目 cache API，不包含 QKV 投影、RoPE、模型其他层或生成服务。正确性支持的 D=1..256 不等于这些维度都有性能证据；本矩阵只测 D=64/128，MHA、其他 page/chunk/topk、碎片化页和多 GPU 都不在性能外推范围。

小 batch 下 host 检查、CPU 页分配计划、页表重建与 kernel dispatch 仍占明显开销；部分配置慢于 dense FA。没有“普遍优于 FlashAttention”或“完整模型吞吐提升”的结论。后续优先考虑 metadata buffer 预分配、页表增量更新，以及固定布局下消除同步，再验证 CUDA Graph 的可行性。新 cache 路径仍 inference-only、单 stream，未接入完整模型，未实现 Graph 或 backward。
