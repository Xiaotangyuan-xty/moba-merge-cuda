# 低延迟 PreparedDecode、计算路径 CUDA Graph 与实际权重层验证

日期：2026-11-02。三个阶段均完成：固定活动请求的准备/执行接口、GPU 计算路径 Graph、64K/128K 与真实 Qwen2 权重层验证。最终完整回归 **515 通过、2 个双卡测试跳过**。

## API 与生命周期

```python
# Q/K/V 已完成投影和位置编码；本例每步两条请求各追加一个 token。
cache.append(k_prefill, v_prefill, sequence_ids=[0, 1], cu_seqlens=cpu_cu)
with cache.prepare_decode([0, 1], num_q_heads=32, topk=4,
                          check_finite=True) as decoder:
    decoder.capture("step")       # 首次准备和 warmup 不计入单步延迟
    for q, k, v in projected_tokens:
        result = decoder.step(q, k, v, use_graph=True)
        saved_output = result.output.clone()  # 需跨步保留结果时复制
# close 后恢复普通 cache.append/attend/reset，用于更换请求与重建计划。
```

- 活动 `sequence_ids`、每步 token 数（每个活动请求一个）、Hq/Hkv、D、dtype、设备、stream、topk、mode 和容量固定。独立查询用 `decoder.attend(q)`，读取各活动请求最后一个位置；可先 `capture("attend")` 再 replay。
- 输入固定 shape、同设备同 dtype、完整 contiguous、无梯度。结构/容量/页池检查始终执行。默认 `check_finite=True` 同步检查 NaN/Inf；显式设为 False 时调用方保证有限输入，类似 FA 的输入契约。
- 输出、LSE、selected_blocks 是持久工作区视图，下一次调用覆盖。需要保留时 clone；固定 selected 槽位含 -1 padding，tie 不承诺与其他布局逐位相同。
- lease 期间 cache.lengths/page_table 可读，materialize 可诊断读回；普通 append/attend/reset 和第二个 prepare 明确拒绝。close 后一次性发布不可变 CPU 布局。更换活动集合、重置 slot 或形状变化需 close 后重新 prepare。
- q/k/v 固定输入 buffer 可由生产者直接填充，再传入 `decoder.q/k/v` 避免 staging copy；不得交叉别名其他输入工作区或 KV 物理池。
- 新路径仍 inference-only、单设备单 stream。CUDA 运行错误后的 cache 必须丢弃；验证/容量失败在发布追加状态前报错，不承诺 CUDA 异步故障事务恢复。

## 持久工作区与增量页表

prepare 复制一次 CPU 页表到可变的独占布局，分配稳定地址的 GPU 页表、长度、Q/K/V、scores、topk 输出、选块、Attention 输出与 LSE。每步 CPU 只处理活动请求，复杂度 O(active batch)，不遍历或转换整张页表。

长度在 GPU 中推进。跨物理页时，CPU 从现有固定页池分配页，并上传 `2×active_batch` 个 int32 的小 delta；其余步不上传 query 位置或全量页表。delta 使用同步小 H2D 防止 pinned host buffer 被提前复用。Graph 中应用的旧 delta 重复写同一个已分配映射，直到下一次跨页更新。

原 28 配置矩阵中 page=16，36 次追加仅 3 次 delta 上传；33/36 步无需 CPU metadata H2D。这里的计数不包含输入 staging、严格模式的有限值检查或其他模型数据传输。

scores 扫描宽度按容量固定。为很短的请求准备过大的 max_seq_len 可能浪费路由计算；按长度桶准备和更灵活调度是后续工作。topk 的内部临时存储由 PyTorch/Graph memory pool 管理，不声明所有 eager 调用都没有 allocator 活动。

## CUDA Graph 的准确范围

step Graph 包含：页表 delta 应用、位置/长度推进、KV scatter、刚完成块的均值更新、路由打分、topk、选块写入和分页 Attention。attend Graph 不追加 KV。

CPU 页分配/容量检查、输入契约检查、Q/K/V staging、跨页 delta H2D、投影/RoPE/其他模型层位于 Graph 之外。**这是固定活动集合的 GPU 计算路径捕获，不是完整模型或服务全图捕获。** 普通 cache API 仍拒绝外部 capture。

capture 在受控侧 stream warmup，结束后恢复可见页表、长度和块均值；warmup 只写未来不可见 KV 槽，已有前缀逐元素保持。测试覆盖空缓存、已有 live cache、页/chunk 边界、最终容量位置、D=1/256、chunk=1024、BF16/FP16、GQA/MQA、输出复用和跨 stream 拒绝。

## 性能：同输入、同有限值契约

基线固定为上一阶段 `cab3c35132e2d7cf5bc4be841198485aa859e173` 的 eager cache。冻结副本只改 import；trusted 对照仅在 benchmark 子类中跳过 finite 检查，保留原结构和容量校验。严格模式与严格模式比较，trusted 与 trusted 比较，四类结果不混合。

沿用全部 28 配置（B=1/4/16、2K/8K/32K、D=64/128、GQA/MQA、BF16/补充 FP16）。两个独立进程 seed=701/702；每类 5 次预热、3 轮×10 同输入 AB/BA 配对。追加两侧推进相同位置；准备、JIT、capture 和数据构造不计入单步，输入 staging、容量管理和跨页更新计入。

主指标为同步 CPU wall latency，先计算每轮两侧中位数之比，再取跨轮中位数。CUDA event 含 host dispatch 空隙，不称为纯 kernel 时间。下表为各类两轮共 56 行的等权几何平均，相对同契约 cab3 eager；Graph 行包含准备路径的整体效果。

| 有限值契约 | 执行方式 | attend | step（追加并计算） |
| --- | --- | --- | --- |
| 严格检查 | Prepared eager | 1.401× | 1.822× |
| 严格检查 | Prepared Graph | **2.017×** | **2.612×** |
| 调用方保证有限输入 | Prepared eager | 1.544× | 2.235× |
| 调用方保证有限输入 | Prepared Graph | 2.495× | 4.031× |

两轮的全部点估计均高于 1，不承诺矩阵外形状收益。全部 **224 行**（28 配置×2 契约×2 执行方式×2 进程）见 [CSV](data/prepared_20261102_core.csv)，[JSON 汇总](data/prepared_20261102_core.json) 包含每轮范围与退化 ID 检查。

seed 701 的 P01（B=1、2K、D64、32/8）严格 Graph attend 为 489→211 µs，step 为 738→361 µs；trusted Graph 为 410→128 µs、534→161 µs。P18（B16、32K、D128）严格 Graph step 为 3138→582 µs。表中速度比按逐轮统计，不一定等于两个聚合延迟相除。

dense FA core 在 trusted Graph 两轮中仍快于当前 attend 的 **15/28、14/28** 个配置。FA core 读全部历史 token，无 finite 检查或页管理；当前接口包含管理及 staging，选块集合也不同。没有普遍优于 FA 或完整模型吞吐提升的结论。

## 长上下文与碎片化页

6 个补充配置：B1/4×64K/128K GQA；B4 不等长 `[127,4097,8193,32767]`、28/4、page7；B1 128K MQA。固定页池的 free-page 顺序由 seed 打乱，验证非连续物理放置；这是碎片化布局实验，不是 serving scheduler 的负载仿真。

一次独立进程中各类均测 3×10 配对样本。严格 Graph 的 attend/step 几何平均分别 1.970×/4.001×，trusted Graph 为 2.409×/6.652×。每个 query 抽查两个 head：用基线独立 Torch gather 获取所选 KV，CPU FP64 softmax 验证最终输出/LSE，最大输出绝对误差 0.002753。不是长上下文全部 head 的穷举 FP64 验证。

全部 **24 行**见 [CSV](data/prepared_20261102_realistic.csv) / [JSON](data/prepared_20261102_realistic.json)。本补充矩阵只有一轮独立进程，不与核心两轮混成重复性声明。

## 真实预训练权重层

复用本地 `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`，revision `916b56a44061fd5cd7d6a8fb632557ed4f724f60`。仅按 safetensors 索引读取 layer0 Attention、input RMSNorm 和选中 token embedding，共 59,367,424 bytes（约 56.6 MiB），没有加载整模型或下载权重。

真实架构为 Hq/Hkv=28/4、D128。执行真实 embedding、RMSNorm、QKV 投影、RoPE、KV cache 和输出投影；64-token prefill 后连续 24 步 teacher-forced decode：

- Graph 与 cab3 eager 的 Attention 层输出最大绝对差为 0；原始 K/V 缓存读回逐元素相同。
- 固定实际选块后对照独立 FP64，稀疏输出最大绝对误差 0.007656；dense 对照 HF Qwen2Attention SDPA，24 步层输出相对 L2 误差最高 0.002981。
- 初次 HF eager BF16 参考出现差异，保留失败日志。相同 Q/K/V 的手工 BF16 路径逐位复现 HF eager；手工 FP32 路径与 SDPA 在既定容差内对齐，独立 FP64 也通过。HF eager/SDPA 全 prefill 层输出相对差约 7.89%，不能要求不同中间精度路径逐位一致，也不能把它算作 kernel 误差。
- 稀疏与 dense 层输出相对 L2 差为约 4.1%–14.0%，属于不同选块的算法差异。没有完整模型 logits、困惑度或生成质量验证，不声明质量不变。

证据含参数 hash、配置 hash、逐步数值与诊断样本，见 [单层记录](data/prepared_20261102_qwen2.json)。单层 wall 样本用于接入诊断，未按核心矩阵方法完整预热统计，不作为正式模型加速结论。

## 验收、资源与复现

生产实现自 `750a64a` 后未修改；正式性能与补充矩阵代码为 `9dec24b`。最终边界测试和精度归因记录为 `0c04e15`，**515 通过、2 双卡项跳过**（517 collected，pytest 20.68 秒）。78 个源码/配置文件 hash 与 Mac 核对一致。

本批 13 个 GPU 进程累计墙钟 **350.53 秒（5 分 51 秒）**，包括失败的模型参考试跑；低于获准 1,200 秒。每秒采样显存最高 **4,975 MiB**，低于监控线 8,192 MiB；采样不等于瞬时峰值。仅绑定物理 GPU1，无资源冲突中断；完成后 0 MiB、无本批 GPU 进程。CPU/编译线程最多 2、nice=10。没有新增依赖、模型下载、其他 GPU 使用或他人进程操作；未运行 compute-sanitizer。

原始记录：`results/prepared_graph_20261102/`；最终 JUnit 为 `final_boundaries.xml`，最终日志为 `logs/prepared_graph_20261102/final_boundaries.log`。机器信息和原始资源记录不入公开仓库。

复现前取得当批 GPU 许可并配置忽略的 env.local.sh。每轮使用新 tag/输出路径，保留累计预算。

```bash
source scripts/env.sh
python scripts/run_shared_gpu_batch.py --tag prepared_recheck -- \
  python -m benchmarks.benchmark_prepared --seed 701 \
  --out "results/${MOBA_BATCH_ID}/prepared_recheck.jsonl"
python scripts/run_shared_gpu_batch.py --tag long_recheck -- \
  python -m benchmarks.benchmark_prepared --realistic \
  --out "results/${MOBA_BATCH_ID}/long_recheck.jsonl"
python -m scripts.summarize_prepared "results/${MOBA_BATCH_ID}/prepared_recheck.jsonl" \
  --out-dir "results/${MOBA_BATCH_ID}/summary_prepared"
# 单层验证：MOBA_QWEN_SNAPSHOT 只在私有环境中设置为已有本地 snapshot。
python scripts/run_shared_gpu_batch.py --tag qwen_recheck -- \
  python -m scripts.validate_qwen2_layer --checkpoint "$MOBA_QWEN_SNAPSHOT" \
  --out "results/${MOBA_BATCH_ID}/qwen_recheck.json"
```
