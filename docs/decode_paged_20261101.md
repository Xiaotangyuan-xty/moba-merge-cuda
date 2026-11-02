# 增量 decode、KV cache 与直接分页 Attention

开始日期：2026-11-01，验收日期：2026-11-02。本阶段代码从 Mac 提交到 GitHub，服务器只同步代码并计算验证。
状态：本文件记录功能扩展阶段的验收：**464 项通过、2 项双卡测试跳过**。后续路由/metadata 优化、485 项通过的统一回归与相对 v1 的性能取证见 [后续报告](decode_perf_20261102.md)，本文件保留原阶段口径。

## 接口与语义

```python
from moba_opt import PagedKVCache, ContiguousKVCache
import torch

cache = PagedKVCache(batch_size=2, max_seq_len=4096, num_kv_heads=4,
                     head_dim=128, page_size=16, chunk_size=128,
                     device="cuda", dtype=torch.bfloat16)
# 先用已完成投影和位置编码的 K/V 填充上下文。
cache.append(k_prefill, v_prefill, sequence_ids=[0,1],
             cu_seqlens=[0, context_len_0, context_len_0+context_len_1])
# 一步中两条序列各追加一个 token，并读取对应 query 输出。
result = cache.append_and_attend(q_new, k_new, v_new,
                                  sequence_ids=[0,1], topk=4)
out, global_lse = result.output, result.lse
cache.reset([0])  # 结束该 batch slot 的请求并回收其物理页。
```

- `append` 支持 packed 新 token，CPU `cu_seqlens` 指定每条活动序列的新增数量；同次 append 的 sequence_ids 不重复。
- `attend` 支持一个或多个 query，sequence_ids 每行一个，positions 是每条序列从 0 开始的绝对 token 位置，默认取缓存最后位置。
- `append_and_attend` 自动为新增 query 生成绝对位置；即使同次追加多个 token，早期 query 也不能看见后续 token。
- `DecodeResult` 返回输出、FP32 LSE、实际选块及 CPU query 元数据；dense 模式不返回稀疏选块。
- 当前块必选，只选择完整历史块；topk 包含当前块。保留固定上游 topk=1 的 dense causal fallback；`mode="dense"` 也可显式选择 dense prefix Attention。

## 存储和计算

`ContiguousKVCache` 每条序列使用一个连续 slab；`PagedKVCache` 使用 CPU 管理的逻辑页表，物理页可分散、回收和复用。生产 attention 路径不将全量 KV cache gather 成连续张量。

三个 Triton 路径分别完成：按页表 scatter 新 K/V；只为新完成的 MoBA 块计算 key 均值；直接读取物理分页 K/V 的 attention。路由使用缓存块均值、逐 query-head FP32 打分和 torch.topk，随后在线 softmax 遍历当前块及选中历史块。MHA/GQA/MQA 共用 KV head，未复制全长 K/V 到 query 头。

分页大小和 MoBA 块大小独立，不要求整除。仅完整块的均值参与历史选择，因此同次追加中未来块的均值不会影响更早 query。append 分段方式不会引入运行和式的累积漂移。

页号和长度元数据用 int32，全局存储地址的乘加使用 int64，避免较大 KV 张量的元素偏移先在 int32 中溢出；本批不因此宣称验证了大于 8 GiB 的 cache。

## 支持边界

- inference-only，不提供新 cache/decode 路径的 backward；requires_grad 输入显式拒绝。原 prefill 的输出/梯度接口保留。
- 新 decode 路径 D=1..256 的整数，BF16/FP16，Hq 为 Hkv 的正整数倍。原 FlashAttention prefill 路径仍要求 D=8..256、8 的倍数。
- 单设备、每个 cache 绑定创建时的 CUDA stream，host 操作加锁；跨 stream 和 CUDA Graph capture 显式拒绝。
- 非空序列，最大长度与物理页池固定；可通过 reset 复用 batch slot，不提供服务调度器、跨请求共享页/COW、滑动窗口或自动扩容。
- RoPE、QKV 投影和模型层集成由调用方负责，本阶段不加载模型权重，不声明完整模型质量或服务吞吐。
- GPU metadata 由 CPU 管理并同步，默认输入检查也会同步；不是无 host 开销的实现。
- mapping/routing tie 跟随 torch.topk 的非稳定行为，不承诺 tie 时与另一布局逐位选择同一块。独立参考与无 tie 的构造用例分别检查 attention 数学和路由规则。
- 页池/长度/输入验证失败不发布新布局；CUDA 异步执行错误后的缓存需重建，不承诺设备故障下的事务恢复。
- append 输入不得与该 cache 的物理 K/V storage 别名；需要 clone 后使用。`materialize` 是诊断用独立拷贝，不在 attention 执行路径中。

## 验证设计

CPU 测试覆盖页分配计划、容量失败无状态变更、陈旧计划、重置和非连续页复用；独立 FP64 参考只读取逻辑 K/V 和 query 位置，不复用页表或在线 softmax。

GPU 测试覆盖连续与分页 cache、逐 token 和多 token append、GQA/MQA、32 个对齐 D 和 15 个非对齐 D×两种 dtype、非整除 page/chunk、页回收、因果和请求隔离、对原 prefill 最后一行对照、8K 上下文、极端 logits、chunk_size 1/1024、max topk=9、不规则头数，以及显式错误边界。D=1..256 是接口支持范围，实际抽测 47 个不同 D，不声称穷举所有形状组合。

验证链分开检查原始 K/V 写入、路由和 attention：主要参数化与增量测试要求缓存读回和原始 K/V 逐元素相同；完整块均值对照 FP64 均值后量化值；路由合法性、数量与独立 FP64 打分排名交叉检查（近似 tie 允许保守的 FP32 dot 舍入界）；固定实际选块后，输出/LSE 再对照独立 FP64 attention。另有无 tie 的构造用例检查逐 query-head 选块规则。

## 最终验收记录

- 最终运行代码：`6ebfc72e541ef0ab3070ceeb4d115c6de760d776`，2026-11-02。后续验收文档提交不改变被测源码。
- 收集 **466** 项：**464 通过、2 跳过**。其中新增 CPU allocator/reference 测试 23 项、新增 GPU decode 测试 136 项全部通过；原 prefill、梯度及合并测试一并回归。2 个跳过项是已有双卡设备绑定测试，本批未获双卡授权。
- 最终 pytest 耗时 **144.64 秒**，守护进程记录的该轮进程墙钟 **145.94 秒**。仅有既有诊断路径 `index_reduce` beta API 提示，未出现测试失败。
- 初轮 103 项 decode、初版统一回归 431 通过/2 跳过、非对齐 D 补充 134 项 decode 的记录保留；最终声明以上述完整回归为准，不将不同版本通过数相加。
- 仅使用物理 GPU1（A40）；4 轮 GPU 进程累计墙钟 **313.31 秒**，低于获准的 1,200 秒。1 秒采样记录的设备显存峰值 **3,273 MiB**，低于本批监控限额 8,192 MiB；这不是精确的瞬时峰值测量。
- 每轮先检查显存与进程并绑定目标卡；监控发现他人进程/超额时只终止自身进程组。本批无冲突中断，结束后目标卡 0 MiB、无本批 GPU 进程。未使用其他 GPU 或改动他人进程。
- 复用现有 Torch/FlashAttention/Triton 环境；没有新增依赖或下载模型。CPU/编译线程上限 2，任务使用 nice=10。
- pytest 日志：`logs/decode_paged_20261101/decode_final_suite.log`；JUnit：`results/decode_paged_20261101/final_suite.xml`；资源、源码 hash 与提交记录：`results/decode_paged_20261101/gpu_budget.json`（这些机器相关证据不入公开仓库）。

## 复现

先取得当批 GPU 许可，并在忽略的 `scripts/env.local.sh` 中配置已批准的物理 GPU1 UUID、`MOBA_BATCH_ID`、累计时间和显存限额；共享节点纪律见 README。脚本只接受物理 GPU1，不会自动换卡。保留已有预算记录，重复运行使用新 tag，避免覆盖证据。

```bash
source scripts/env.sh
# CPU 安全检查，不初始化可见 GPU：
CUDA_VISIBLE_DEVICES="" nice -n 10 python -m pytest tests/test_cache_layout.py -q
# 仅在当批许可和私有环境配置完成后：
python scripts/run_shared_gpu_batch.py --tag decode_recheck_01 -- \
  python -m pytest tests/ -q -x --durations=10 \
  --junitxml="results/${MOBA_BATCH_ID}/decode_recheck_01.xml"
```

上述 GPU 命令需要已有原始 fixture 与冻结的 third_party 上游；缺少 fixture 时部分原有测试会跳过，应核对跳过原因，不能直接沿用 464/2 的验收数字。本批未运行 compute-sanitizer。

本阶段不把旧 MHA 的加速数字外推到新 decode kernel。正确性与接口支持验证完成后，才另做公平性能矩阵。
