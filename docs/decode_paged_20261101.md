# 增量 decode、KV cache 与直接分页 Attention

开始日期：2026-11-01，验收日期：2026-11-02。本阶段代码从 Mac 提交到 GitHub，服务器只同步代码并计算验证。
状态：实现与测试用例已建立；最终验收结果在实验完成后填入。

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

GPU 测试覆盖连续与分页 cache、逐 token 和多 token append、GQA/MQA、32 个对齐 D 和 15 个非对齐 D×两种 dtype、非整除 page/chunk、页回收、因果和请求隔离、对原 prefill 最后一行对照、8K 上下文、极端 logits、chunk_size 1/1024、max topk=9、不规则头数，以及显式错误边界。代码中包含的测试不是已通过声明，最终结果以保存的日志/XML 为准。

初轮 decode 103 项通过；加入非对齐 D 和 ragged/max-topk 后 134 项通过。旧 prefill 与首轮 cache 的整体验证为 431 通过/2 个双卡项跳过。最终版本还需包含追加 storage 别名保护后的统一回归，以下以最终记录为准。

本批只获准物理 GPU1；预计设备显存上限 8 GiB、GPU 进程累计墙钟上限 20 分钟。每轮检查显存及进程，发现其他任务进入目标卡时只停止本批自身进程组。结果和源码 hash 保存在 `results/decode_paged_20261101/`。

本阶段不把旧 MHA 的加速数字外推到新 decode kernel。正确性与接口支持验证完成后，才另做公平性能矩阵。
