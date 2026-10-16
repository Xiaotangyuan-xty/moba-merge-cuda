# MoBA 源码语义记录

基于固定版本 `b5d58363311d3ca946f1ec444182727c15e338b5` 的实际源码阅读（`moba/moba_efficient.py`、`moba/moba_naive.py`、`moba/wrapper.py`、`tests/`）。后续所有优化必须与本文事实一致；如换版本需重读。

## 1. 记号与布局

- Q/K/V：`[T, H, D]`，T 为全部序列 token 总数（packed varlen）。
- `cu_seqlens`：CUDA int32，长度 B+1，从 0 到 T，主线要求严格递增、非空序列。
- `C` = moba_chunk_size；`K` = moba_topk；`F` = 候选历史块数；`M` = 打包后远端 query 行数。
- 块划分按每条序列独立进行（`calc_chunks`）：每条序列 `ceil(L_b / C)` 个块，允许不完整尾块。

## 2. 两条 Attention 分支（efficient 实现）

1. **当前块分支**：以 `cu_chunk` 为 cu_seqlen 的分块 causal FlashAttention。每个 query 只看自己所在块内的历史（含尾块自身的块内 causal）。
2. **远端块分支**：候选块 = 每条序列中除最后一个块外的所有完整/不完整块（`filtered_chunk_indices` 排除每条序列的最后一块；尾块不作为任何 query 的历史候选）。路由对这些候选块的 key 均值打分，每 query/head 选至多 `K-1` 个，打包成 varlen FlashAttention（causal=False，max_seqlen_k=C）。

`K` 的语义 = 含当前块在内的总块数：efficient 中 `moba_topk = min(K-1, F)`；naive 中通过对当前块 gate 置 `+inf` 保证必选，topk 取 `min(K, num_block)`。两者口径一致。

## 3. k=1 fallback（必须单独处理）

`need_moba_attn=False`（即 K-1=0 或 F=0）时，直接对**整条原始序列**调用全量 causal `flash_attn_varlen_func`。因此 K=1 不是"只看当前块"，而是全量 Attention。主稀疏实验不得混入 K=1；它只作为兼容性分支单独测试。任何优化不得悄悄改变该分支行为。

## 4. 路由细节

- 块均值：`filtered_kv[:, 0].view(F, C, H, D).mean(dim=1)` —— 在输入 dtype（BF16）下求均值，**然后** `.float()`。不能默认"均值全程 FP32"。
- gate：`q` 临时转 FP32 与均值做 einsum（`nhd,shd->nhs`），算完转回输入 dtype。第一版保留该计算顺序与 dtype。
- mask：对块结束位置之后的 query 置 `-inf`（块内及块前不可见），对跨序列的 query 置 `-inf`。
- topk：`torch.topk(gate, k, dim=0, sorted=False)`；再用 `scatter_` 得到选中 mask 与"非 inf"mask 相与。因此每 query 实际远端块数可 < K-1（候选不足时）。tie 行为跟随 torch.topk，主线不修改。
- 零专家剔除：没有任何 query 选中的 (chunk, head) 对被从打包中删除（否则梯度会 NaN——源码注释）。

## 5. 打包与映射

- 打包顺序：`gate_mask` 为 `[F, H, T]`，`nonzero` 在 `(h, s)` 展开的扁平索引上 → 远端行按 **(候选块, head)** 分组的"专家"顺序排列，**不是**按 query 排列。
- `moba_q_sh_indices = idx % T * H + idx // T`：远端第 m 行 → 全局 `(s, h)` 的扁平位置 `p = s*H + h`。同一 query 可出现于多个 (chunk, head) 组，即在打包结果中出现多行。
- 远端 KV：按 head 重排后按块切分（`[F*H 个专家, C, 1, D]`），每专家定长 C；`moba_cu_seqlen_kv = arange * C`。
- 远端 q 的 cu_seqlen 由各专家实际 query 数 cumsum 得到。

## 6. 结果合并（`MixedAttention.forward`）

输入：当前块 out `[T,H,D]`、lse `[H,T]`；远端 out `[M,1,D]`、lse `[1,M]`。lse 来自私有接口，布局为 `[H, T]`，转置为 `[T,H]` / `[M]` 后使用。

数学（与方案 4.3 一致）：`m = max_j(l_j)`；`w_j = exp(l_j - m)`；`O = Σ w_j o_j / Σ w_j`；`L = m + log Σ w_j`。

实现特征（优化候选，未证明是瓶颈）：

- `index_reduce("amax")` 求全局 max；两次 `index_select` 回查；`exp`、两次 `index_add_`、广播乘、`log`；输出 buffer 为 FP32 再转回输入 dtype。
- 合并后保存 `output`（全局 O）与 `mixed_attn_lse_sh`（全局 LSE, `[T,H]`）供反向。

## 7. 反向关键约束

`MixedAttention.backward` 对两个分支分别调用私有 FA backward，但 **out 与 softmax_lse 都用合并后的全局 O、全局 LSE**（自注意力分支 lse 转置回 `[H,T]`；远端分支按 `moba_q_sh_indices` index_select 出全局 O/dO/LSE）。不能改成"各分支用局部 O/LSE 各自 backward 再相加"。两分支均 `deterministic=True`。dq/dk/dv 来自当前块分支；dmq/dmkv 来自远端分支，经 autograd 的 index_select 逆映射自动累加回原始 q/kv（同一 query 多次出现的梯度由 autograd 正确处理）。

## 8. 元数据缓存

`calc_chunks` 用 `lru_cache(maxsize=16)`，键为 **Tensor 对象**。相同数值的新 Tensor 不一定命中缓存；不得原地修改已缓存的 cu_seqlens。主稳态实验：两侧共用同一个不可变 cu_seqlens 对象并预热；动态实验单列。

## 9. wrapper 与范围限制

- HF wrapper（`moba_layer`）：prefill 走 MoBA；**GQA/MQA 通过 `repeat_interleave` 复制 K/V 实现**，不是原生支持；decode（q_len≠kv_len）退回普通 `flash_attn_func`（无 KV cache 分页）。
- 主线限定：MHA、causal、dropout=0、head_dim 64/128、BF16 优先。不支持项显式报错或文档化 fallback。

## 10. 官方测试口径

`tests/test_moba_attn.py`：batch{1,4,7} × head{1,2,4,8} × seqlen{512,1024,2048} × chunk{128,256,1024} × topk{2,3,4}，D=128，BF16。efficient vs naive（naive 的 qk/softmax/PV 在 FP32）：`allclose(atol=rtol=2e-2)` 且 max<4e-2、mean<4e-4（输出与堆叠后的 dQ/dK/dV 同标准）。`tests/test_moba_speedup.py`：3 次预热、10 次计时、墙钟，仅作路径可运行确认，不能替代本项目正式基准。
