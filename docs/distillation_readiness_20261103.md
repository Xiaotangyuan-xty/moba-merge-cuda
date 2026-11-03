# 多长度校准之后，是否开始蒸馏

状态：**只完成决策与试验方案，尚未训练权重**。证据见 [多长度校准报告](multilen_calibration_20261103.md)。

当前不建议直接启动 1.5B 或长时间蒸馏。多长度校准已经找到达到本轮质量预算的共享策略，但 0.5B/1.5B 仅保留 3/24、4/28 层稀疏。旧的半数层稀疏策略在新测试中仍有明显质量损失；它在 2K/8K/16K 下也没有稳定快于同模型 dense Graph。

蒸馏可能改善固定稀疏结构的质量，不能预期自动减少该结构的路由、cache 与 Attention 执行开销。若项目目标是可部署加速，应先明确有收益的长度/形状；若目标包括验证“训练能否恢复有用的稀疏比例”，下一步适合做一个受限的 0.5B 质量研究小试。

## 建议的小试范围（未执行）

- 模型：固定原 Qwen2.5-0.5B base。Teacher 使用 dense，Student 固定旧策略的 12/24 层稀疏、chunk=128、topk=4，不能靠逐步回退到 dense 获得通过结果。
- Student 启用确定顺序合并、FP32 均值和固定同分规则。路由选择离散，不对 top-k 索引求梯度；通过被选中 Attention 路径更新参数。
- 冻结原始参数，对 Q/K/V/O 投影加入 rank=8、alpha=16 的 LoRA，约 108 万可训练参数。保存 adapter 和配置，原始 checkpoint 不覆盖。
- 使用现有 PyTorch/FA2 手写小型 LoRA 包装，优先不新增依赖。新增依赖若确有必要，需另建环境。
- Teacher 无梯度，Student 使用 prefill autograd 路径。现有 `FullQwen2Session` 和 KV cache 为 inference-only，不能直接作为训练入口；需单独实现训练 forward，复用已验证的 Attention 原语。
- context=2K/8K，batch=1，先做单步显存探测与三步梯度检查，再决定是否能在同一显存限额内完成小试。8K 不可运行时如实缩小范围，不能声称已训练多长度模型。
- 初始预算：100 个 optimizer update，单 seed 只作可行性诊断。梯度累积按显存实测设置，不能将 microbatch 数混写为 optimizer step 数。
- 每个样本预先固定 64 个预测位置，计算完整词表 Teacher/Student KL，避免保留整条序列的全部词表 logits。建议 `0.8 × T² × KL + 0.2 × CE`，T=2，AdamW lr=1e-4，grad clip=1；这些是待验证的初始设置。
- 对 Transformer block 使用 activation checkpointing，dropout=0。先验证重新计算没有改变路由/输出，再训练；不根据测试集修改损失权重、训练步数或 sparse mask。
- 推理时分别检查未合并 adapter 和合并后的权重。正式延迟使用合并后的新副本，保留原始 dense 基线；合并引入的 BF16 舍入变化也要重新验收。

## 训练前后的必要证据

1. LoRA 零初始化时 Student 与相同稀疏基线在既定容差内一致，Teacher 输出在训练前后不变，原始权重 hash 不变。
2. 三步检查中 loss/梯度 finite，adapter 的 B 梯度首先非零，更新后 A/B 均能得到有效梯度；冻结参数没有梯度。不能要求 B=0 初始化时 A 的首步梯度非零。
3. 不只展示训练 loss：对独立 valid 窗口比较不训练 Student、训练 Student、dense Teacher，并保留所有长度和退化行。
4. 训练选择与 early stopping 使用 valid。可为下一次试验预留 valid 目标边界 196608、229376，test 目标边界 65536、229376；不同 split 为不同文件，test 的输入范围与本项目已使用窗口不重叠。正式开始前再冻结完整 recipe 和 hash。
5. 最终在 2K/8K/16K 新测试窗口按每窗口 ΔNLL≤0.05 验收，并维持固定 50% 稀疏层比例。小试失败也要保留；不能用当前校准的 3/24、4/28 层策略替代来宣称训练成功。
6. 两个独立进程对比 merged Student 与同模型 dense Graph，报告准备成本与稳态计时。质量恢复和速度收益分别判断；100 步单 seed 不能证明稳定收敛或通用质量。

任何实际训练都应作为明确的新批次执行，仅使用当次获准的 GPU1，先检查显存与进程，串行、限显存、限时。当前批次已完成，没有消耗剩余时间启动训练。
