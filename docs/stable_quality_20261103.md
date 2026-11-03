# 稀疏质量适配、路由可重复性与更长上下文

后续已完成 [多长度校准](multilen_calibration_20261103.md)，以新窗口和更保守的共享层表验收。下文保留本阶段半数层稀疏策略的原始结果与限制。

2026-11-03，单 A40、BF16、batch=1。后续阶段，承接 [完整模型实验](full_model_20261103.md)。

本轮增加确定顺序的 CUDA 合并、FP32 路由与同分规则，以及独立数据校准的逐层 dense/sparse 混合策略。**在当前测试中消除了重复 prefill 漂移，并显著恢复了质量；尚未得到无损或普遍更快的稀疏模型。** 已验证 Qwen2.5-0.5B 至 32704-token prefix、Qwen2.5-1.5B 至 16384-token prefix。1.5B 的约 32K 测试触及本批内存上限，全部保留。

最终核心与实验代码 `5999a60189da4902c2e9c69defe6eb4a95949f21`。早期实验分别位于 `cf06782`、`5470866`；最终核心与 `5470866` 相同，后者之后保留的实验代码变化仅为 OOM 对象生命周期修复。后续 CPU 汇总及文档提交不改变 GPU 实现。

## 1. 重复运行漂移的定位与修复

原 CUDA 路径先用原子计数建立逆映射，每行贡献的 ticket 顺序可能变化。即使每次选中相同贡献，FP32 加权求和的次序也可能不同。BF16 舍入后，这种差异可能进入下一层并改变离散路由。

新增 `merge_forward_cuda(..., deterministic=True)`：在 warp 内对每行最多八个贡献的**原始索引**做 bitonic 排序，再按该顺序计算权重归约和输出归约。原子建图保留；无效槽排序到末尾并被屏蔽，容量检查保留，无 fast-math。独立子算子的 FP64 对照及 20 次位级重复检查覆盖 FP32/FP16/BF16、kmax=1/3/8。

完整模型实验固定同一份 2048-token 输入，每种方案做四次独立 prefill，检查末尾 16 个位置的全部词表 logits 和每层选块 hash：

| 模型 | 原子顺序：最大 logits 相对 L2 | 原子顺序：最多发生路由差异的层数 | 仅确定合并顺序 | FP32 稳定策略 |
|---|---:|---:|---|---|
| 0.5B /24 层 | 0.08038 | 22 | 四次 logits/选块逐位一致 | 四次逐位一致 |
| 1.5B /28 层 | 0.13990 | 26 | 四次 logits/选块逐位一致 | 四次逐位一致 |

只改合并顺序后，当前重复用例的差异即消失，支持原子贡献顺序是本次漂移的重要来源。**不保证跨硬件、编译器版本、不同 GEMM 形状、量化或微小输入扰动后仍保持相同路由。**

新增路由策略参数：

- `deterministic_merge=True`：固定合并顺序。
- `route_precision='fp32'`：prefill 块均值直接在 FP32 计算，避免先舍入为 BF16。decode 原本即使用 FP32 均值。
- `stable_routes=True`：按分数降序，完全同分时取更小 chunk index。prefill 使用重复 argmax；prepared decode 使用固定地址的 Triton top-k，支持 Graph。

FP32 策略可能改变上游 BF16 均值下的选块；它是明确的策略变体。仅开启排序时，对于固定 Q/K/V，离散路由与原路径一致；不据此宣称不同完整模型路径的后续层选块相同。两端均使用 FP32，也不意味着 PyTorch/Triton 的归约次序跨路径逐位相同。分数接近截断阈值时仍可能翻转，没有提供扰动鲁棒性保证。

旧 API 默认保留原行为。新策略通过显式参数启用，训练权重没有变化。

## 2. 校准协议：先冻结策略，再读测试结果

使用与上一阶段相同固定提交中的 WikiText-2 tokenized train/valid/test，hash 与模型 revision 见 `data/stable_quality_20261103_assets.json`。

1. 在 train 前 2×2048 token 上运行 dense teacher。逐层比较同一 dense 输入下的 dense/sparse Attention，经 o_proj 后在最后 512 个 token 的相对残差误差；按误差降序排列层。
2. 在 valid 前 2×2048 token 与每块末尾 32 个独立 continuation 目标上，验证预先固定的七种 dense 层预算：约 0、1/6、1/3、1/2、2/3、5/6、全部层。
3. 选择同时满足两种 ΔNLL≤0.05 的最小 dense 层预算。目标约等于 PPL 增幅≤5.13%。若只有全 dense 达标，输出全 dense；不隐瞒 fallback。
4. 冻结层表与校准文件 hash，在 test 上评估。没有使用测试集重新排序或挑选层数。

结果（层编号从 0 开始）：

- 0.5B：dense 层 `[3,4,5,6,7,9,10,11,13,16,17,18]`，剩余 **12/24 层稀疏**。
- 1.5B：dense 层 `[2,3,4,5,6,8,9,10,12,14,19,20,22,23]`，剩余 **14/28 层稀疏**。

这是不改权重的逐层策略适配，**没有进行稀疏训练、LoRA 或蒸馏训练**。小规模 valid 校准能否泛化，由下方 held-out 结果判断。

## 3. 测试集质量：恢复明显，仍有损失

固定 test 前 4×2048-token 块，prefill 共 8188 个预测目标。另用每块末尾 128 个 token 测 cache continuation，共 512 个目标。两种指标分开加权汇总。与旧报告的 128-target continuation 不是相同样本口径。

| 模型 | 方法 | Prefill PPL /8188 targets | Continuation PPL /512 targets |
|---|---|---:|---:|
| 0.5B | dense | 12.4017 | 12.2089 |
| 0.5B | FP32 稳定全层 k4 | 24.4978 | 35.9837 |
| 0.5B | FP32 稳定全层 k8 | 14.7601 | 18.1548 |
| 0.5B | 校准混合 k4 | **12.8563** | **12.9706** |
| 0.5B | dense prefill +稀疏 k4 decode | 12.4017 | 40.8117 |
| 1.5B | dense | 9.1850 | 9.5699 |
| 1.5B | FP32 稳定全层 k4 | 16.8570 | 25.3494 |
| 1.5B | FP32 稳定全层 k8 | 10.2140 | 14.0274 |
| 1.5B | 校准混合 k4 | **9.7799** | **11.0976** |
| 1.5B | dense prefill +稀疏 k4 decode | 9.1850 | 22.2365 |

两个独立进程的完整质量结果逐项相同，原始文件分别保留。0.5B 混合策略的 prefill/continuation PPL 相对 dense 增加 **3.67%/6.24%**；1.5B 增加 **6.48%/15.96%**。因此**预设校准目标没有在全部测试指标上成立**，不能宣称已经达到 5% 质量预算，更不能宣称无损。仅使用 dense prefill 也不足以修复稀疏 decode 的质量问题。

## 4. 更大模型与长上下文

新增 Qwen2.5-1.5B base，revision `8faed761d45a263340a0528343f099c05c9a4323`，28 层、hidden=1536、Hq/Hkv=12/2、D=128。权重 3087467144 bytes，SHA256 `a961db72e75d52b18e6b0c9d379e51a26973b233385e0e127fdda7d648aec796`，Mac 与服务器分别校验。HF FA2 对照的 128-prefix+12-step 完整模型集成通过原定数值门槛。

长上下文比较使用 **dense 完整模型 Graph +FA2 自动 split-KV**，直接与候选 AB/BA 配对。每行预热 3 步，3×8 对计时样本，两个独立进程。长度为 8192、16384、32704；为后续 token 留出原生 32768 位置上限内的空间。计时包括 CPU cache 管理和 token 写入，不包括采样、网络和服务调度。

全部四种候选、两种模型、三种长度、两次运行共 **48 行**；其中 8 行为内存上限内 OOM，剩余 40 行有完整计时，28 行慢于 dense Graph。没有删除退化或 OOM 行。表中仅汇总预先选定的混合策略，其他候选全部见 CSV。

| 模型 | prefix | 混合 k4 对 dense Graph 的 speedup，两次运行范围 | 末尾 512 目标 ΔNLL |
|---|---:|---:|---:|
| 0.5B | 8192 | 0.9656–0.9663× | +0.11195 |
| 0.5B | 16384 | 1.0002–1.0006× | +0.09933 |
| 0.5B | 32704 | 1.0281–1.0314× | +0.12924 |
| 1.5B | 8192 | 0.9634–0.9639× | +0.19925 |
| 1.5B | 16384 | 0.9886–0.9888× | +0.13509 |
| 1.5B | 32704 | OOM，未给出速度 | 未完成 |

接近 1× 的微小差异不解释为显著收益。0.5B 约 32K 的混合策略只带来约 3% 稳态改善，末尾目标 PPL 却增加约 13.8%。单独全层 k4 约 6.6% 的速度改善伴随更严重质量损失，不能挑出作为主加速声明。

长上下文质量只是每个固定 prefix 末尾 512 个预测，以及单独 24 个 continuation 目标的诊断，**不是长文本完整 PPL 或通用长上下文质量 benchmark**。2K 校准策略的质量预算没有泛化到这些更长输入。

1.5B 另测 8K/16K/约 32K、针位置 25%/75% 的六个 synthetic passkey 用例，每种方法保留完整结果。8K/16K 的四个已完成用例中，dense 与混合策略均 **4/4**，全层 k4 为 **0/4**；32K 的两个用例各方法都触及 allocator cap，不能记为成功或一般检索失败。它是小样本诊断，不是通用 QA 能力证明。

## 5. 资源限制与被拒绝的候选

仅使用授权 GPU1，单进程串行、CPU/编译线程≤2、nice=10。每次由 guard 先检查显存与进程，运行中监测外部占用；只管理本项目子进程组。显存监控上限 8192 MiB，PyTorch allocator 限额 7 GiB，累计进程上限 1200 秒。本批累计 **633.61 秒**，设备监控峰值 **7199 MiB**，未触及监控上限；1.5B32K 的 OOM 在 allocator 限制内记录。

尝试过按 token 分块执行 prefill MLP 以降低激活峰值。小模型单元测试通过，但 0.5B 的 8K 真实模型比较出现 maxabs=1.953125、relative L2=0.10057、KL=0.03083，未通过原定 0.5/0.02/0.002 门槛。**候选已撤回，没有放宽阈值，也没有据此宣传 1.5B32K 支持。** 分块改变了执行形状，完整模型差异的根因尚未进一步定位；该失败也限制了对“稳定路由”的泛化表述。

检索实验的 OOM 分支改为在独立函数帧中创建/关闭 session，使异常先释放所有 GPU 引用再尝试下一用例。修复后重复了全部检索配置，结果单独保留。

最后回归：**546 passed、2 skipped**（双卡测试超出单卡授权范围），22.92 秒。此前包含已撤回 MLP 候选的 548-pass 记录只保留作历史证据，不作为当前测试数量。GPU 结束检查为 0 MiB、无计算进程。复用既有环境，无新增依赖；权重保留于服务器数据盘，Mac 临时权重已删除。

## 6. 复现与后续方向

GPU 命令需要当次许可及共享卡检查，路径和 GPU UUID 由私有环境设置。模型和 WikiText train/valid/test 的固定来源/hash 公开于 assets JSON，原始权重与语料不入库。

```bash
source scripts/env.sh
# 每次选择唯一的 tag/out，通过 guard 运行以下 stage：
python scripts/run_shared_gpu_batch.py --tag sparse_stability_01 -- \
  python -m scripts.evaluate_sparse_adaptation --model "$MODEL_DIR" \
  --corpus "$TEST_CORPUS" --stage stability --out results/sparse_stability_01.json
# calibrate: --corpus "$VALID_CORPUS" --train "$TRAIN_CORPUS"
# evaluate/long/retrieve: --calibration <已冻结的calibrate输出>
# integration: HF FA2 完整模型对照。long 使用两个独立进程。

# CPU：从全部公开原始记录重建 40 行质量与 48 行长上下文计时。
python -m scripts.summarize_sparse_adaptation \
  --input-dir docs/data --out-dir results/stable_quality_repro
```

公开的 `stable_quality_20261103_*` 包含所有正式配置、校准候选、原始配对样本、独立复测、OOM、资产校验、被拒绝候选说明与 manifest。CPU 汇总检查配置矩阵、逐块 token 权重、AB/BA 样本完整性和非有限损失；人工损坏输入检查确认这些错误会被拒绝。

下一步应首先在 train/valid 的多个上下文长度上做校准，扩大独立验证目标数，并以新的未使用测试窗口验收；随后再考虑 dense teacher 蒸馏或稀疏适配训练。改变矩阵形状、精度或量化时，应重新检查完整模型路由与质量。当前更合适的默认性能基线仍是 dense Graph。
