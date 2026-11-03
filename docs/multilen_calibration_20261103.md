# 多长度校准与蒸馏前置评估

2026-11-03。复用固定的 Qwen2.5-0.5B/1.5B base、现有环境与 WikiText-2 tokenized 语料，无新增依赖或下载。GPU 实验提交为 `7469e69`（粗网格）和 `d9c83db`（细化）；后续 CPU 汇总及文档不改变推理实现。资产 revision/hash 沿用 `data/stable_quality_20261103_assets.json`。

后续独立批次已完成 [0.5B固定半数层稀疏蒸馏小试](distill_pilot_20261103.md)，仅部分恢复质量。下文“尚未训练”描述本校准阶段结束时的状态。

**结果：多长度校准后，0.5B 保留 3/24 层稀疏、1.5B 保留 4/28 层稀疏，可以通过本轮新测试窗口的质量预算；没有观察到稳定的推理加速。** 这是不改权重的策略校准，蒸馏训练尚未启动。

## 预先固定的协议

长度为 2048、8192、16384。每个 split 使用两个固定的绝对目标边界 T；长度 L 的输入为 `[T-L,T)`，prefix 末尾质量目标为 `[T-512,T)`，独立 continuation 目标为 `[T,T+256)`。这样不同长度预测**相同的目标 token**，仅历史上下文长度变化。每条请求位置从零开始，方法间保持一致。

每个长度分别有 **1024 个 prefix tail 目标、512 个 continuation 目标**。同一批目标会在三个上下文下重复预测，不能把它们当作三个独立语料样本。这里不是完整 WikiText PPL，也不能与旧报告的 8188-target 全 prefill PPL 直接混用。

- Train 目标边界：131072、262144。逐层在相同 dense teacher 输入上比较 sparse/dense Attention 的 o_proj 相对残差误差，统计最后 512 个位置；每个长度先平均两个窗口，再取三个长度的最大值作为层排序分数。
- Valid 目标边界：65536、131072。所有长度、每个窗口、tail 和 continuation 均须满足 **ΔNLL≤0.02**。这是留有余量的校准目标，约相当于 PPL 增幅≤2.02%。
- Held-out 目标：每窗口两种指标均 **ΔNLL≤0.05**，约相当于 PPL 增幅≤5.13%。阈值在两阶段中保持不变。
- 候选策略按同一个 train 排序保留最敏感的若干层为 dense，其余层使用 FP32 稳定 k4。主策略为一张跨长度固定层表；另报告三个离散长度各自的选择。请求进行中不切换层表，可继续使用完整模型 Graph。
- Teacher/Student 原始权重均不更新。完全 dense 必须作为显式 fallback，不能将回退结果宣传为稀疏质量成功。

## 两阶段校准与数据隔离

第一阶段按 0、1/6、1/3、1/2、2/3、5/6、全部层七个预算校准。在该粗候选族中，两个模型的跨长度策略都回退到全 dense：0.5B 保留20层dense时，最差continuation ΔNLL=0.02129；1.5B保留23层时，最差tail ΔNLL=0.03145，均超过0.02。

粗阶段测试边界为 98304、196608，输入及 continuation 只覆盖 `[81920,98560)`、`[180224,196864)` 的子区间，与项目此前使用的 test `[0,32768)` 不重叠。

第二阶段只依据 valid 的临界预算细化：0.5B 增加21/22/23层dense，1.5B增加24/25/26/27层dense。Train 排序、valid 窗口、原有分数与阈值都不变，旧测量通过 parent calibration hash 复用。所有新增整数预算均评估，没有假定“增加dense层一定单调改善质量”。

细化阶段改用另一个新测试 cohort：边界 **147456、278528**，最长输入与continuation区间为 `[131072,147712)`、`[262144,278784)`，与前两批 test 输入均不重叠。没有针对第一批测试分数重新选层，也没有在第二批测试后再次调整策略。

每阶段先冻结两个模型的校准输出，再开始该阶段测试；公开的 run manifest 和 calibration hash 可核查顺序与文件身份。粗网格所有回退、第二阶段所有候选、两批测试均保留。

## 最终共享策略与质量

下表是第二批测试的结果。每个模型共六个窗口（三长度×两边界），每个窗口分别检查两项质量指标。

| 模型 | 保留的 sparse 层（从0开始） | 其余 dense 层数 | 最差单窗口 ΔNLL | 对应最大 PPL 增幅 | 通过窗口 |
|---|---|---:|---:|---:|---:|
| 0.5B | `[0,2,23]` | 21/24 | 0.039846 | **4.07%** | 6/6 |
| 1.5B | `[0,1,18,27]` | 24/28 | 0.008834 | **0.89%** | 6/6 |

主策略的按长度平均 ΔNLL如下；这些平均值不能代替上表更严格的逐窗口检查。

| 模型 | prefix | tail ΔNLL /1024 targets | continuation ΔNLL /512 targets |
|---|---:|---:|---:|
| 0.5B | 2048 | +0.000485 | +0.010286 |
| 0.5B | 8192 | +0.001950 | +0.019889 |
| 0.5B | 16384 | +0.003130 | +0.024759 |
| 1.5B | 2048 | -0.000365 | +0.000184 |
| 1.5B | 8192 | +0.001620 | -0.004886 |
| 1.5B | 16384 | +0.003364 | +0.003100 |

按长度选择的次要策略中，0.5B 的 2K/16K 可保留4层稀疏，8K为3层；1.5B均保留4层。它们也通过本批测试，但只验证了这些离散点，不保证中间长度或任意prompt。

旧2K校准的半数层稀疏策略，在第二批测试共有11/12个窗口超出质量预算。新策略降低了质量损失，同时将稀疏比例从50%降到 **12.5%/14.3%**，不能说是在相同稀疏度下恢复质量。少量负ΔNLL也不能解释为通用质量提升。

## 性能：质量达标没有自动换来速度

完整模型 Graph，配对基线为同模型、同输入、FA2自动split-KV的dense Graph。每行3步预热、3×8对AB/BA计时，两独立进程。包含CPU cache管理、token写入和同步，不含采样、网络或调度。准备成本单次观测仅作诊断。

最终共享策略相对dense Graph的配对speedup范围：

| 模型 | 2048 | 8192 | 16384 |
|---|---:|---:|---:|
| 0.5B | 0.9852–0.9856× | 0.9909–0.9909× | 1.0005–1.0005× |
| 1.5B | 0.9806–0.9807× | 0.9894–0.9898× | 0.9958–0.9961× |

约1.0005×视为接近持平，不作为加速。第二阶段全部36行中有30行的比值小于1；接近1的微小差异不作显著性推断。粗阶段另外36行完整保留，全dense fallback的约1×也不算稀疏收益。

旧半数层稀疏策略在本轮2K/8K/16K的范围为：0.5B **0.9452–1.0019×**，1.5B **0.9359–0.9882×**。因此不能预期只训练权重，就自动消除当前固定结构的执行开销。

## 可复用产物

- `data/multilen_refine_20261103_policy_05.json`、`policy_15.json`：固定共享层表、原checkpoint revision/config hash、calibration hash、稳定策略选项及适用限制。
- `configs/multilen_20261103.json`、`configs/multilen_refine_20261103.json`：两阶段预先固定的窗口、阈值与预算。
- 两族公开原始记录与CSV：分别为 `multilen_20261103_*`、`multilen_refine_20261103_*`。共72行计时，粗/细两族分别汇总，不能选最快的一批合并。细阶段126行validation候选记录中84行来自原校准，其余42行为新增测量。
- `scripts/calibrate_multilen.py`：calibrate/evaluate/latency；`scripts/summarize_multilen.py`：CPU审计与汇总。旧阶段汇总按文件族筛选，仍可精确重建。

模型须按已固定revision加载、eval并关闭梯度；以下示意复用已有准备/执行接口，质量结论仅限上述实测窗口：

```python
policy = json.load(open("docs/data/multilen_refine_20261103_policy_05.json"))
session = FullQwen2Session(model, capacity, backend="model_graph",
                          dense_layers=policy["dense_layers"], **policy["options"])
hidden = session.prefill_hidden(input_ids)
session.prepare()
logits = session.step(next_teacher_forced_token)
session.close()
```

GPU复现仍需当次许可与GPU1检查。粗阶段使用默认recipe；细阶段校准增加 `--recipe configs/multilen_refine_20261103.json --parent-calibration <粗阶段输出>`，先冻结两个模型的校准，再分别evaluate和两次独立latency。CPU重建：

```bash
python -m scripts.summarize_multilen --input-dir docs/data --out-dir results/multilen_repro
python -m scripts.summarize_multilen --input-dir docs/data \
  --recipe configs/multilen_refine_20261103.json \
  --pattern 'multilen_refine_20261103_*.json' --out-dir results/multilen_refine_repro
```

汇总检查完整模型/长度/方法矩阵、配对样本和AB/BA顺序、target hash跨长度一致、quality与timing输入一致、实际mask与冻结策略一致、parent测量未被改写。破坏配置、计时pair、order或非有限样本的CPU检查都会被拒绝。

## 资源与边界

全过程仅物理GPU1；guard和脚本在分配前验证物理index=1，并用UUID屏蔽其他卡。显存和进程查询也定向GPU1，GPU0/GPU2没有运行本项目任务或配置操作。发现外部进程时只结束本项目子进程组。

串行、nice=10、CPU/编译线程≤2，设备监控上限8192MiB、allocator上限7GiB、累计进程上限1200秒。本批18个GPU进程累计 **968.82秒**，设备监控峰值 **6695MiB**；两阶段共用同一账本，没有重置。最终GPU回归 **554 passed、2 skipped**，双卡测试按授权范围跳过。GPU1结束检查无本项目进程。

这些是固定A40、BF16、batch=1和固定checkpoint的有限实验。共同验证上限16K prefix和256个续写目标，不新增1.5B32K支持；不保证任意语料/模型的质量。候选族也不是所有层组合的穷举，“最小dense预算”仅指已测试的嵌套候选。

## 蒸馏判断

多长度校准的质量预算已在新的有限测试中达到，但代价是大部分层回到dense。若要恢复50%稀疏比例，权重适配值得研究；当前固定50%结构在已测长度没有显著速度余量，训练不构成加速承诺。

**本轮不直接启动1.5B或长时间蒸馏。** 下一步若继续质量研究，建议只做0.5B、固定50%稀疏的小规模LoRA/KL可行性试验，并重新保留未使用的valid/test窗口；具体未执行方案见 [蒸馏前置评估](distillation_readiness_20261103.md)。若以部署速度为主要目标，则应先改善稀疏执行路径的性能。
