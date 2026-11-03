# 第二轮0.5B适配训练协议（尚未执行）

2026-11-03。状态：**已冻结的提案，训练runner尚未实现，本轮没有新增训练更新**。机器可读配置为 [distill_v2_proposal_20261103.json](../configs/distill_v2_proposal_20261103.json)，CPU可检查 `python -m scripts.check_distill_v2_proposal`。旧v1训练runner不能用于这个v2配置。

本轮[合并诊断与扩展验证](merge_diagnosis_20261103.md)表明：局部FP32合并代数正确，但分离LoRA与合并投影的BF16计算路径不同，离散路由会放大扰动。把合并权重改成FP32也没有通过旧adapter的门槛。因此下一轮改变**Student训练前向的数值形式**，而不是放宽门槛、改称旧导出已修好。

## 对照设计

同一Qwen2.5-0.5B checkpoint、固定12/24层稀疏、chunk128/topk4、稳定FP32路由和确定顺序合并。Q/K/V/O的rank8/alpha16、1,081,344个参数不变。Teacher始终原始dense，绕过adapter，base冻结。

两个arm从相同的旧step000 adapter（B=0）及全新AdamW开始，使用相同seed、输入窗口、监督位置、优化器与验证时点。旧实验未保存optimizer状态，不能把加载step100称作原训练续跑。

| arm | Student投影前向 | 用途 |
|---|---|---|
| split_bf16_control | `base(x) + scale × B(A(x))`，各矩阵乘法输出为BF16 | 保留分离计算形式的配对对照 |
| export_aware_bf16 | `linear(x, BF16(W_base.float() + scale × B.float() @ A.float()))` | 训练时直接使用实际BF16导出的投影形式 |

Export-aware arm每次forward由FP32 master A/B构建权重，梯度必须通过合并和cast回到A/B，不能detach或跨optimizer步缓存旧合并权重。BF16 cast反向使用PyTorch提供的直通梯度近似；它不是真实舍入函数的导数。top-k索引仍不求导。部署时才一次性materialize权重，并在全新副本中验收。

这是**新模型的适配方法**：将训练形式与导出形式对齐，不会追溯改变旧25/100步adapter的失败结论。两个新arm之间主要比较前向数值形式；新旧训练同时改变了长度、监督量和学习率，不能把新旧差异全部归因于某一项。

## 预算与监督

每arm固定120次optimizer update、batch1、累积1，单seed20261103。循环 `[2K,2K,8K,2K,8K,16K]` 共20次，即60/40/20次三个长度；每arm778,240个输入token访问，不能写成唯一token数。新增16K训练覆盖此前只在验证中出现的长度。

每步256个监督位置：128均匀抽样，另128来自最后512个位置且不重复；标签严格偏移+1。每arm30,720个监督位置，为旧100步小试的4.8倍。对这些位置计算完整词表 `0.8 × T² × KL + 0.2 × CE`，T=2，KL按预测位置平均。

完整120步输入起止与监督位置已用CPU固定，schedule SHA256为`0c424b0474a950f4f2102c83b236b395072232a63e38a884d5d64edacbb8535f`，由checker重建。Train长度2,549,174，end在128-token网格上按seed无放回抽样；位置使用seed+100000+step抽样。两个arm必须使用同一张表。配置同时固定模型config/weight、三个语料split与step000 adapter的hash。

AdamW lr=3e-5，warmup10，cosine降到3e-6，weight decay0，gradient clip1；没有依据本轮test调参。每30步验证并保存，包含step0/30/60/90/120。不因test不好而追加步数。初始方案只做这对单seed可行性对照；成功也不构成多seed收敛证明。

## 验证、选点与新测试

Valid保持扩展后的8个目标边界：`32768,65536,98304,131072,163840,196608,229376,262144`，长度2K/8K/16K，共24个窗口。每窗512个tail目标和256个teacher-forced continuation目标。跨长度预测相同目标，不能将24窗当作24份独立文本；这8组valid已经参与诊断和协议设计，属于开发集。

主选点规则继续最小化**最差窗口、最差指标的ΔNLL**，平分选更早。另报告mean/P90、每长度、通过窗数，不能用平均值掩盖超标窗。选择分数最小并不等于成功：全部24个valid窗口、两指标都≤0.05，才达到valid预算；未过则如实记录失败。

为下一轮保留新的test边界 `115200,164352,246272,295424`，三个长度共12窗，最长输入加256个续写目标分别覆盖：

- `[98816,115456)`
- `[147968,164608)`
- `[229888,246528)`
- `[279040,295680)`

CPU已检查均在309,009-token test范围内，与项目此前所有test输入加continuation区间及彼此不重叠。本轮只检查tokenizer长度，**没有对这些窗口做模型推理或质量评分**。两个arm的valid选择冻结后，才一次性比较各自选中模型的训练时eval形式及导出形式；若看过test后改协议，必须换另一个新cohort，不能反复在这12窗上挑模型。

## 实现与开跑门槛

先实现新的prefill训练入口，再完成以下预检；通过前不运行120步训练：

1. 零adapter与既有固定稀疏基线一致；新旧arm初始化一致，Teacher/base和原始checkpoint hash不变。
2. Export-aware eval和实际materialized BF16副本，在2K/8K/16K valid probe上投影、路由和logits一致；先预期逐位一致，发生差异时先定位，不放宽原门槛。
3. 三步诊断loss/梯度finite，B首步非零，A随后非零，冻结参数无梯度；重算路由一致，checkpoint梯度经过对照。
4. 三种长度、256监督位置均在7GiB allocator/8GiB设备监控限制内完成前反向。16K失败就中止并修改尚未执行的方案，不能默默回退2K/8K仍声称覆盖长上下文。

最终导出仍使用原门槛：relativeL2≤0.02、KL≤0.002、maxabs≤0.5，参考为**该新arm的训练时eval形式**，并同时公开它相对旧split形式的变化。每个checkpoint的训练质量、导出质量分开记录；旧adapter不会因为参考形式改变而被重新宣布通过。

保存adapter、optimizer、scheduler、Python/Torch/CUDA RNG、update、配置hash和数据位置到新目录，支持真正续跑；不覆盖旧adapter与基础checkpoint。中断、非有限值、资源超限及外部GPU任务进入都保留记录。

## 性能与执行范围

本轮诊断和扩展验证没有训练上述arm。未来实际执行使用单张当次获准GPU、串行、CPU/编译线程≤2、共享节点guard，每个进程前检查显存和进程；不占用另一张卡等待。训练前需要按预检实测制定该批时间预算，不能沿用已结束实验的剩余账本。

只有质量和导出门槛都通过，才对选中候选做两独立进程AB/BA dense Graph计时。split adapter开销、prefill/capture成本单列。本项目当前固定50%层稀疏结构在2K/8K/16K没有稳定速度余量，**这份协议没有加速承诺**；通过质量试验也不能直接写成可部署的稀疏加速。
