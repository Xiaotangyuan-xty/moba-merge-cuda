# 数值验证记录（随项目进展更新）

## 验证链现状（2026-10-16）

| 层级 | 内容 | 结果 | 证据 |
|---|---|---|---|
| 接口探针 | 私有 FA 返回布局 `[T,H,D]`/LSE `[H,T]` FP32、MoBA 小输入 VJP、k=1 fallback 恒等 | 6/6 通过 | `results/environment/probe_20261016T041438Z.json` |
| 官方测试 | 固定 SHA 官方 324 例（efficient vs naive，BF16） | 324/324 通过，36s | `logs/pytest_official_20261016T041507Z.log` |
| 参考自验证 | 固定路由参考 vs 稠密/逐 query gather；合并参考手算/有限差分 | 11/11 通过（CPU） | `tests/test_reference.py` |
| 真实路由对照 | 官方（BF16 FA+FP32 合并）vs 固定路由 FP64 参考，S01–S05 | max_abs ≤8.0e-3、mean_abs ≤2.7e-4、rel_l2 ≈2.2e-3（BF16 舍入量级） | `results/raw/fixed_route_check_20261016T050311Z.json` |
| 合并数学交叉验证 | 真实 fixture 上官方合并 vs 方案 4.3 公式（FP64 重组） | out max_abs ≤1.9e-3（bf16 cast）、lse max_abs ≤5.6e-7 | `tests/test_merge_fixtures.py`（3 fixtures） |

## 已确立的事实

1. 环境数值路径与官方一致，无需兼容补丁。
2. 路由导出器与固定路由参考在真实官方路由上端到端可信（差异 = BF16 舍入）。
3. 官方合并阶段的数学就是方案 4.3 的全局 softmax 归一化（max→exp→加权→归一化，全局 LSE），优化 kernel 可以建立在该公式上。
4. 官方混合 LSE 为 FP32，输出 cast 为输入 dtype；合并内部累加为 FP32。

## 待制定（性能调优前冻结）

- FP32 合并子算子容差起点 atol=1e-5 / rtol=1e-4（方案 7.4），待 kernel 出来后用 FP64 参考校准并冻结。
- 完整 BF16 路径沿用官方口径（atol/rtol=2e-2，max<4e-2，mean<4e-4）。
- FP16 路径误差分布未建立。
