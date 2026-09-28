# 示范模型继续 PPO 训练：单种子保留能力检查

固定 seed 37，先用 16 个独立训练回合拟合价值输出层，再执行 512 次 PPO 动作采样、4 次更新。学习率 0.00003；冻结果蝇连接组和共享隐藏层，只更新动作和价值输出层。

| 动作选择 | 验证场景 | 续训前 | 续训后 |
|---|---|---:|---:|
| argmax | previous8 | 8/8 | 8/8 |
| argmax | additional8 | 8/8 | 8/8 |
| sampled | previous8 | 8/8 | 8/8 |
| sampled | additional8 | 8/8 | 8/8 |

保留能力检查：通过。正式阶段门禁 MODEL_READY_FOR_NEXT_STAGE 仍为 false。
原有 8 场景与另外 8 场景分别报告；两种动作选择使用相同场景，不可相加解释为 32 个独立场景。采样动作使用匹配随机数。只评估固定预算的最终模型，没有按验证成绩挑选中间模型。

价值拟合前后动作参数完全一致；PPO 后共享隐藏层完全一致，动作和价值输出层确实改变；检查点重新加载决策完全一致。

价值拟合训练误差：14.582822799682617 → 1.0926846265792847，这是训练拟合误差，不是独立的价值预测验证。

后续：Repeat the fixed protocol on seeds 23 and 37 before four-direction training.

protocol.json 保存预先固定的配置；summary.json 保存成绩及参数审计；sampled-before.json / sampled-after.json 保存逐场景采样结果；s11.log 保存执行日志。

范围：本轮仍只训练固定高度的近距离前后移动，未执行四方向、避障、300 场景封存评估或真机飞行。
