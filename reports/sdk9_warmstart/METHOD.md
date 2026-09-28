# 复现本轮训练

WSL 根目录：/home/denny/projects/flybrain_lab_4spark_v0_1
Python：/home/denny/projects/flybrain-env/bin/python

```bash
python -m flydrone.tellosim.training.runner --graph data/male-v1.npz --device cuda --profile balanced_rate_v3 --curriculum C0-near-x --options 512 --rollout 128 --eval-cases 16 --seed 11 --init-from runs/tellosim-sdk9/nearx-bootstrap-s11-20260926/checkpoint.pt --critic-episodes 16 --freeze-body --learning-rate 0.00003 --no-publish --out runs/tellosim-sdk9/NEW_NAME
```

分别对源检查点和新检查点执行配对采样评估：

```bash
python -m flydrone.tellosim.training.evaluation --graph data/male-v1.npz --checkpoint CHECKPOINT --out NEW_RESULT.json --case-count 16 --repeats 1 --sample-only
```

seed 23 和 37 使用相同配置，只改变 seed 和对应源检查点。repeat.py 记录了本轮精确的顺序执行命令；再次运行前须将脚本中的 tag 与输出路径改为全新名称，不能覆盖原训练目录。build_report.py 与 finish.py 使用本轮路径生成审计、汇总和 Windows 交付。PPO 是显式 warm-start：优化器和环境重置，并非精确断点续训。

critic-data.npz 保存模型采样轨迹的神经特征和完整回合折扣回报。价值拟合不接触验证数据；共享隐藏特征 detach 后，只优化 critic 输出层。PPO 阶段固定 body，优化 actor 与 critic 输出层，维持已有编码、图、动作掩码、时间尺度、成功半径和稳定保持时长。
