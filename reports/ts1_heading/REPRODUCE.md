# 复现与产物

在交付源码根目录及锁定依赖环境中运行；训练需要外部 data/male-v1.npz，哈希见 DELIVERY_MANIFEST.json。现有工作区已完成的训练目录禁止覆盖。复现请使用一个新实验根目录：复制源码、configs、contracts、tests、旧 C1 的 runs/tellosim-sdk9/c1-sXX/ppo/checkpoint.pt 及 reports/ts1_c1（用于保持对照），并放入原图。reports/ts1_heading 只复制本轮 .py 脚本和说明，不复制已生成的结果；不要复制 runs/tellosim-sdk9/heading-sXX。解压后的完整交付包自带已完成结果，直接再次训练会明确拒绝覆盖。

```bash
python -m pytest tests/flydrone -q
python -m pytest tests/test_flylab.py -q
python -m flydrone.tellosim.training.heading_campaign prepare
python -m flydrone.tellosim.training.heading_campaign baselines
python -m reports.ts1_heading.run_batch
python -m reports.ts1_heading.build_report
```

run_batch 等待规则基线完成，训练 3 个种子，冻结权重，评估全部场景，再生成结论。每个训练进程预算 1800 秒，整个评估批次预算 3600 秒。出错只停止本批次拥有的进程，保留日志和已完成阶段，不自动修改参数重试。

H1 checkpoint.pt 用 heading.load_heading 加载；独立任务合同会拒绝旧导航权重。旧导航仍通过原 loader 加载。H1 stage 权重只能明确地做阶段热启动；没有宣称支持精确中途恢复。
