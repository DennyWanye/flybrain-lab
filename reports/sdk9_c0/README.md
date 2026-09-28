# 完整 C0 模拟训练交付

三个模型均通过 C0 封存学习门槛；可进入下一课程，实机飞行仍未开放。

独立初始化读出，经示范、DAgger 和 PPO 训练；连接组冻结。封存结果只评价该仿真空房间分布；不代表避障、转向、高度课程或真实无人机能力。

| 正式种子 | 开发验证 | 封存测试 | 训练状态 |
|---|---:|---:|---|
| 11 | 98/100 | 298/300 | completed |
| 22 | 99/100 | 292/300 | completed |
| 33 | 97/100 | 292/300 | completed |

指标：目标半径 20 cm、水平速度 <=8 cm/s、高度误差 <=10 cm，并保持 2 秒。期限 60 模拟秒。
正式部署模式为 argmax：每次选择模型给出最高概率的合法动作。概率采样只作探索诊断；不会重复抽样挑成功回合。
正式通过要求每个种子 300 个相同封存场景成功率 >=90%、碰撞与越界 <=1%，比均匀随机至少提高 20 个百分点。

## 文件

summary.json 保存逐场景结果；cases.json 是训练前固定的验证/封存场景；各模型目录保存初始化、分阶段、PPO 前及最终权重。
训练结果以保存并重新加载后的权重进行评估；正式模型固定前不得开启封存集。旧模型和历史结果均保留。
USER_TEST_GUIDE.md 提供浏览器操作步骤；DIAGNOSTICS.md 列出故障与全部失败；MODEL_INVENTORY.json 提供最终权重哈希；PACKAGE_AUDIT.json 位于 Windows 交付根目录。

## 复现

在 WSL 项目根目录激活 /home/denny/projects/flybrain-env 后运行：
```bash
python -m flydrone.tellosim.training.c0_campaign train --seed 11 --out runs/tellosim-sdk9/NEW_NAME
python -m flydrone.tellosim.training.c0_campaign evaluate --checkpoint runs/tellosim-sdk9/NEW_NAME/checkpoint.pt --out runs/tellosim-sdk9/NEW_NAME/validation.json
python -m flyview serve --project-root . --host 127.0.0.1 --port 8765
```

浏览器：http://127.0.0.1:8765/tellosim 。默认展示固定种子 11，不根据最好成绩选代表。
数据图路径 data/male-v1.npz；约 16.67 万神经元、2558 万条边，不重复打包原始图或虚拟环境。
精确断点续训未开放：checkpoint 支持推理或显式热启动；不能将热启动称为相同轨迹续训。

## 验收边界

FULL_TS1_ACCEPTANCE 和 REAL_FLIGHT_READY 仍为 NO。完整 SDK loopback/遥测、真实姿态力矩控制和硬件标定不能由 C0 成绩推断通过，详见 ACCEPTANCE.md。
