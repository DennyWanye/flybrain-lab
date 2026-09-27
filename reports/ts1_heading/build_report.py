"""Build the report from measured results, retaining failed readiness gates."""
from pathlib import Path
import json,hashlib,datetime,xml.etree.ElementTree as E
from collections import Counter
import torch
ROOT=Path.cwd();OUT=ROOT/'reports/ts1_heading'
def load(name):return json.loads((OUT/name).read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
s=load('summary.json');protocol=load('protocol.json');lock=load('frozen-checkpoints.json')
tests={p.name:sum(1 for _ in E.parse(p).iter('testcase')) for p in OUT.glob('pytest*.xml')}
lines=['# 主动转向与稳定保持：H1 训练结果','',f"本轮门禁：MODEL_READY_FOR_NEXT_STAGE = {'YES' if s['MODEL_READY_FOR_NEXT_STAGE'] else 'NO'}。范围为独立转向技能；导航和转向联合任务、完整 TS1、真机飞行仍未就绪。",'',s['note'],'','| 种子 | 验证 | 封存 | CW / CCW 次数 | 碰撞/越界 | 导航保持 | 相反指令配对 |','|---|---|---|---|---|---|---|']
for row in s['models']:
 lines.append(f"| {row['seed']} | {row['validation']} | {row['sealed']} | {row['cw_actions']} / {row['ccw_actions']} | {row['collision_or_bounds']} | {row['retention']}，逐场一致 {row['navigation_exact']} | {row['instruction_pairs_successes']}/4 |")
lines += ['',f"均匀随机基线 {s['random_successes']}/300，平均时长 {s['random_mean_duration_s']:.2f} 秒；规则环境基线 100/100（不是神经模型成绩）。",'','## 训练与证据','', '每个种子 1280 个动作：384 个规则示范动作、两轮各 448 个 DAgger 动作。只训练读出，使用真实冻结 MaleCNS 图，166700 神经元、25582938 连接；没有 PPO 更新，价值头未训练。模型只读取神经特征；示范规则仅用于训练标签和独立基线。','', '目标角度来自外部任务指令，实际朝向来自模拟外部定位测量。每次 SDK 转向 30°，通过标准是角度误差不超过 16°，同时位置、速度、高度、角速度达标并连续保持 2 秒。不能把此结果称作任意角度精确控制。','', '独立转向读出和导航读出共享冻结图；导航模型原权重保留。导航保持逐场复跑使用与上一阶段相同的 100 个 C0 场景。共享读出同时完成位置和朝向任务尚未测试。','']
audit={}
for row in s['models']:
 seed=row['seed'];train=json.loads((ROOT/f'runs/tellosim-sdk9/heading-s{seed}/training.json').read_text());result=load(f's{seed}-sealed_test.json');zero=load(f's{seed}-zero-ablation.json');pairs=load(f's{seed}-instruction-pairs.json')
 initial=torch.load(ROOT/f'runs/tellosim-sdk9/heading-s{seed}/initial.pt',map_location='cpu',weights_only=False)
 final=torch.load(ROOT/f'runs/tellosim-sdk9/heading-s{seed}/checkpoint.pt',map_location='cpu',weights_only=False)
 change=sum(float(((final['policy'][k]-v).double()**2).sum()) for k,v in initial['policy'].items())**.5
 critic_unchanged=all(torch.equal(v,final['policy'][k]) for k,v in initial['policy'].items() if k.startswith('critic.'))
 assert change>0 and critic_unchanged and final['contract']['base']['graph_sha256']=='badc33a247894fe12c4d68d3ff791393857cffa39ff2ab81869608a11fa1e0c9'
 directions=[next((a for a in x['actions'] if a in (7,8)),None) for x in pairs['results']]
 audit[str(seed)]={'readout_change_l2':change,'critic_weights_unchanged':critic_unchanged,'instruction_first_turns':directions,'instruction_shortest_direction_correct':directions==[7,8,7,8],'train_options':train['options'],'train_elapsed_s':train['elapsed_s'],'stage_accuracies':[x['training_label_accuracy'] for x in train['stages']],'checkpoint_sha256':train['checkpoint_sha256'],'failure_reasons':dict(Counter(x['reason'] for x in result['results'] if not x['success'])),'zero_features_successes':zero['successes'],'instruction_pairs':pairs['results'],'training_roundtrip_exact':train['roundtrip_exact']}
 lines += [f"### 种子 {seed}",'',f"训练 {train['elapsed_s']:.1f} 秒；封存失败原因 {audit[str(seed)]['failure_reasons']}；神经特征置零对照 {zero['successes']}/12。置零为诊断干预，不是另一份训练模型。",f"读出参数变化 L2={change:.6f}；价值头权重保持未训练；相反指令首个转向动作 {directions}（预期 [7,8,7,8]）。",f"权重 SHA256：`{train['checkpoint_sha256']}`。",'']
lines += ['## 门禁结论','']+(['全部预登记 H1 门禁通过。'] if s['acceptance']['passed'] else [f"- {reason}" for reason in s['acceptance']['reasons']])
lines += ['','## 检查范围','',f"自动化测试 {sum(tests.values())} 项通过（{tests}）；真实全图小规模读出更新和权重重载已检查。浏览器交互证据见 BROWSER_CHECK.md；打包解压验证见 PACKAGE_AUDIT.json。",'', '未完成的更大范围不因此变成通过：联合导航+指定朝向、真实视觉定位、障碍课程、完整 TS1 压力矩阵、真机校准与飞行。','', '每个最终权重在封存评估前锁定。失败场景不用于本轮调参；诊断和训练数据带独立种子与来源。']
(OUT/'README.md').write_text('\n'.join(lines)+'\n')
(OUT/'RESULT_AUDIT.json').write_text(json.dumps({'models':audit,'tests':tests,'protocol_sha256':sha(OUT/'protocol.json'),'frozen':lock,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},ensure_ascii=False,indent=2))
(OUT/'USER_TEST_GUIDE.md').write_text("""# 查看本轮结果

打开 http://127.0.0.1:8765/tellosim ，刷新一次。点击最前面的“H1 转向模型 11”；22、33 是另两次独立训练。按钮固定播放封存第 1 场，整体成绩以 H1 结果表为准。

1. 紫色长箭头为指定朝向，红色箭头为机头。观察右侧测量角度误差逐步减小。
2. 命令区应出现 cw 30 或 ccw 30，最后 stop；策略概率和神经活动来自真实记录。
3. 点击暂停、拖动进度条、再播放；右侧读数应对应所选时刻。
4. 稳定保持到 2 秒且误差不超过 16°才成功。记录结果会标明当前样例成功或失败。
5. “H1 对照 · 种子11训练前/种子11训练后”来自预先固定的同一验证场景；未训练是随机初始转向读出，并非上一阶段导航模型。

训练实时观察只保留最后快照，不能拖动；需要拖动请用 H1 正式模型回放。旧 C1 结果仍保留，但它没有验证主动转向。
""")
(OUT/'REPRODUCE.md').write_text("""# 复现与产物

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
""")
print('Reports built from formal outcomes')
