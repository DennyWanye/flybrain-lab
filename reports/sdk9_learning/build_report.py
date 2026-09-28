from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[2];out=root/'reports/sdk9_learning'
comparison=json.loads((out/'comparison-sampled.json').read_text())
bootstrap=json.loads((out/'bootstrap-comparison.json').read_text())
probe=json.loads((out/'decoder-probe.json').read_text())
v3=json.loads((root/'runs/tellosim-sdk9/nearx-v3-s11-20260926/summary.json').read_text())
labels={'reservoir':'果蝇读出','raw_observation_control':'原始观测对照','zero_brain_control':'脑特征置零对照'}
rows=comparison['rows'];boot=bootstrap['results']
lines=['# 果蝇近距离控制：诊断、修复与学习实验','',
'本轮已完成信号修复、小课程、多种子对照和独立的规则示范启动。正式阶段门禁仍为 **MODEL_READY_FOR_NEXT_STAGE = false**。','',
'## 信号修复','',
'原配置在 15 cm 处不能区分四个目标方向，30 cm 处只有两种输出历史。修正感知编码与下游读出覆盖后，四种距离的四个方向均可区分。',
f"带干扰变量的独立观测解码：旧配置 {round(probe['rows'][0]['heldout_accuracy']*64)}/64，新配置 {round(probe['rows'][1]['heldout_accuracy']*64)}/64。这是信息诊断，不是飞行成绩。",'',
'## 纯 PPO：同预算的三种子对照','',
'每次 1536 个动作、12 次更新；同种子使用相同初始权重和场景。固定动作选择为 argmax，概率选择使用匹配的随机数种子。每种方法的 24 个评估回合来自 3 个训练种子 × 相同 8 个验证场景，不是 24 个独立场景。','',
'| 种子 | 方法 | 训练中成功回合 | 固定动作成功 | 概率动作成功 |','|---|---|---:|---:|---:|']
for r in sorted(rows,key=lambda r:(r['seed'],r['source'])):
 lines.append(f"| {r['seed']} | {labels[r['source']]} | {r['training_successes']}/{r['training_episodes']} | {r['trained_successes']}/{r['evaluation_episodes']} | {r['sampled_successes']}/{r['sampled_episodes']} |")
lines+=['','纯 PPO 的实际成绩全部保留，不以训练中的成功次数替代验证成绩。单个 seed-11 的额外 3 组概率动作试验为 11/24；该组是补充诊断，未重复计入上表。','',
f"停止动作延长至 2 秒的独立 v3 实验：训练中成功 {sum(s['reason']=='success' for s in v3['episodes'])}/{len(v3['episodes'])}，固定动作验证 {v3['trained']['successes']}/8。单独改变停止动作时长没有解决固定动作评估失败。",'',
'## 规则示范启动：独立学习模型的验证','',
'每个种子使用 32 个规则示范回合，再用模型自己执行的 2×16 个回合补充专家标签。最终验证时由学习模型独立决策，规则不参与动作执行。输入仍为果蝇神经状态；未使用原始观测旁路。此方法是监督学习与 DAgger，**PPO 更新为 0**，不能称为纯强化学习从零学会。价值头尚未训练。','',
'| 种子 | 初始模型 | 学习后模型 | 学习后脑特征置零 |','|---|---:|---:|---:|']
for r in boot:lines.append(f"| {r['seed']} | {r['baselines'][0]['successes']}/8 | {r['trained']['successes']}/8 | {r['brain_zero_ablation']['successes']}/8 |")
lines+=['','## 当前适用范围','',
'固定 1 m 高度、随机起点、前后方向 0.30–0.65 m 的目标；执行停止、前进 20 cm、后退 20 cm。目标半径仍为 20 cm，稳定保持 2 秒，任务期限 60 秒。尚未证明任意方向、障碍回避、噪声鲁棒性、正式 3×300 封存场景或实机控制。',
'当前连接组权重冻结，只训练动作读出。编码、神经元映射及物理模型是工程近似，不等于生物学或真机标定。',
'下一步应先确认示范启动模型达到可复现的近距离控制，再用其 actor 做显式 PPO warm-start、重新训练 critic，并检查继续优化是否破坏已学行为；随后扩大至四方向课程。不要直接把模型接到真实无人机。','',
'## 验证与产物','',
'详细方法与复现命令见 METHOD.md。训练/神经时钟、数据分片与检查点哈希见 artifact-audit.json；回归结果见 pytest-final.txt。检查点重新加载的决策一致性在每次训练中验证。',
'新增查看器代码通过语法检查，数据 API 由对应回归覆盖；本轮没有绕过浏览器工具此前的安全拒绝，新增界面交互状态仍为 NOT_VERIFIED。','',
'图中相同场景跨训练种子重复，柱高不代表正式泛化成功率。','',
'![实验成绩](learning-results.png)','',
'源码与全量回放位于 WSL 项目 /home/denny/projects/flybrain_lab_4spark_v0_1；Windows 交付目录只复制报告、日志和模型，不重复复制大型连接图。']
(out/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
colors=['#2563eb','#d97706']
keys=['reservoir','raw_observation_control','zero_brain_control'];x=np.arange(3)
for j,(success,denominator,label) in enumerate([('trained_successes','evaluation_episodes','Argmax'),('sampled_successes','sampled_episodes','Sampled')]):
 nums=[sum(r[success] for r in rows if r['source']==k) for k in keys]
 dens=[sum(r[denominator] for r in rows if r['source']==k) for k in keys]
 bars=axes[0].bar(x+(j-.5)*.36,np.array(nums)/dens*100,.36,label=label,color=colors[j])
 axes[0].bar_label(bars,labels=[f'{n}/{d}' for n,d in zip(nums,dens)],padding=3)
axes[0].set_xticks(x,['Brain','Raw control','Zero control']);axes[0].set_title('PPO: 3 seeds, same 8 validation cases')
for j,(key,label) in enumerate([('trained','Learned actor'),('brain_zero_ablation','Brain zeroed')]):
 nums=[r[key]['successes'] for r in boot]
 bars=axes[1].bar(x+(j-.5)*.36,np.asarray(nums)/8*100,.36,label=label,color=colors[j])
 axes[1].bar_label(bars,labels=[f'{n}/8' for n in nums],padding=3)
axes[1].set_xticks(x,[f"Seed {r['seed']}" for r in boot]);axes[1].set_title('Demonstration + DAgger: argmax policy')
for ax in axes:
 ax.set_ylim(0,115);ax.set_ylabel('Validation success (%)');ax.legend(loc='upper center');ax.spines[['top','right']].set_visible(False)
fig.suptitle('Near-axis curriculum only; not full C0 or real-flight acceptance',fontsize=12)
fig.savefig(out/'learning-results.png',dpi=160);plt.close(fig)
print('Wrote README.md and learning-results.png')
