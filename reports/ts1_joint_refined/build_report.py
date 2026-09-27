"""Generate evidence report from completed fixed-split evaluations."""
from pathlib import Path
import json,xml.etree.ElementTree as ET
from collections import Counter
ROOT=Path.cwd();OUT=ROOT/'reports/ts1_joint_refined'
def load(name):return json.loads((OUT/name).read_text())
s=load('summary.json');n=len(list(ET.parse(OUT/'pytest.xml').iter('testcase')))
lines=['# J1R 到达指定位置、转向与稳定保持','',f"MODEL_READY_FOR_NEXT_STAGE = {'YES' if s['MODEL_READY_FOR_NEXT_STAGE'] else 'NO'}；仅适用于下述模拟组合任务。",'',s['note'],'','| 种子 | 验证 | 封存 | 成功率 | 碰撞/越界 | 相反指令 | 零神经特征 |','|---|---|---|---|---|---|---|']
failures={}
for row in s['models']:
    seed=row['seed'];result=load(f's{seed}-sealed_test.json')
    lines.append(f"| {seed} | {row['validation']} | {row['sealed']} | {row['success_rate']:.1%} | {row['collision_or_bounds']} | {row['instruction_pairs_successes']}/4 | {row['zero_features_successes']}/12 |")
    failures[str(seed)]={'reasons':dict(Counter(r['reason'] for r in result['results'] if not r['success'])),'final_phase':dict(Counter(r['final_phase'] for r in result['results'] if not r['success'])),'cases':[{k:r[k] for k in ['case_id','reason','final_phase','horizontal_distance_m','heading_error_deg','phase_events']} for r in result['results'] if not r['success']]}
lines+=['',f"规则基线 {load('rule-validation.json')['successes']}/100；均匀随机基线 {s['random_successes']}/300。这两项不是果蝇模型成绩。",'',f"自动化测试 {n} 项通过；浏览器验证见 BROWSER_VERIFICATION.json。",'','## 验收条件','', '同一连续物理场景，水平位置误差≤0.2米、高度误差≤0.1米、水平速度≤0.08米/秒、朝向误差≤16度、角速度≤0.08弧度/秒，全部同时满足并保持2秒。SDK转向动作每次30度。导航到位保持2秒后切换转向，若漂出原始全局目标区则返回导航。切换不重置物理状态，不重复起飞。总限时120秒、每阶段60秒。', '', '## 模型与范围','', '三组种子各包含新训练的导航读出和保留的原H1转向读出，均共享同一冻结 MaleCNS 图。组合描述符 policy-bundle.json 记录三个新导航checkpoint、三个原H1 checkpoint的SHA256和组合源码契约；每种子新增2816个导航训练动作，转向权重保持；SINGLE_POLICY_JOINT_TRAINED=false。phase manager只选技能，不选移动方向；每次动作由当前技能的神经特征读出决定。','', '本轮100验证/300封存场景按种子固定，所有模型使用同一封存集。封存前冻结权重及源码，不依据封存成绩改参数。最终成功率门槛为每种子≥90%、碰撞/越界≤1%、领先随机≥20个百分点，并通过指令对照、特征消融、神经时钟与状态连续性检查。', '', '场景限定为空房间、固定高度、模拟外部定位。完整TS1、真实无人机、视觉定位、动态障碍物与单一模型端到端联合学习均未就绪。历史独立技能成绩只作背景，本轮没有重跑历史独立C1/H1场景。', '', '## 未通过项与失败归因','']
lines.extend(s['acceptance']['reasons'] or ['所有预注册门禁通过。'])
for seed,detail in failures.items():lines.append(f"种子{seed}：失败原因 {detail['reasons']}；失败时阶段 {detail['final_phase']}。逐场误差见 failure-analysis.json。")
lines+=['','## 如何查看','', '打开 http://127.0.0.1:8765/tellosim ，点击顶部 J1R 联合任务11、22、33。三个入口固定显示各自封存第1场，未筛选漂亮样例。可拖动时间轴查看从导航到转向的阶段切换；右侧显示目标距离、角度误差与联合保持时间。', '', '训练与评估：flydrone/tellosim/training/joint_refine.py；继承的联合环境：joint.py。运行 `python -m flydrone.tellosim.training.joint_refine prepare` 锁定协议，再运行 `python reports/ts1_joint_refined/run_campaign.py`。完整新评估需要外部 MaleCNS 大图与既有依赖；回放不需要大图。']
(OUT/'README.md').write_text('\n'.join(lines)+'\n')
(OUT/'failure-analysis.json').write_text(json.dumps(failures,ensure_ascii=False,indent=2))
print('Joint report generated')
