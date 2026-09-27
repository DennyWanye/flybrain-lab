"""Produce the report only from completed frozen evaluations."""
from pathlib import Path
from collections import Counter
import json,xml.etree.ElementTree as ET
root=Path.cwd();out=root/'reports/ts1_stability'
def load(name):return json.loads((out/name).read_text())
s=load('summary.json');tests=sum(len(list(ET.parse(p).iter('testcase'))) for p in out.glob('pytest*.xml'))
lines=['# J2R 连续状态监督：轻微扰动下的联合任务可靠性','',f"MODEL_READY_FOR_NEXT_STAGE = {'YES' if s['MODEL_READY_FOR_NEXT_STAGE'] else 'NO'}",'',s['note'],'','|种子|验证|同场景J2|J2R封存|成功率|碰撞/越界|指令对照|零特征|','|---|---|---|---|---|---|---|---|']
failures={}
for row in s['models']:
    seed=row['seed'];data=load(f's{seed}-sealed_test.json')
    lines.append(f"|{seed}|{row['validation']}|{row['before_sealed']}|{row['sealed']}|{row['success_rate']:.1%}|{row['collision_or_bounds']}|{row['instruction_pairs_successes']}/4|{row['zero_features_successes']}/12|")
    failures[str(seed)]={'reasons':dict(Counter(r['reason'] for r in data['results'] if not r['success'])), 'phases':dict(Counter(r['final_phase'] for r in data['results'] if not r['success'])), 'profiles':dict(Counter(r['disturbance']['profile'] for r in data['results'] if not r['success'])), 'cases':[r for r in data['results'] if not r['success']]}
lines+=['','## 各场景同场对照','', '|种子|场景|J2（训练前）|J2R（训练后）|','|---|---|---|---|']
names={'clean':'无扰动','pose':'2毫米噪声+100毫秒延迟','force':'短时外力','combined':'叠加扰动'}
for row in s['models']:
    for profile,value in row['profiles'].items():
        old=row['before_profiles'][profile];lines.append(f"|{row['seed']}|{names[profile]}|{old['successes']}/{old['episodes']}|{value['successes']}/{value['episodes']}|")
paired={}
lines+=['','## 逐场成功变化','', '|种子|旧失败→新成功|旧成功→新失败|同时成功|同时失败|','|---|---|---|---|---|']
for row in s['models']:
    seed=row['seed'];old=load(f's{seed}-before-sealed.json')['results'];new=load(f's{seed}-sealed_test.json')['results']
    assert [r['case_id'] for r in old]==[r['case_id'] for r in new]
    categories={'recovered':[],'regressed':[],'both_success':[],'both_failure':[]}
    for a,b in zip(old,new):
        key='both_success' if a['success'] and b['success'] else 'recovered' if b['success'] else 'regressed' if a['success'] else 'both_failure'
        categories[key].append(b['case_id'])
    paired[str(seed)]=categories
    lines.append(f"|{seed}|{len(categories['recovered'])}|{len(categories['regressed'])}|{len(categories['both_success'])}|{len(categories['both_failure'])}|")
(out/'paired-changes.json').write_text(json.dumps(paired,indent=2))
lines+=['','## 门禁与范围','', '每种子封存总成功率≥90%；无扰动≥90%，其余每类≥85%；碰撞/越界≤1%；指令4/4；零特征0/12；无扰动较旧模型下降≤5个百分点；675个扰动场景合计成功数不低于旧模型。固定最后权重，封存前冻结，未以封存挑选权重或种子。', '', '任务标准沿用J1：水平误差≤0.2米、高度误差≤0.1米、朝向误差≤16度、水平速度≤0.08米/秒、角速度≤0.08弧度/秒，同时保持2秒。阶段限时60秒，总限时120秒。物理位置不重置，阶段切换保持连续；策略只读神经特征。', '', '外力为水平0.006牛顿、偏航力矩0.00003牛米；起飞完成第5秒开始，每12秒施加2秒。方向由场景种子确定。通过MuJoCo积分，不写位置或速度。工程扰动未按真实无人机标定。', '', '每种子完整组合任务执行2048个真实训练动作，合计6144；另外每两个真实传感器样本标注一次神经状态用于监督学习，标签数不计作额外动作。原J2数据参与复习。冻结MaleCNS，仅更新两个读出，教师仅训练和规则基线使用。这不是果蝇全部突触端到端学习。固定高度、空房间、模拟外部定位；完整TS1、视觉定位、障碍物、真实飞行均未就绪。', '', '## 未通过项与失败分析','']
lines.extend(s['acceptance']['reasons'] or ['全部预注册门禁通过。'])
for seed,data in failures.items():lines.append(f"种子{seed}：{data['reasons']}；阶段{data['phases']}；场景{data['profiles']}。")
lines+=['','## 验证与查看','', f"自动化测试{tests}项通过。规则基线{load('rule-validation.json')['successes']}/100，均匀随机基线{load('random-sealed.json')['successes']}/300；基线不属于果蝇模型成绩。此前浏览器读取被工具安全策略拦截，尚无解除证据；本轮未重试或绕过，未完成实际点击验证，详见BROWSER_VERIFICATION.json。仅做离线回放数据与代码检查，训练和回放审计见TRAINING_AUDIT.json与REPLAY_AUDIT.json。", '', '打开 http://127.0.0.1:8765/tellosim ，先用Ctrl+R刷新整个页面，再点击J2R联合任务11、22、33。固定使用各自封存第4场叠加扰动，另外三个训练前按钮播放完全同场景J2模型。单场样例不能代替总体成绩。', '', '正式训练与评估入口：python reports/ts1_stability/run_campaign.py。已完成目录不可覆盖；再次实验必须使用隔离的新版本协议和输出。交付包回放无需外部大图，重训需data/male-v1.npz和既有WSL依赖；不是新机器安装验证。']
(out/'README.md').write_text('\n'.join(lines)+'\n');(out/'failure-analysis.json').write_text(json.dumps(failures,ensure_ascii=False,indent=2))
print('report generated')
