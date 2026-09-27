"""Finish local evidence after the bounded campaign exits successfully."""
from pathlib import Path
import json,subprocess,sys
root=Path.cwd();out=root/'reports/ts1_robust'
summary=json.loads((out/'summary.json').read_text());assert summary['status']=='evaluated'
history=json.loads((out/'process-cleanup.json').read_text())
assert len(history)==7 and all(x['reaped'] and x['returncode']==0 for x in history.values())
assert all(not Path(f"/proc/{x['pid']}").exists() for x in history.values())
(out/'PROCESS_VERIFICATION.json').write_text(json.dumps({'all_campaign_children_reaped':True,'all_recorded_pids_absent':True,'jobs':history,'viewer_kept_running_port':8765},indent=2))
for name in ('audit_training.py','audit_replays.py','build_report.py','plot_results.py'):
    subprocess.run([sys.executable,str(out/name)],check=True)
passed=summary['MODEL_READY_FOR_NEXT_STAGE'];failure=json.loads((out/'failure-analysis.json').read_text())
count=sum(sum(d['reasons'].values()) for d in failure.values())
lines=['# 本轮结论与下一阶段','',f"J2 扰动阶段正式门禁：{'通过' if passed else '未通过'}。本轮封存、协议、权重不可改写以追认通过。",'',f"900场J2封存共有{count}场失败；具体阶段和扰动类别见failure-analysis.json。与J1R的对照使用完全相同的300场景和扰动种子。",'', '未通过项：'+('；'.join(summary['acceptance']['reasons']) or '无'),'','本轮只训练导航和转向两个读出，冻结MaleCNS。固定高度、空房间、工程外部定位，未进行真实飞行。']
if passed:
    lines+=['','建议下一阶段扩大到不同目标高度的三维位置与朝向任务，先增加垂直动作的独立训练与验收，再与现有技能组合。继续保留干净与扰动场景回归，使用新的训练、验证和封存种子。']
else:
    lines+=['','开发验证记录已观察到临近目标时往返移动、顺逆时针交替的现象：例如种子11的j2-val-021最后十个动作为7,8,0,7,8,0,7,8,0,7。下一轮先用开发验证追踪神经特征、动作概率和观测延迟，区分表示波动与停止标签问题，再选择有针对性的训练改进；不直接用位置规则接管动作，也不放宽验收标准。使用新训练场景和新封存集，不用本轮封存挑权重。暂不进入高度扩展，上一轮J1R通过记录保留。']
lines+=['','浏览器工具因URL安全策略阻塞，实际页面点击验证尚未完成；本轮只验证离线代码、数据与交付包，未以旧截图替代新证据。']
(out/'NEXT_STAGE.md').write_text('\n'.join(lines)+'\n')
print('local evidence finalized')
