"""Finish evidence once all owned campaign children have exited."""
from pathlib import Path
import json,subprocess,sys
root=Path.cwd();out=root/'reports/ts1_stability';s=json.loads((out/'summary.json').read_text())
history=json.loads((out/'process-cleanup.json').read_text())
assert len(history)==7 and all(v['reaped'] and v['returncode']==0 for v in history.values())
assert all(not Path(f"/proc/{v['pid']}").exists() for v in history.values())
memory=json.loads((out/'owned-process-memory-sample.json').read_text())
(out/'PROCESS_VERIFICATION.json').write_text(json.dumps(dict(all_campaign_children_reaped=True,all_recorded_pids_absent=True,jobs=history,observed_private_kib_before_exit=sum(v['private_kib'] for v in memory.values()),memory_note='Snapshot of these owned evaluation processes before exit; all corresponding PIDs now absent.'),indent=2))
for name in ['audit_training.py','audit_replays.py','build_report.py','plot_results.py']:
    subprocess.run([sys.executable,str(out/name)],check=True)
passed=s['MODEL_READY_FOR_NEXT_STAGE']
lines=['# J2R 结论与下一阶段','',f"MODEL_READY_FOR_NEXT_STAGE = {'YES' if passed else 'NO'}",'',
       '本轮使用连续组合任务与密集神经状态监督，历史J2权重、源代码和封存结果保持不变。',
       '门禁未通过项：'+('；'.join(s['acceptance']['reasons']) or '无'),'',
       '固定高度、空房间、模拟外部定位；冻结MaleCNS，只训练导航与转向读出。完整TS1和真机尚未就绪。']
if passed:lines+=['','下一阶段先加入独立的垂直定位技能，验证高度控制，再组合三维位置、朝向与保持任务。保留本阶段全部扰动门禁，使用新训练/验证/封存场景。']
else:lines+=['','下一轮应使用开发验证分析剩余失败，检查连续神经特征对微小目标误差的可分辨性，以及导航到位后转向/恢复的数据覆盖。不以本轮封存选权重、不直接用位置规则接管，也不放宽阈值。暂不扩大到高度训练。']
lines+=['','浏览器策略阻塞尚未解除，本轮没有实际页面点击验证。离线回放/代码/交付包检查不等同于浏览器交互验证。']
(out/'NEXT_STAGE.md').write_text('\n'.join(lines)+'\n')
print('J2R local evidence finalized')
