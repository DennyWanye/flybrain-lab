"""Consolidate60-row TS1 evidence without re-running historical campaigns."""
from pathlib import Path
import json,hashlib,subprocess,sys,xml.etree.ElementTree as ET
root=Path.cwd();out=root/'reports/ts1_completion'
def read(p):return json.loads((root/p).read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
r1=read('reports/ts1_altitude_refined/summary.json');r2=read('reports/ts1_spatial_orientation/summary.json')
assert r1['MODEL_READY_FOR_NEXT_STAGE'] and r2['status']=='evaluated'
reg=ET.parse(out/'regression-c2q.xml').getroot();suites=reg.findall('testsuite');assert sum(int(x.get('tests','0')) for x in suites)==146 and all(int(x.get('failures','0'))==int(x.get('errors','0'))==0 for x in suites)
for name in ['T14-UI.json','T25-dynamics.json','T32-backend.json','T55-full.json','T55-long500.json']:
 assert json.loads((out/name).read_text())['status']=='PASS'
assert read('reports/ts1_mlp_pilot/summary.json')['software_acceptance_T44']
assert read('reports/ts1_completion/BROWSER_C2Q.json')['status']=='PASS'
old=read('reports/ts1_c1/TEST_MATRIX.json');rows=[]
updates={
'T14':('执行中Future取消/超时不停止设备，回执归属保持；浏览器关闭时cw90继续完成；结束模拟明确非设备停止回执',['T14-focused.xml','T14-UI.json','BROWSER_CHECK.json']),
'T25':('27项参数/饱和/外力矩阵通过；目标倾角限幅与实际动态角度分开报告，未将瞬态超调伪称硬状态上限',['T25-dynamics.json']),
'T32':('真实MaleCNS、物理轨迹和梯度更新在0/30/60帧消费及断开观察下逐字节等价；浏览器帧率/播放/暂停/断开重连实测',['T32-backend.json','BROWSER_CHECK.json','BROWSER_C2Q.json']),
'T44':('新刚体环境单种子普通MLP与V1R11配对60场景均60/60；来源分开，无连接组优势结论',['../ts1_mlp_pilot/summary.json','../ts1_mlp_pilot/TRAINING_AUDIT.json']),
'T55':('真实256MiB配额压力、有效前缀/缺失区间/缓冲释放；forward500 speed10实际51秒和29分块；浏览器跨分块到50秒及57.6秒末尾',['T55-full.json','T55-long500.json','BROWSER_CHECK.json']),
'T60':('本轮交付包按清单逐文件哈希；新目录解包，加载本包源码和真实模型、最新API/回放/静态依赖；现有WSL依赖，非新机器安装',['PACKAGE_AUDIT.json'])}
for previous in old:
 row=dict(previous);row['historical_evidence_root']='reports/ts1_c1';row['historical_row']=previous;row['current_regression']='reports/ts1_completion/regression-c2q.xml';row['code_version']='implementation-hashes.json'
 if row['test_id'] in updates:
  actual,artifacts=updates[row['test_id']];row.update(status='PASS',actual=actual,artifact=artifacts,elapsed_s=None,timing_note='See individual evidence; no fabricated per-row timing')
 else:row['evidence_scope']='Retained historical acceptance plus current146-test software regression; historical model scores not rerun'
 if row['test_id']=='T60':row['status']='PENDING_PACKAGE'
 rows.append(row)
(out/'TEST_MATRIX.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
files={str(p):sha(p) for folder in ['flydrone','flyview','tests'] for p in Path(folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix in ['.py','.js','.html','.css']}
(out/'implementation-hashes.json').write_text(json.dumps(files,indent=2))
lock=subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True,text=True,check=True).stdout;(out/'requirements-runtime.lock.txt').write_text(lock)
summary={'FULL_TS1_READY':False,'SOFTWARE_MATRIX_PASSED':False,'SOFTWARE_FUNCTIONAL_ACCEPTANCE_PASSED':True,'PACKAGE_VERIFICATION_PENDING':True,'ALTITUDE_TASK_LEARNED':True,'JOINT_3D_TASK_VERIFIED':r2['JOINT_3D_TASK_VERIFIED'],'SIMULATION_GOAL_READY':False,'REAL_FLIGHT_READY':False,'test_rows':60,'test_rows_passed':59,'regression_tests':146,'R1':r1['models'],'R2':r2['models'],'scope':'TS1 original software acceptance plus separate expanded3D learning gate; final flags require package audit'}
(out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
lines=['# TS1本轮验收与交付','', '本轮主体代码均由当前主代理完成；未调用plan-test、未委派代码。历史V1/J2R及失败C2原件保留，新版均使用新目录与划分。','',
'R1 V1R已通过：种子11/22/33封存298/300、300/300、300/300；边界均12/12，指令均4/4，零特征均0/12；原J2R12场逐场保持。',
'',f"R2 C2Q三维联合门槛：{'YES' if r2['JOINT_3D_TASK_VERIFIED'] else 'NO'}。详细种子成绩及未通过项见../ts1_spatial_orientation/README.md。C2首轮因模拟器180秒关闭传播错误中止，失败源码、日志及中间权重均保留；C2R修复终止传播，但正式指令门槛失败；C2P依据新开发案例改进导航编码，但边界仍未达标；C2Q依据独立开发记录改进朝向编码并重新训练，但正式边界和规则门槛仍失败；C2S/T/U仅修复阶段测量管理，均在开发门槛失败后保留，没有进入新正式评估；其中S/U迁移的权重逐张量等于C2Q，新训练动作0。C2U在90场开发进度时已有2个失败，无法达到127/128要求，按精确进程终止并保存部分结果。历次失败均保留，未调封存。",
'', '未完成：R2联合模型门槛仍为NO；R8整体模拟交付门槛因此保持NO。交付包仅确认现有软件和证据可复核，不能视为可靠联合控制已完成。',
'', 'R3取消等待/关闭观看不等于停止；R4完成动力学矩阵；R5完成普通MLP pilot；R6浏览器及观察独立性；R7真实256MiB压力与长命令回放均有当前证据。',
'', 'T01—T60见TEST_MATRIX.json；原55项保留历史适用证据并通过本轮146项回归，新增5项完整证据。T60以PACKAGE_AUDIT.json最终解包检查为准。历史学习成绩没有重新训练或重新评估。',
'', '动力学限制的含义：加速度请求、推力和力矩受控制器限幅；倾角目标受限，实际姿态按刚体动力学积分，在极端50m/s²请求测试下有瞬态超调。外力独立施加，未钳制真实状态伪造稳定。',
'', '普通MLP26→128→128→9，没有MaleCNS/神经reservoir；与V1R11同60场景均60/60，训练数据不同，只有单种子工程对照，不证明连接组优势。',
'', '回放：http://127.0.0.1:8765/tellosim 。最新三维回放与V1R上升/下降入口，旧失败成绩仍保留。真实适配器继续禁用。',
'', '真机另需真实定位、标定、适配器与shadow验证、受控真机测试四阶段；模拟验收不代表整个无人机目标完成。']
(out/'README.md').write_text('\n'.join(lines)+'\n')
print('matrix staged; package verification still required')
