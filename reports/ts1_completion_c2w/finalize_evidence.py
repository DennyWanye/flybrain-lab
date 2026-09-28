from pathlib import Path
import json,hashlib,xml.etree.ElementTree as ET,subprocess,sys
root=Path.cwd();out=root/'reports/ts1_completion_c2w'
def read(p):return json.loads((root/p).read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
r2=read('reports/ts1_spatial_continuous/summary.json');assert r2['acceptance']['passed']
assert read('reports/ts1_spatial_continuous/TRAINING_AND_EVALUATION_AUDIT.json')['passed']
assert read('reports/ts1_spatial_continuous/FORMAL_REPLAY_AUDIT.json')['passed']
for p in ['BROWSER_C2W.json','T32-C2W.json','API_C2W.json']:assert json.loads((out/p).read_text())['status']=='PASS'
suites=ET.parse(out/'regression.xml').getroot().findall('testsuite');count=sum(int(x.get('tests','0')) for x in suites)
assert count>0 and all(int(x.get('failures','0'))==int(x.get('errors','0'))==int(x.get('skipped','0'))==0 for x in suites)
inventory=read('reports/ts1_control_development_v/historical-inventory.json');preserved=[];updated=[]
for name,entry in inventory.items():
 path=root/name
 if sha(path)==entry['sha256']:preserved.append(name)
 else:
  backup=out/'history'/name;assert backup.exists() and sha(backup)==entry['sha256'],name
  updated.append({'path':name,'preserved_copy':str(backup.relative_to(root)),'original_sha256':entry['sha256'],'current_sha256':sha(path)})
(out/'HISTORY_AUDIT.json').write_text(json.dumps({'all_original_bytes_preserved':True,'unchanged_files':len(preserved),'intentional_viewer_updates':updated},indent=2))
for name in ['T14-UI.json','T25-dynamics.json','T32-backend.json','T55-full.json','T55-long500.json']:assert read('reports/ts1_completion/'+name)['status']=='PASS'
assert read('reports/ts1_mlp_pilot/summary.json')['software_acceptance_T44']
rows=read('reports/ts1_completion/TEST_MATRIX.json')
for row in rows:
 row['historical_matrix']='reports/ts1_completion/TEST_MATRIX.json'
 row['current_regression']='reports/ts1_completion_c2w/regression.xml'
 row['code_version']='reports/ts1_completion_c2w/implementation-hashes.json'
 row['evidence_scope']='Historical acceptance retained where implementation is unchanged; current211-test software regression; historical learning experiments not rerun'
 if row['test_id']=='T32':row.update(current_additional_evidence=['T32-C2W.json','BROWSER_C2W.json'],actual='C2W full MaleCNS inference:0/30/60FPS-equivalent and disconnected consumer produce identical physical/neural/readout/clock states; current browser playback,pause,seek,FPS,pan verified; historical gradient-update evidence retained')
 if row['test_id']=='T25':row['compatibility_note']='Legacy physics/world/visual source unchanged. New opt-in command anchoring and120Hz hold are covered by current spatial_continuous and spatial_anchored tests plus full formal physics audits. Historical27-case dynamics matrix retained, not falsely relabelled as rerun.'
 if row['test_id']=='T44':row['compatibility_note']='MLP/altitude/C0 default executor unchanged; prior paired60-case pilot retained. No claim that this is a new C2W-trained MLP comparison or a connectome advantage.'
 if row['test_id']=='T60':row.update(status='PENDING_PACKAGE',artifact=['PACKAGE_AUDIT.json','PACKAGE_CONTENT_AUDIT.json'])
(out/'TEST_MATRIX.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
files={str(p):sha(p) for folder in ['flydrone','flyview','tests'] for p in Path(folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix in ['.py','.js','.html','.css']}
(out/'implementation-hashes.json').write_text(json.dumps(files,indent=2))
(out/'tracked-source.diff').write_bytes(subprocess.run(['git','diff','--binary'],capture_output=True,check=True).stdout)
(out/'requirements-runtime.lock.txt').write_text(subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True,text=True,check=True).stdout)
summary=json.loads((out/'summary.json').read_text());summary.update(SOFTWARE_FUNCTIONAL_ACCEPTANCE_PASSED=True,test_rows_passed=59,regression_tests=count,scope='R2 passed with original physical thresholds; R8 functional/browser/regression passed; package verification pending; real flight disabled')
(out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
impact={'status':'PASS','legacy_executor':'VisualSession/1.0 bytes unchanged; C2W opt-in AnchoredVisualSession; isolation regression passed','physics':'world.py and rigid.py unchanged; original T25 evidence remains applicable','sensors_and_neural_encoding':'raw26 inputs and encoders unchanged; stationary height mean only for phase management','new_continuous_hold':'120Hz evaluator clock verified with actual physical trajectories; no truth leakage into neural decisions','old_skills':'V1R/J2R source and checkpoint bytes preserved; current legacy tests pass; old formal learning scores not rerun','MLP':'same prior C0 pilot and default executor; unchanged source verified; not a new C2W MLP study','recording_and_async':'unchanged implementation; current regression and browser replay checked; old T14/T55 physicalUI stress evidence retained','regression_tests':count}
(out/'IMPACT_AND_RETENTION.json').write_text(json.dumps(impact,indent=2))
print(json.dumps({'regression_tests':count,'functional_acceptance':True,'rows_pending_package':1,'historical_files_preserved':len(inventory)}))
