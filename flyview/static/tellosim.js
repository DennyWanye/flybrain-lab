import * as THREE from 'three';
import {OrbitControls} from 'three/addons/OrbitControls.js';

const $ = id => document.getElementById(id);
const text = (id, value) => { $(id).textContent = value; };
const fmt = (x, n=2) => Number.isFinite(x) ? x.toFixed(n) : '—';
const api = '/api/tellosim/';
const isSdkObservation = schema => ['tellosim.observation26/2.0','tellosim.heading_observation26/1.0','tellosim.joint_observation26/1.0', 'tellosim.altitude_observation26/1.0'].includes(schema);
let token, manifest, currentRun, epoch, live=false, playing=false, pausedLive=false;
let simTime=0, lastAnimation=0, replayBusy=false, generation=0, lastSeq=-1, skipped=0;
let sceneGroup, drone, ghost, trajectory, activeFrame={}, recentTrail=[], commandRows=[];
let knownRuns=[];
const cache=new Map();

function notice(message, error=false) {text('notice',message);$('notice').classList.toggle('error',error);}
async function request(path, body) {
  const options=body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json','X-Sandbox-Token':token},body:JSON.stringify(body)};
  const response=await fetch(api+path,options);
  if(!response.ok) {let info=await response.json();throw Error(info.error||response.statusText);}
  return response.json();
}
function attempt(fn) {return async (...args)=>{try{await fn(...args);}catch(error){notice(error.message,true);}};}

const renderer=new THREE.WebGLRenderer({antialias:true,alpha:false});
renderer.setPixelRatio(Math.min(devicePixelRatio,2));
renderer.setClearColor(0x0e1928);
$('scene').append(renderer.domElement);
renderer.domElement.setAttribute('aria-label','三维模拟画布');
const scene=new THREE.Scene();
const camera=new THREE.PerspectiveCamera(45,1,.02,100);
camera.up.set(0,0,1);
const controls=new OrbitControls(camera,renderer.domElement);
controls.enableDamping=true; controls.minDistance=1; controls.maxDistance=25;
controls.addEventListener('change',()=>{
  $('scene').dataset.camera=camera.position.toArray().map(v=>v.toFixed(3)).join(',');
  $('scene').dataset.cameraTarget=controls.target.toArray().map(v=>v.toFixed(3)).join(',');
});
scene.add(new THREE.HemisphereLight(0xd9f5ff,0x2a3a4a,2.2));
const light=new THREE.DirectionalLight(0xffffff,2.5);light.position.set(-3,-4,7);scene.add(light);
function resetCamera(){const damping=controls.enableDamping;controls.enableDamping=false;controls.update();camera.position.set(7,-9,7);controls.target.set(0,0,1);controls.update();controls.enableDamping=damping;}
resetCamera();
new ResizeObserver(()=>{const {width,height}=$('scene').getBoundingClientRect();renderer.setSize(width,height);camera.aspect=width/height;camera.updateProjectionMatrix();}).observe($('scene'));

function mesh(geometry,color,opacity=1){return new THREE.Mesh(geometry,new THREE.MeshStandardMaterial({color,transparent:opacity<1,opacity,roughness:.7,depthWrite:opacity===1}));}
function line(points,color){return new THREE.Line(new THREE.BufferGeometry().setFromPoints(points.map(p=>new THREE.Vector3(...p))),new THREE.LineBasicMaterial({color}));}
function makeDrone(wire=false){
  const group=new THREE.Group();
  const body=mesh(new THREE.BoxGeometry(.2,.2,.09),wire?0xf4c883:0x8ce8d2,wire?.4:1);
  if(wire)body.material.wireframe=true;
  group.add(body);
  for(const x of [-.12,.12])for(const y of [-.12,.12]){
    const ring=mesh(new THREE.TorusGeometry(.07,.01,6,20),wire?0xf4c883:0x658f9b,wire?.35:1);
    ring.position.set(x,y,.03);group.add(ring);
  }
  group.add(new THREE.ArrowHelper(new THREE.Vector3(1,0,0),new THREE.Vector3(0,0,.04),.42,wire?0xf4c883:0xee806d,.1,.07));
  return group;
}
function buildScene(config){
  if(sceneGroup){scene.remove(sceneGroup);sceneGroup.traverse(o=>{o.geometry?.dispose();if(o.material) o.material.dispose();});}
  sceneGroup=new THREE.Group();scene.add(sceneGroup);
  const [sx,sy,sz]=config.room_size_m;
  const floor=mesh(new THREE.PlaneGeometry(sx,sy),0x172e3b);floor.position.z=-.002;sceneGroup.add(floor);
  for(let x=-sx/2;x<=sx/2+.01;x+=.5)sceneGroup.add(line([[x,-sy/2,.002],[x,sy/2,.002]],0x2c4656));
  for(let y=-sy/2;y<=sy/2+.01;y+=.5)sceneGroup.add(line([[-sx/2,y,.002],[sx/2,y,.002]],0x2c4656));
  const box=new THREE.BoxGeometry(sx,sy,sz);const outline=new THREE.LineSegments(new THREE.EdgesGeometry(box),new THREE.LineBasicMaterial({color:0x43677c}));box.dispose();outline.position.z=sz/2;sceneGroup.add(outline);
  if(config.solid_walls){
    for(const [size,position] of [[[sx,.025,sz],[0,sy/2,sz/2]],[[.025,sy,sz],[-sx/2,0,sz/2]],[[sx,.025,sz],[0,-sy/2,sz/2]],[[.025,sy,sz],[sx/2,0,sz/2]]]){
      const wall=mesh(new THREE.BoxGeometry(...size),0x7293ad,.055);wall.position.set(...position);sceneGroup.add(wall);
    }
  }
  for(const o of config.obstacles){const obstacle=mesh(new THREE.BoxGeometry(...o.half_extents_m.map(v=>v*2)),0x976e56,.85);obstacle.position.set(...o.center_m);sceneGroup.add(obstacle);}
  const pad=config.landing_pad;const center=pad.center_xy_m||pad.center_m;
  const padMesh=mesh(new THREE.RingGeometry(pad.radius_m-.025,pad.radius_m,48),0x79c9ba);padMesh.position.set(...center,.008);sceneGroup.add(padMesh);
  sceneGroup.add(line([[center[0]-.12,center[1],.01],[center[0]+.12,center[1],.01]],0x79c9ba));
  const altitudeTask=config.task_kind==='altitude_hold';const target=mesh(altitudeTask?new THREE.CylinderGeometry(config.target_radius_m,config.target_radius_m,2*config.height_tolerance_m,32):new THREE.SphereGeometry(config.target_radius_m,24,16),0xb9a3fa,.18);if(altitudeTask)target.rotation.x=Math.PI/2;target.position.set(...config.target_xyz_m);sceneGroup.add(target);
  const ring=mesh(new THREE.RingGeometry(config.target_radius_m-.015,config.target_radius_m,48),0xb5a1f3);ring.position.set(config.target_xyz_m[0],config.target_xyz_m[1],.015);sceneGroup.add(ring);
  if(Number.isFinite(config.target_yaw_rad)){
    const targetDirection=new THREE.Vector3(Math.cos(config.target_yaw_rad),Math.sin(config.target_yaw_rad),0);
    sceneGroup.add(new THREE.ArrowHelper(targetDirection,new THREE.Vector3(...config.target_xyz_m).add(new THREE.Vector3(0,0,.18)),.85,0xb9a3fa,.18,.12));
  }
  const axes=new THREE.AxesHelper(.65);axes.position.set(-sx/2,-sy/2,.015);sceneGroup.add(axes);
  drone=makeDrone();ghost=makeDrone(true);ghost.visible=false;sceneGroup.add(drone,ghost);
  trajectory=line([],0x6eb9ac);sceneGroup.add(trajectory);
  text('room-kind',config.solid_walls?'MuJoCo · 实体墙体':'Golden · 边界线（无实体墙）');
}

function drawTopdown(position,yaw,trail,sensor){
  const c=$('topdown').getContext('2d');const w=220; c.clearRect(0,0,w,w);
  const [sx,sy]=manifest.scene.room_size_m;const scale=184/Math.max(sx,sy);
  const xy=p=>[110+p[0]*scale,110-p[1]*scale];
  c.strokeStyle='#3a586c';c.strokeRect(110-sx*scale/2,110-sy*scale/2,sx*scale,sy*scale);
  c.fillStyle='#8f7562';for(const o of manifest.scene.obstacles){const p=xy(o.center_m);c.fillRect(p[0]-o.half_extents_m[0]*scale,p[1]-o.half_extents_m[1]*scale,o.half_extents_m[0]*scale*2,o.half_extents_m[1]*scale*2);}
  const t=xy(manifest.scene.target_xyz_m);c.strokeStyle='#b7a1f3';c.beginPath();c.arc(...t,manifest.scene.target_radius_m*scale,0,Math.PI*2);c.stroke();
  if(Number.isFinite(manifest.scene.target_yaw_rad)){const a=manifest.scene.target_yaw_rad;c.save();c.translate(...t);c.rotate(-a);c.strokeStyle='#b7a1f3';c.beginPath();c.moveTo(0,0);c.lineTo(28,0);c.lineTo(20,-5);c.moveTo(28,0);c.lineTo(20,5);c.stroke();c.restore();}
  c.strokeStyle='#639f99';c.beginPath();trail.forEach((p,i)=>{const a=xy(p);i?c.lineTo(...a):c.moveTo(...a);});c.stroke();
  if(sensor?.valid&&sensor.position_m){const s=xy(sensor.position_m);c.strokeStyle='#efc58b';c.strokeRect(s[0]-5,s[1]-5,10,10);}
  const p=xy(position);c.save();c.translate(...p);c.rotate(-yaw);c.fillStyle='#95e7d0';c.beginPath();c.moveTo(9,0);c.lineTo(-5,-5);c.lineTo(-3,0);c.lineTo(-5,5);c.closePath();c.fill();c.restore();
  c.font='11px sans-serif';c.fillStyle='#98b0c2';c.fillText('俯视 · X → Y ↑',12,18);c.fillText('1 m',12,204);c.strokeStyle='#98b0c2';c.beginPath();c.moveTo(45,201);c.lineTo(45+scale,201);c.stroke();
}

function renderFrame(frame, trail=[]){
  if(!manifest||!frame.truth)return;
  if(manifest.observer_only&&frame.scene&&manifest.episode_id!==frame.episode_id){
    manifest.scene=frame.scene;manifest.episode_id=frame.episode_id;buildScene(frame.scene);recentTrail=[];
  }
  text('runid',`run ${currentRun} · epoch ${epoch.slice(0,8)} · env ${frame.env_id??manifest.env_id??0} · ${frame.episode_id??manifest.episode_id??'episode-0'}`);
  if(manifest.observer_only){const p=frame.training_progress||{};text('watch-progress',`环境 ${frame.env_id} · ${frame.episode_id} · 本阶段已提交动作 ${p.options??0} · 本局动作 ${frame.episode_steps??0} · 更新 ${p.updates??0} · 本局回报 ${fmt(frame.episode_return)} · ${frame.termination_reason||p.status||'运行中'} · 只读观察`);}
  activeFrame=frame;
  const t=frame.truth,p=t.position_m,yaw=t.yaw_rad||0;
  drone.position.set(...p);
  if(t.quaternion_wxyz&&manifest.scene?.controller==='rigid_body_thrust_v2'){const [w,x,y,z]=t.quaternion_wxyz;drone.quaternion.set(x,y,z,w);}else drone.rotation.set(0,0,yaw);
  ghost.visible=!!(frame.sensor?.valid&&frame.sensor.position_m);
  if(ghost.visible){ghost.position.set(...frame.sensor.position_m);ghost.quaternion.copy(drone.quaternion);}
  trajectory.geometry.dispose();trajectory.geometry=new THREE.BufferGeometry().setFromPoints(trail.map(p=>new THREE.Vector3(...p)));
  drawTopdown(p,yaw,trail,frame.sensor);
  $('scene').dataset.position=p.map(v=>v.toFixed(6)).join(',');$('scene').dataset.tick=frame.sim_tick;
  text('clock',`${fmt(frame.time_s)} s · tick ${frame.sim_tick}`);
  text('position',`X ${fmt(p[0])} · Y ${fmt(p[1])} · Z ${fmt(p[2])} m`);
  text('speed-readout',`${fmt(Math.hypot(...(t.velocity_mps||[0,0,0])))} m/s`);
  text('distance',`${fmt(Math.hypot(...p.map((v,i)=>v-manifest.scene.target_xyz_m[i])))} m`);
  text('hold',`${fmt(frame.stable_hold_s)} / ${manifest.scene.stable_hold_required_s} s`);
  $('joint-phase').hidden=!frame.task_phase;
  text('joint-phase',frame.task_phase==='navigation'?`阶段 1 / 2：到达指定位置 · 到位保持 ${fmt(frame.navigation_hold_s,1)} / 2.0 秒`:frame.task_phase==='heading'?`阶段 2 / 2：转到指定朝向 · 联合保持 ${fmt(frame.stable_hold_s,1)} / 2.0 秒`:'');
  const altitude=frame.altitude;$('altitude-metrics').hidden=!altitude;
  for(const [id,key,unit] of [['altitude-target','target_z_m','m'],['altitude-measured','measured_z_m','m'],['altitude-error','error_m','m'],['altitude-speed','vertical_speed_mps','m/s']])text(id,Number.isFinite(altitude?.[key])?`${fmt(altitude[key],3)} ${unit}`:'—');
  const heading=frame.heading;$('heading-metrics').hidden=!heading;
  for(const [id,key] of [['heading-target','target_yaw_rad'],['heading-measured','measured_yaw_rad'],['heading-error','error_rad'],['heading-tolerance','tolerance_rad']]){
    const degrees=heading?.[key]*180/Math.PI;
    const angle=['target_yaw_rad','measured_yaw_rad'].includes(key)?((degrees+180)%360+360)%360-180:degrees;
    text(id,Number.isFinite(heading?.[key])?`${fmt(angle,1)}°`:'—');
  }
  text('pose-valid',frame.sensor?(frame.sensor.valid?'有效':'无效 · 不补零'):'not_recorded');
  text('pose-age',frame.sensor?`${fmt(frame.sensor.age_s,3)} s`:'—');
  text('sensor-source',isSdkObservation(manifest.observation_schema)?'模型读取模拟外部定位与 SDK 状态。位置／速度来自测量；电量使用未标定的工程模型。':frame.sensor?'实心为 MuJoCo 真值；线框为带噪声、延迟和掉帧的外部位姿。电量未建模。':'此回放未记录外部位姿传感器；不显示虚构观测。');
  text('truth',JSON.stringify(t,null,2));
  text('observation-tick',`遥测 tick ${frame.observation_tick??frame.sim_tick}`);
  text('observation-note',manifest.feature_source&&manifest.feature_source!=='reservoir'?'诊断对照：'+manifest.feature_source+'；不作为果蝇策略成绩。':isSdkObservation(manifest.observation_schema)?'SDK9：26 维测量观测 → 34 编码通道 → 果蝇网络；策略仅接收神经 v/trace。无效组置零且有效位为 0。':'旧版 pose/1.0：前 15 项有效，后 11 项保留。Golden 策略实际使用 8 维输入。');
  const obs=$('observations');obs.replaceChildren();
  (frame.observation||[]).forEach((v,i)=>{const cell=document.createElement('span');const valid=frame.observation_valid?.[i]!==false;cell.textContent=`${String(i).padStart(2,'0')} ${valid?fmt(v,4):'reserved / invalid'}`;cell.title=(frame.observation_names||manifest.observation_names)?.[i]||`channel ${i}`;cell.classList.toggle('invalid',!valid);obs.append(cell);});
  renderOperation(frame.operation,frame.time_s);
  const probabilities=$('probabilities');probabilities.replaceChildren();
  const labels=['STOP','FORWARD','BACK','LEFT','RIGHT','UP','DOWN','CW','CCW'];
  (frame.policy?.probabilities||[]).forEach((prob,i)=>{
    const row=document.createElement('div');row.className='prob-row';
    const label=document.createElement('label');label.textContent=`${i===frame.policy.selected_action?'▸':' '} ${labels[i]}`;
    const meter=document.createElement('meter');meter.min=0;meter.max=1;meter.value=prob;
    const value=document.createElement('b');value.textContent=`${(prob*100).toFixed(1)}%`;row.append(label,meter,value);probabilities.append(row);
  });
  text('policy-source',frame.policy?(manifest.feature_source&&manifest.feature_source!=='reservoir'?'对照 MLP':'MaleCNS reservoir'):'not_recorded');
  text('policy-value',frame.policy?`${frame.policy.value_trained===false?'V 未训练':`V = ${fmt(frame.policy.value_estimate,4)}`} · 策略输入 ${frame.policy.feature_dim||frame.brain_observation?.length||'—'}D · ${frame.policy.input_source||'legacy'}`:(manifest.observer_only?'训练采集帧：本帧未记录模型动作概率，可能来自示范或训练扰动；实际命令见下方状态栏。':manifest.policy_source?.includes('random')?'均匀随机基线；动作不读取果蝇读出。':manifest.policy_source?.includes('rule')?'规则控制基线；不代表学习模型。':isSdkObservation(manifest.observation_schema)?'当前帧处于任务管理器阶段，尚未记录模型决策。':'脚本 / 手动命令，无模型决策。'));
  text('reward',frame.reward?`上一个完成动作奖励 ${fmt(frame.reward.reward_total,4)} · ${JSON.stringify(frame.reward.reward_components)}`:'奖励：not_recorded');
  text('mask',frame.policy?.mask?`允许动作：${labels.filter((_,i)=>frame.policy.mask[i]).join(' / ')}`:frame.policy?'动作掩码：not_recorded':'动作掩码：不适用');
  renderNeurons(frame);
  if(live){
    text('stream-status',`latest snapshot · seq ${frame.seq} · 跳过 ${skipped} 帧${frame.recording_partial?' · PARTIAL':''}`);
    const busy=frame.operation&&['sent','unknown_execution'].includes(frame.operation.client);
    document.querySelectorAll('[data-command]').forEach(b=>{b.disabled=frame.finished||(busy&&b.dataset.command!=='stop');});
    text('command-history',(frame.history||[]).slice(-7).map(o=>`${o.operation_id}  ${o.wire.padEnd(12)}  ${o.device} / ${o.client}`).join('\n'));
    $('command-history').style.whiteSpace='pre-wrap';
  }
  text('time-label',`${fmt(frame.time_s)} / ${fmt(live?frame.time_s:manifest.duration_s)} s`);
  $('timeline').value=frame.time_s;
}
function renderOperation(op,time){
  text('cmd-wire',op?.wire||'—');text('cmd-id',(op&&['command','takeoff','land'].includes(op.wire?.split(' ')[0])?'mission_manager · ':'')+(op?.operation_id||'等待命令'));
  text('cmd-device',op?.device||'—');text('cmd-client',op?.client||'—');
  $('cmd-client').classList.toggle('error',op?.client==='unknown_execution');
  text('cmd-execution',op?.completed_tick!==undefined?`完成 tick ${op.completed_tick}`:op?`开始 tick ${op.sent_tick}`:'—');
  text('cmd-response',op?.response??(op?.client==='unknown_execution'?'回执丢失 / 不自动重试':'尚无回执'));
  text('command-age',op?`${fmt(Math.max(0,Math.min(time,(op.completed_tick??Infinity)/120)-op.sent_tick/120))} s`:'');
}
function renderNeurons(frame){
  const neurons=frame.brain?.neurons?.slice(0,64)||[];
  const c=$('neurons').getContext('2d');c.clearRect(0,0,420,110);
  text('neural-tick',neurons.length?`tick ${frame.neural_tick} / substep ${frame.neural_substep}`:'not_recorded');
  text('neural-status',neurons.length?`${neurons.length} 个记录神经元 · 柱高为 v，亮点为 spike`:'当前来源没有神经活动记录。');
  neurons.forEach((n,i)=>{const x=8+i*6.3;const h=Math.max(0,Math.min(80,n.v*130));c.fillStyle='#67b7ac';c.fillRect(x,96-h,4.5,h);if(n.spike){c.fillStyle='#ffd58b';c.fillRect(x,8,4.5,5);}});
  const container=$('neuron-table');container.replaceChildren();
  if(neurons.length){const table=document.createElement('table');let head=table.insertRow();['ID','v','spike','trace'].forEach(s=>{const th=document.createElement('th');th.textContent=s;head.append(th);});neurons.forEach(n=>{let row=table.insertRow();[n.neuron_id,fmt(n.v,4),n.spike,fmt(n.trace,4)].forEach(s=>{row.insertCell().textContent=s;});});container.append(table);}
}

async function chunk(kind, descriptor, selectedRun=currentRun){
  const key=selectedRun+'/'+descriptor.file;
  if(cache.has(key))return cache.get(key);
  const response=await fetch(`${api}runs/${selectedRun}/chunks/${descriptor.file}`);
  if(!response.ok)throw Error(`回放缺块或校验失败：${descriptor.file}（HTTP ${response.status}）`);
  const data=await response.text();const rows=data.trim().split('\n').filter(Boolean).map(JSON.parse);
  cache.set(key,rows);if(cache.size>64)cache.delete(cache.keys().next().value);return rows;
}
async function around(kind,tick,range=0){
  const chunks=manifest.streams[kind]||[];
  let candidates=chunks.filter(c=>c.first_tick<=tick&&c.last_tick>=tick-range);
  if(!candidates.length){const previous=chunks.filter(c=>c.first_tick<=tick).at(-1);if(previous)candidates=[previous];}
  // Include previous chunk when a boundary has no exact sample at the requested tick.
  const index=chunks.indexOf(candidates[0]);if(index>0)candidates=[chunks[index-1],...candidates];
  return (await Promise.all(candidates.map(c=>chunk(kind,c)))).flat().filter(r=>r.sim_tick<=tick);
}
async function seek(seconds){
  if(live||!manifest)return;
  const ticket=generation;const tick=Math.floor(seconds*120+1e-5);
  const [positions,transitions,operations,neurals]=await Promise.all([around('trajectory',tick,1200),around('transition',tick,1200),around('operation',tick),around('neural',tick)]);
  if(ticket!==generation)return;
  const decision=transitions.filter(r=>!r.result_only).at(-1)||{};
  let pos=positions.at(-1);
  // Early recordings saved the reset pose in transition tick zero.
  if(!pos&&tick===0&&decision.truth){const p=decision.truth.position_m;pos={x_m:p[0],y_m:p[1],z_m:p[2],yaw_rad:decision.truth.yaw_rad,velocity_mps:decision.truth.velocity_mps,sim_tick:0,time_s:0};}
  if(!pos)throw Error('当前时间的物理轨迹未记录');
  const result=transitions.filter(r=>r.result_only).at(-1)||{};
  const neuron=neurals.at(-1)||{};
  // A completed option may precede a newer measured frame. Only its reward
  // persists; do not overwrite current observation/heading/hold with old state.
  const latest=transitions.at(-1)||{};
  const state={...decision,...latest,reward:result.reward||latest.reward,operation:operations.at(-1)?.operation||decision.operation,
    brain:neuron.brain||decision.brain,neural_tick:neuron.sim_tick,neural_substep:neuron.neural_substep,
    truth:{position_m:[pos.x_m,pos.y_m,pos.z_m],velocity_mps:pos.velocity_mps||[0,0,0],yaw_rad:pos.yaw_rad,quaternion_wxyz:pos.quaternion_wxyz,angular_velocity_body_rad_s:pos.angular_velocity_body_rad_s,collision:pos.collision,out_of_bounds:pos.out_of_bounds,ground_contact:pos.ground_contact},
    sim_tick:pos.sim_tick,time_s:pos.time_s,observation_tick:decision.sim_tick};
  renderFrame(state,positions.filter((_,i)=>i%6===0).map(p=>[p.x_m,p.y_m,p.z_m]));
  text('stream-status',`${manifest.partial?'PARTIAL':'完整记录'} · 120 Hz 轨迹 · ${cache.size} 个缓存分块`);
  text('command-history',operations.slice(-5).map(r=>`tick ${r.sim_tick}  ${r.operation.wire}  ${r.operation.device} / ${r.operation.client}`).join(' | '));
}
async function loadRun(runId,isLive=knownRuns.some(r=>r.run_id===runId&&(r.active||r.observer_only))){
  generation++;playing=false;live=isLive;pausedLive=false;currentRun=runId;lastSeq=-1;skipped=0;simTime=0;
  $('timeline').disabled=true;$('play').disabled=true;$('command-jump').disabled=true;$('scene').dataset.loading='true';
  const loadTicket=generation;
  const loaded=await request(`runs/${runId}/manifest`);
  if(loadTicket!==generation)return;
  if(!['tellosim.view/2.0','tellosim.observer/1.0'].includes(loaded.schema_version)){
    $('scene').style.visibility='hidden';throw Error(`不支持的回放格式：${loaded.schema_version??'缺少版本'}。请选择已支持的记录。`);
  }
  $('scene').style.visibility='visible';
  manifest=loaded;epoch=manifest.epoch;cache.clear();recentTrail=[];
  buildScene(manifest.scene);resetCamera();
  text('mode',live?'LIVE':'REPLAY');text('source',manifest.policy_source);text('runid',`run ${runId} · epoch ${epoch.slice(0,8)} · env ${manifest.env_id??0}`);
  text('physics-version',manifest.scene?.controller==='rigid_body_thrust_v2'?'物理：六自由度 / 机身推力 v2':'物理：旧版近似环境');
  text('graph',manifest.graph_sha256?`graph ${manifest.graph_sha256.slice(0,12)}`:'graph: not_recorded');
  const outcomes={success:'成功（此场景）',task_deadline:'任务超时（此场景失败）',golden_success:'模型成功（此场景）',golden_failure:'模型失败',script_completed:'脚本完成（非模型成绩）',script_failed:'脚本失败',closed_by_user:'手动结束',sandbox_time_limit_180s:'沙盒时间上限',interrupted:'记录中断 / PARTIAL'};
  text('outcome',live?'运行中':`记录结果：${outcomes[manifest.outcome]||manifest.outcome||'未记录'}`);
  text('provenance',JSON.stringify(manifest,null,2));
  $('manual').hidden=!(live&&manifest.policy_source==='manual_sandbox');
  $('rate').disabled=live;$('timeline').max=manifest.duration_s||1;
  text('play',live?'暂停观察':'播放');
  $('runs').value=runId;
  notice(manifest.feature_source&&manifest.feature_source!=='reservoir'
    ? '诊断对照模型：'+manifest.feature_source+'。此记录不代表果蝇神经网络控制效果。'
    : isSdkObservation(manifest.observation_schema)
    ? 'SDK9 控制链：26 维测量观测 → 34 通道编码 → 冻结果蝇网络 → 动作读出 → 模拟 SDK。训练与评估只读；记录的是实际神经活动和物理运动。'
    : manifest.scene.solid_walls
    ? 'MuJoCo 工程近似模拟：房间与障碍参与碰撞。脚本／手动命令用于演示，不代表模型成绩。'
    : '真实记录的模型回放：实心无人机为模拟真值，边界线不是实体墙。此记录未采集外部位姿噪声；输入与神经采样范围见下方数据来源。');
  $('command-jump').replaceChildren();
  if(!live){
    commandRows=(await Promise.all((manifest.streams.operation||[]).map(c=>chunk('operation',c)))).flat().filter(r=>r.operation.client==='sent');
    if(loadTicket!==generation)return;
    commandRows.forEach(r=>{const option=document.createElement('option');option.value=r.time_s;option.textContent=`${fmt(r.time_s)} s · ${r.operation.wire}`;$('command-jump').append(option);});
    await seek(0);
  }else await poll();
  if(loadTicket===generation){$('timeline').disabled=live;$('play').disabled=!!(live&&manifest.observer_only&&activeFrame.finished);$('command-jump').disabled=live;$('scene').dataset.loading='false';}
}
let latestTrainingRun=null;
async function catalog(){
  const data=await request('runs');const selected=currentRun;$('runs').replaceChildren();
  knownRuns=data.runs;renderWatchGroups(data.runs);
  const altitude=data.altitude;text('altitude-ready',altitude?.ALTITUDE_TASK_LEARNED?'V1 · 独立高度任务通过':altitude?.status==='evaluated'?'V1 · 未达标':'V1 · 尚未完成评估');
  const altitudeBox=$('altitude-summary');altitudeBox.replaceChildren();const altitudeNote=document.createElement('p');altitudeNote.textContent=altitude?.note||'正在建立独立高度控制技能。原J2R导航与朝向结果保留；独立高度通过后再整合三维任务。';altitudeBox.append(altitudeNote);
  for(const row of altitude?.models||[]){const item=document.createElement('p');item.textContent=`种子 ${row.seed}：验证 ${row.validation} · 封存 ${row.sealed} · 碰撞/越界 ${row.collision_or_bounds} · 高度指令 ${row.instruction_pairs_successes}/4 · 边界 ${row.boundary_successes}/12 · 零特征 ${row.zero_features_successes}/12 · 原组合任务 ${row.legacy_exact?'逐场保持':'存在差异'}`;altitudeBox.append(item);}
  if(altitude?.acceptance?.reasons?.length){const item=document.createElement('p');item.textContent='未通过项：'+altitude.acceptance.reasons.join('；');altitudeBox.append(item);}
  const joint=data.joint,jointTag=['J2','J2R'].includes(joint?.stage)?joint.stage:(joint?.new_training_actions||0)>0?'J1R':'J1';text('joint-title',`${jointTag} · 到达位置、转向并稳定保持`);text('joint-ready',joint?.COMPOSED_JOINT_TASK_VERIFIED?`${jointTag} · 组合任务通过`:joint?.status==='evaluated'?`${jointTag} · 未达标`:`${jointTag} · 尚未完成评估`);
  const jointBox=$('joint-summary');jointBox.replaceChildren();
  const jointNote=document.createElement('p');jointNote.textContent=joint?.note||'两个已训练技能在连续物理场景接力：导航到位，再转向并保持。正式评估完成后显示成绩。';jointBox.append(jointNote);
  for(const row of joint?.models||[]){const item=document.createElement('p');item.textContent=`种子 ${row.seed}：验证 ${row.validation} · 封存 ${row.sealed} · 碰撞/越界 ${row.collision_or_bounds} · 指令对照 ${row.instruction_pairs_successes}/4 · 零神经特征 ${row.zero_features_successes}/12`;jointBox.append(item);}
  if(['J2','J2R'].includes(joint?.stage)){const names={clean:'无扰动',pose:'定位延迟与噪声',force:'短时外力',combined:'叠加扰动'};for(const row of joint.models||[]){const item=document.createElement('p');item.textContent=`种子 ${row.seed} 同场景训练前 ${row.before_sealed} → 训练后 ${row.sealed}；`+Object.entries(row.profiles||{}).map(([key,value])=>`${names[key]||key} ${value.successes}/${value.episodes}`).join(' · ');jointBox.append(item);}}
  if(joint?.acceptance?.reasons?.length){const item=document.createElement('p');item.textContent='未通过项：'+joint.acceptance.reasons.map(r=>r.replace(/^(\d+): success below 90%$/, '种子 $1：成功率低于90%').replace(/^(\d+): instruction pairs failed$/, '种子 $1：指令对照未全部通过').replace(/^(\d+): (clean|pose|force|combined) success below (\d+)%$/,(_,seed,profile,rate)=>`种子 ${seed}：${({clean:'无扰动',pose:'定位延迟与噪声',force:'短时外力',combined:'叠加扰动'})[profile]}成功率低于${rate}%`)).join('；');jointBox.append(item);}
  const heading=data.heading;text('heading-ready',heading?.HEADING_TASK_LEARNED?'H1 · YES':heading?.status==='evaluated'?'H1 · 未达标':'H1 · 尚未完成评估');
  const headingBox=$('heading-summary');headingBox.replaceChildren();
  const headingNote=document.createElement('p');headingNote.textContent=heading?.note||'独立学习转向读出，共享冻结 MaleCNS 神经网络。导航权重保留；正式结果完成后显示。';headingBox.append(headingNote);
  for(const row of heading?.models||[]){const item=document.createElement('p');item.textContent=`种子 ${row.seed}：验证 ${row.validation} · 封存 ${row.sealed} · 顺时针 ${row.cw_actions} 次 / 逆时针 ${row.ccw_actions} 次 · 碰撞/越界 ${row.collision_or_bounds} · 导航保持 ${row.retention}（${row.navigation_exact?'逐场一致':'存在差异'}）`;headingBox.append(item);}
  if(heading?.acceptance?.reasons?.length){const failure=document.createElement('p');failure.textContent='未通过项：'+heading.acceptance.reasons.join('；');headingBox.append(failure);}
  const c1=data.c1;text('c1-ready',c1?.C1_TASK_LEARNED?'C1 · YES':c1?'C1 · '+(c1.status==='evaluated'?'未达标':c1.status||'尚未评估'):'C1 · 尚未评估');
  const c1Box=$('c1-summary');c1Box.replaceChildren();
  const c1Note=document.createElement('p');c1Note.textContent=c1?.note||'当前已有 C0 基准。C1 加入随机初始朝向与转向动作，需独立评估。';c1Box.append(c1Note);
  for(const row of c1?.models||[]){const item=document.createElement('p');item.textContent=`种子 ${row.seed}：C1 验证 ${row.validation} · 封存 ${row.sealed} · 碰撞/越界 ${row.collision_or_bounds} · 同场景 C0 保持 ${row.retention_before}/100 → ${row.retention_after}/100（${row.retention_passed?'通过':'未通过'}）`;c1Box.append(item);}
  if(c1?.comparison?.length){const paired=document.createElement('p');paired.textContent='同场景对照使用预先选定的验证场景；训练前为上一阶段 C0 权重，不是完全未训练模型。单个回放不能替代正式成功率。';c1Box.append(paired);}
  if(c1?.models?.length){const limit=document.createElement('p');limit.textContent='本课程验证随机初始朝向下到达并稳定保持。本次封存没有转向动作，尚未验证主动转向到指定角度。C2、完整 TS1 与真机尚未就绪。';c1Box.append(limit);}
  text('replay-help',altitude?.models?.length?`本轮 V1 独立高度任务${altitude.ALTITUDE_TASK_LEARNED?'通过':'尚未通过'}。先点“V1 高度 11 · 上升”，再看下降入口；固定使用封存第4场和第8场叠加扰动，样例不能替代总体成绩。`:joint?.models?.length?`${joint.COMPOSED_JOINT_TASK_VERIFIED?`本轮 ${jointTag} 组合任务通过。`:`本轮 ${jointTag} 整体门禁未通过，原因见下方评估。`}先点“${jointTag} 联合任务 11”查看导航、转向和稳定保持。${['J2','J2R'].includes(jointTag)?'三个入口固定使用封存第4场，含定位延迟、噪声与外力叠加；可点训练前对照。':'三个入口固定使用各自封存第1场；'}样例不能替代整体成功率。`:heading?.models?.length?'先点“H1 转向模型 11”查看本轮结果。紫色箭头是指定朝向，红色箭头是机头方向；右侧显示实际测量误差。三个按钮固定使用封存第1场，样例不替代整体成功率。':c1?.models?.length?'先点“C1 模型 22”看成功样例，再看11的同场景超时样例。三个入口都固定使用封存第1场；对照按钮另播同一验证场景的 C0 初始权重和 C1 训练后权重。':'C1 正式评估完成后会显示三个模型快捷回放；当前可查看已生成的同场景对照、上一阶段 C0 或训练最后快照。');
  const featured=new Map(),shortcuts=$('replay-shortcuts');shortcuts.replaceChildren();
  const available=new Set(data.runs.map(r=>r.run_id));
  const models=data.training?.c0_campaign?.models||[];
  const shortcut=(run,label,buttonLabel,primary=false)=>{
    if(!run||!available.has(run))return;
    featured.set(run,label);
    if(!run.startsWith('altitude-')&&data.joint?.models?.length&&!run.startsWith(jointTag==='J2R'?'stability-':jointTag==='J2'?'robust-':'joint-'))return;
    if(!run.startsWith('altitude-')&&!data.joint?.models?.length&&data.heading?.models?.length&&!run.startsWith('heading-'))return;
    const button=document.createElement('button');button.textContent=buttonLabel;
    if(primary)button.className='primary';
    button.onclick=attempt(async()=>{await loadRun(run);playing=true;text('play','暂停');});shortcuts.append(button);
  };
  for(const [i,row] of (altitude?.models||[]).entries()){shortcut(row.run_id,`V1 · 高度 ${row.seed} · 上升 · ${row.sealed}`,`V1 高度 ${row.seed} · 上升 · ${row.sample_outcome==='success'?'成功':'失败'}`,i===0);shortcut(row.down_run_id,`V1 · 高度 ${row.seed} · 下降 · ${row.sealed}`,`V1 高度 ${row.seed} · 下降 · ${row.down_sample_outcome==='success'?'成功':'失败'}`);}
  for(const [i,row] of (data.joint?.models||[]).entries())shortcut(row.run_id,`${jointTag} · 联合任务 ${row.seed} · ${row.sealed}`,`${jointTag} 联合任务 ${row.seed} · ${row.sample_outcome==='success'?'样例成功':'样例失败'}`,i===0&&!altitude?.models?.length);
  for(const row of (data.joint?.models||[]))if(row.before_run_id)shortcut(row.before_run_id,`${jointTag} 同场景训练前 · 种子 ${row.seed}`,`训练前对照 ${row.seed}`);
  for(const [i,row] of (data.heading?.models||[]).entries())shortcut(row.run_id,`H1 · 转向模型 ${row.seed} · ${row.sealed}`,`H1 转向模型 ${row.seed} · ${row.sample_outcome==='success'?'样例成功':'样例失败'}`,i===0);
  for(const row of data.heading?.comparison||[]){const label={untrained:'种子11训练前',trained:'种子11训练后'}[row.label]||row.label;shortcut(row.run_id,`H1 同场景对照 · ${label}`,`H1 对照 · ${label}`);}
  for(const [i,row] of (data.c1?.models||[]).entries())shortcut(row.run_id,`C1 · 模型 ${row.seed} · ${row.sealed??'开发验证'}`,`C1 模型 ${row.seed} · ${row.sample_outcome==='success'?'样例成功':row.sample_outcome==='task_deadline'?'样例超时':'点击播放'}`,i===0);
  for(const row of data.c1?.comparison||[])shortcut(row.run_id,`C1 同场景 · ${row.label}`,`C1 对照 · ${row.label}`);
  const rigidModels=(data.rigid?.models||[]).filter(row=>available.has(row.run_id));
  for(const [index,result] of rigidModels.entries())shortcut(result.run_id,`六自由度 · 模型 ${result.seed} · 封存验收 ${result.sealed}`,`新环境模型 ${result.seed} · 点击播放`,index===0&&!(data.c1?.models?.length));
  shortcut(data.rigid?.demo_run,'六自由度 · 物理脚本验证（非模型）','物理脚本验证 · 点击播放',rigidModels.length===0);
  for(const model of models){
    const result=model.sealed||model.validation;
    const run=result?.results?.find(row=>row.run_id)?.run_id;
    const label=`旧环境模型 ${model.seed} · ${model.sealed?'正式测试':'开发验证'}样例（${result?.successes}/${result?.episodes}）`;
    shortcut(run,label,`旧环境模型 ${model.seed} · 点击播放`,false);
  }
  const failed=models.find(model=>model.failure_replay?.results?.some(row=>row.run_id));
  if(failed)shortcut(failed.failure_replay.results.find(row=>row.run_id).run_id,`模型 ${failed.seed} · 开发验证失败样例`,'旧环境失败样例');
  $('replay-start').hidden=shortcuts.childElementCount===0;
  const featuredGroup=document.createElement('optgroup');featuredGroup.label='本轮模型：建议从这里开始';
  const historyGroup=document.createElement('optgroup');historyGroup.label='其他测试与历史回放';
  for(const [run,label] of featured){const option=document.createElement('option');option.value=run;option.textContent=label;featuredGroup.append(option);}
  data.runs.filter(r=>!featured.has(r.run_id)).forEach(r=>{const option=document.createElement('option');option.value=r.run_id;option.textContent=`${r.run_id} · ${r.observer_only?(r.active?'训练观察':'最后快照'):(r.complete?'回放':'记录中')}${r.partial?' · PARTIAL':''}`;historyGroup.append(option);});
  if(featuredGroup.childElementCount)$('runs').append(featuredGroup);
  if(historyGroup.childElementCount)$('runs').append(historyGroup);
  if(selected)$('runs').value=selected;
  const e=data.evaluation;const box=$('evaluation');box.replaceChildren();
  text('model-ready',e?.MODEL_READY_FOR_NEXT_STAGE?'MODEL READY · YES':'MODEL READY · NO');
  if(e){
    const title=document.createElement('div');title.textContent=`诊断集 ${e.successes}/${e.episodes} 成功 (${fmt(e.success_rate*100,1)}%) · 越界 ${e.out_of_bounds} · 碰撞 ${e.collisions}`;box.append(title);
    Object.entries(e.groups).forEach(([name,g])=>{const cell=document.createElement('div');cell.textContent=`${name}：${g.successes}/${g.episodes}`;box.append(cell);});
  }
  const comparison=$('comparison');comparison.replaceChildren();
  if(data.comparison){
    const explanation=document.createElement('p');explanation.textContent='同场景比较：新初始化基线 vs 已有训练模型。历史训练前权重未保存；此处使用真实运行的 seed-11 新初始化模型，不能据此计算历史训练提升。';comparison.append(explanation);
    data.comparison.runs.forEach(r=>{const button=document.createElement('button');button.textContent=`${r.label} · ${r.success?'成功':'失败'} · 距离 ${fmt(r.final_distance_m)} m · 回报 ${fmt(r.episode_return)}`;button.onclick=attempt(()=>loadRun(r.run_id));comparison.append(button);});
  }
  latestTrainingRun=data.training?.c0_campaign?.models?.[0]?.sealed?.results?.find(r=>r.run_id)?.run_id||data.training?.trained?.results?.find(r=>r.run_id)?.run_id||null;
  const learning=$('sdk9-learning');learning.replaceChildren();
  if(data.learning){
    const info=document.createElement('p');info.textContent=`历史：从零开始的纯 PPO 近距离单轴对照 · ${data.learning.seeds.length} 个随机种子 · 每次 ${data.learning.options_per_run} 个动作。固定表示选最高概率动作；概率表示按动作分布采样。封存测试未使用。`;learning.append(info);
    const table=document.createElement('table');
    const header=document.createElement('tr');
    for(const label of ['种子','输入来源','训练前（固定）','训练后（固定）','训练后（概率）']){const cell=document.createElement('th');cell.textContent=label;header.append(cell);}table.append(header);
    const labels={reservoir:'果蝇脑读出',raw_observation_control:'原始观测对照',zero_brain_control:'脑特征置零对照'};
    for(const row of data.learning.rows){
      const tr=document.createElement('tr');
      for(const value of [row.seed,labels[row.source]||row.source,`${row.initial_successes}/${row.evaluation_episodes}`,`${row.trained_successes}/${row.evaluation_episodes}`,row.sampled_episodes?`${row.sampled_successes}/${row.sampled_episodes}`:'待评估']){const cell=document.createElement('td');cell.textContent=String(value);tr.append(cell);}table.append(tr);
    }learning.append(table);
  }
  const training=$('sdk9-training');training.replaceChildren();
  if(data.training){
    const t=data.training;
    if(t.c0_campaign){
      const study=t.c0_campaign;const intro=document.createElement('p');
      intro.textContent='完整 C0：固定高度、前后左右控制，随机起终点距离 0.6–2.5 米。三个种子使用同一冻结连接组，各自训练动作与价值读出。完成验收后，回放按钮展示各模型的第一个封存场景。';training.append(intro);
      const table=document.createElement('table');const head=document.createElement('tr');
      for(const label of ['种子','开发验证','封存验收','碰撞 / 越界','状态','回放']){const th=document.createElement('th');th.textContent=label;head.append(th);}table.append(head);
      for(const r of study.models){const tr=document.createElement('tr');
        for(const value of [r.seed,r.validation?`${r.validation.successes}/${r.validation.episodes}`:'未运行',r.sealed?`${r.sealed.successes}/${r.sealed.episodes}`:'未运行',r.sealed?.collision_or_bounds??'未运行',({completed:'训练完成',INCOMPLETE_OR_RUNNING:'训练中 / 未完成',NOT_RUN:'未开始',budget_exhausted:'达到预算'})[r.status]||r.status]){const td=document.createElement('td');td.textContent=String(value);tr.append(td);}
        const replayCell=document.createElement('td');const replayButton=document.createElement('button');replayButton.textContent=`模型 ${r.seed} 回放`;
        const replayRun=(r.sealed?.results||r.validation?.results||[]).find(v=>v.run_id)?.run_id;replayButton.disabled=!replayRun;replayButton.onclick=attempt(()=>loadRun(replayRun));replayCell.append(replayButton);
        const failureRun=r.failure_replay?.results?.find(v=>v.run_id)?.run_id;if(failureRun){const failed=document.createElement('button');failed.textContent='失败样例';failed.onclick=attempt(()=>loadRun(failureRun));replayCell.append(failed);}
        tr.append(replayCell);table.append(tr);}
      training.append(table);const gate=document.createElement('p');gate.textContent=study.conclusion;training.append(gate);
      const baselineNote=document.createElement('p');baselineNote.textContent='直接 MLP 对照使用 1,536 个纯 PPO 动作；果蝇读出另有示范与纠错训练。训练方法和预算不同，这些成绩不能证明连接组优于普通网络。';training.append(baselineNote);
      text('model-ready',t.MODEL_READY_FOR_NEXT_STAGE?'MODEL READY · YES':'MODEL READY · NO');
    }
    if(t.boundary_comparison){
      const study=t.boundary_comparison;
      const intro=document.createElement('p');intro.textContent=`边界与恢复纠错：${study.seeds.length} 个种子，相同 40 个场景；32 个开发验证场景与本轮另外保留的 8 个场景。教师只用于训练，回放由模型独立决策。`;training.append(intro);
      const table=document.createElement('table');const head=document.createElement('tr');
      for(const label of ['种子','固定动作：前 → 后','概率动作：前 → 后','已知失败修复 / 无回退']){const th=document.createElement('th');th.textContent=label;head.append(th);}table.append(head);
      for(const result of study.results){
        const score=mode=>{const rows=result.comparison.filter(r=>r.mode===mode);const n=rows.reduce((a,r)=>a+r.episodes,0);return `${rows.reduce((a,r)=>a+r.before,0)}/${n} → ${rows.reduce((a,r)=>a+r.after,0)}/${n}`;};
        const row=document.createElement('tr');for(const value of [result.seed,score('argmax'),score('sampled'),`${result.KNOWN_FAILURE_REPAIRED?'是':'否'} / ${result.NO_CASE_REGRESSIONS?'是':'否'}`]){const td=document.createElement('td');td.textContent=String(value);row.append(td);}table.append(row);
      }training.append(table);
    }
    if(t.warmstart_comparison){
      const info=document.createElement('p');info.textContent='示范模型继续 PPO 训练：3 个种子 × 相同 16 个验证场景；前后使用相同动作随机数。原有 8 场景与新增 8 场景的详细成绩见报告。';training.append(info);
      const table=document.createElement('table');const head=document.createElement('tr');
      for(const label of ['种子','固定动作：前 → 后','概率动作：前 → 后','逐场景保留检查']){const cell=document.createElement('th');cell.textContent=label;head.append(cell);}table.append(head);
      for(const result of t.warmstart_comparison.results){
        const score=mode=>{const rows=result.results.filter(r=>r.mode===mode);return `${rows.reduce((n,r)=>n+r.before,0)}/16 → ${rows.reduce((n,r)=>n+r.after,0)}/16`;};
        const row=document.createElement('tr');
        for(const value of [result.seed,score('argmax'),score('sampled'),result.PILOT_RETENTION_CHECK_PASSED?'通过':'未通过']){const cell=document.createElement('td');cell.textContent=String(value);row.append(cell);}table.append(row);
      }training.append(table);
    }
    const info=document.createElement('p');info.textContent=`课程 ${t.curriculum||'C0'} · ${t.c0_campaign?'四方向示范 + DAgger + PPO':t.repair_method?'边界纠错示范（非 PPO）':t.ppo_updates===0?'规则示范启动（非 PPO）':t.initialization?.mode==='warm_start'?'示范模型继续 PPO 训练':'PPO'} · ${t.options} ${t.c0_campaign?'个训练动作':'条训练样本'} · ${t.updates} 次更新 · 读出参数变化 L2=${fmt(t.parameter_delta_l2,4)} · 保存后重新加载一致=${t.checkpoint_roundtrip_exact}。正式达标：${t.MODEL_READY_FOR_NEXT_STAGE?'YES':'NO'}。`;training.append(info);
    for(const evaluation of [...t.baselines,t.trained,...(t.fault_diagnostics?[t.fault_diagnostics]:[])]){
      const button=document.createElement('button');button.textContent=`${({'random-sealed':'封存集随机基线','raw-mlp-validation':'直接 MLP 对照','validation':'C0 训练后','rule-validation':'C0 规则基线','untrained-validation':'C0 训练前','rule':'规则控制','random':'随机控制','untrained':'训练前','trained':'训练后（PPO）','supervised-bootstrap':'示范启动后的模型','warm-start':'热启动模型','before':'纠错前','boundary-repair':'纠错后','fault-diagnostic':'故障诊断'})[evaluation.label]||evaluation.label}：${evaluation.successes}/${evaluation.episodes} 成功 · 回报 ${fmt(evaluation.mean_return)}`;
      const run=evaluation.results.find(r=>r.run_id)?.run_id;button.disabled=!run;button.onclick=attempt(()=>loadRun(run));training.append(button);
    }
    const note=document.createElement('p');note.textContent=t.c0_campaign?t.c0_campaign.scope:(t.repair_method?'独立训练场景上的纠错示范与 DAgger；验证时模型独立决策，价值部分在动作训练后重新拟合。本轮没有 PPO 更新。':t.ppo_updates===0?'规则示范启动：仅监督训练动作读出，价值估计未训练；不计为纯 PPO 成绩。':'该运行是单种子诊断；冻结连接组，仅训练动作／价值读出。')+(t.critic_warmup&&!t.repair_method?' 价值部分先单独训练；PPO 阶段固定共享隐藏层，只更新动作和价值输出层。':'')+' 封存测试集未使用。实机迁移还需要 SDK 传输层和外部定位。';training.append(note);
  }
  const rigid=data.rigid;const rigidBox=$('rigid-summary');rigidBox.replaceChildren();
  text('rigid-ready',rigid?.MODEL_READY_FOR_NEXT_STAGE?'新环境 C0 · YES':'新环境 C0 · NO / 尚未通过');
  if(rigid){
    const info=document.createElement('p');info.textContent=rigid.note;rigidBox.append(info);
    for(const row of rigid.models||[]){const item=document.createElement('p');item.textContent=`种子 ${row.seed}：开发验证 ${row.validation??'未运行'} · 封存验收 ${row.sealed??'未运行'} · ${row.method||'原有读出权重迁移评估'}`;rigidBox.append(item);}
    const scope=document.createElement('p');scope.textContent='工程模拟尚未经过真机校准；C1/C2 和真实飞行不由 C0 成绩推断。';rigidBox.append(scope);
  }else rigidBox.textContent='等待六自由度验证报告。';
  text('model-ready',data.training?.MODEL_READY_FOR_NEXT_STAGE?'旧环境 C0 · YES':'旧环境 C0 · NO');
  if(data.joint?.models?.[0]?.run_id)latestTrainingRun=data.joint.models[0].run_id;
  else if(data.heading?.models?.[0]?.run_id)latestTrainingRun=data.heading.models[0].run_id;
  else if(data.c1?.models?.[0]?.run_id)latestTrainingRun=data.c1.models[0].run_id;
  else if(rigidModels.length)latestTrainingRun=rigidModels[0].run_id;
  else if(rigid?.demo_run)latestTrainingRun=rigid.demo_run;
  return data.runs;
}
async function poll(){
  if(!live)return;
  const selectedRun=currentRun,selectedEpoch=epoch,ticket=generation;
  const frame=await request(`${knownRuns.find(r=>r.run_id===selectedRun)?.external_live?'runs':'sessions'}/${selectedRun}/latest`);
  if(ticket!==generation||frame.run_id!==selectedRun||frame.epoch!==selectedEpoch||frame.seq<=lastSeq)return;
  if(!pausedLive){
    if(lastSeq>=0)skipped+=Math.max(0,frame.seq-lastSeq-1);
    lastSeq=frame.seq;simTime=frame.time_s;renderFrame(frame,frame.trail);
  }
  if(frame.finished&&manifest.observer_only){pausedLive=true;text('play','观察已结束');$('play').disabled=true;text('outcome','训练已结束 · 最后快照');return;}
  if(frame.finished){
    notice(`模拟已结束并保存：${selectedRun}。现在可以拖动回放。`);
    await catalog();await loadRun(selectedRun,false);
  }
}
let polling=false;
setInterval(async()=>{if(polling||!live)return;polling=true;try{await poll();}catch(e){notice(`实时连接中断：${e.message}。重新连接后读取最新快照。`,true);}finally{polling=false;}},100);


function renderWatchGroups(runs){
  const rows=runs.filter(r=>r.observer_only);$('training-watch').hidden=rows.length===0;
  const groups=$('watch-group'),selected=groups.value;groups.replaceChildren();
  const keys=[...new Set(rows.map(r=>r.training_group+'|'+r.epoch))];
  for(const key of keys){const opt=document.createElement('option');opt.value=key;opt.textContent=key.replace('|',' · ');groups.append(opt);}
  const activeKey=rows.find(r=>r.active)||rows.find(r=>r.training_group?.endsWith('-ppo'))||rows[0];groups.value=keys.includes(selected)?selected:(activeKey?activeKey.training_group+'|'+activeKey.epoch:keys.at(-1)||'');renderWatchEnvs();
}
function renderWatchEnvs(){
  const key=$('watch-group').value,selected=$('watch-env').value,box=$('watch-env');box.replaceChildren();
  for(const row of knownRuns.filter(r=>r.observer_only&&r.training_group+'|'+r.epoch===key).sort((a,b)=>a.env_id-b.env_id)){
    const opt=document.createElement('option');opt.value=row.run_id;opt.textContent=`环境 ${row.env_id} · ${row.active?'训练中':'最后快照'}`;box.append(opt);
  }
  if([...box.options].some(o=>o.value===selected))box.value=selected;
}
$('watch-group').onchange=attempt(async()=>{renderWatchEnvs();if($('watch-env').value)await loadRun($('watch-env').value,true);});
$('watch-env').onchange=attempt(()=>loadRun($('watch-env').value,true));
$('watch-refresh').onclick=attempt(catalog);
$('watch-open').onclick=attempt(async()=>{if($('watch-env').value)await loadRun($('watch-env').value,true);});

$('runs').onchange=attempt(()=>loadRun($('runs').value));
$('refresh').onclick=attempt(catalog);
$('camera-reset').onclick=resetCamera;
$('camera-pan').onclick=()=>{const pan=$('camera-pan').getAttribute('aria-pressed')!=='true';controls.mouseButtons.LEFT=pan?THREE.MOUSE.PAN:THREE.MOUSE.ROTATE;$('camera-pan').setAttribute('aria-pressed',String(pan));text('camera-pan',pan?'切回旋转':'平移模式');};
$('demo').onclick=attempt(async()=>{const r=await request('sessions',{mode:'script',obstacles:$('obstacles').checked,seed:11});await catalog();await loadRun(r.run_id,true);});
$('sandbox').onclick=attempt(async()=>{const r=await request('sessions',{mode:'manual',obstacles:$('obstacles').checked,seed:11});await catalog();await loadRun(r.run_id,true);});
$('close-session').onclick=attempt(async()=>{await request(`sessions/${currentRun}/close`,{epoch});await catalog();await loadRun(currentRun);});
document.querySelectorAll('[data-command]').forEach(b=>{b.onclick=attempt(async()=>{
  const op=await request(`sessions/${currentRun}/command`,{epoch,wire:b.dataset.command,request_id:crypto.randomUUID(),lose_reply:$('lose-reply').checked});
  $('lose-reply').checked=false;notice(`已发送 ${op.wire} · ${op.operation_id}`);await poll();
});});
$('play').onclick=()=>{
  if(live){pausedLive=!pausedLive;text('play',pausedLive?'继续观察':'暂停观察');notice(pausedLive?'已暂停观察；后台物理模拟继续运行。':'已恢复到最新模拟快照。');}
  else {if(simTime>=manifest.duration_s)simTime=0;playing=!playing;if(!playing)generation++;text('play',playing?'暂停':'播放');}
};
$('timeline').oninput=attempt(async()=>{playing=false;text('play','播放');generation++;simTime=Number($('timeline').value);await seek(simTime);});
$('command-jump').onchange=attempt(async()=>{playing=false;text('play','播放');generation++;simTime=Number($('command-jump').value);await seek(simTime);});
function animate(now){
  const elapsed=lastAnimation?Math.min((now-lastAnimation)/1000,.1):0;lastAnimation=now;
  if(playing&&!live){
    simTime=Math.min(manifest.duration_s,simTime+elapsed*Number($('rate').value));
  }
  if(playing&&!live&&!replayBusy){
    replayBusy=true;
    seek(simTime).catch(e=>{playing=false;text('play','播放');notice(e.message,true);}).finally(()=>{replayBusy=false;});
    if(simTime>=manifest.duration_s){playing=false;text('play','播放');}
  }
  controls.update();renderer.render(scene,camera);requestAnimationFrame(animate);
}
requestAnimationFrame(animate);
try{token=(await request('bootstrap')).token;const runs=await catalog();if(runs.length)await loadRun(latestTrainingRun||runs.find(r=>r.run_id==='golden-episode')?.run_id||runs[0].run_id);else notice('请选择运行脚本演示或新建手动沙盒。');}
catch(error){notice(error.message,true);}
