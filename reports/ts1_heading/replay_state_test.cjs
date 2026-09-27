const fs=require('node:fs'),assert=require('node:assert/strict');
const source=fs.readFileSync(process.argv[2],'utf8');
const seek=source.slice(source.indexOf('async function seek('),source.indexOf('async function loadRun('));
const frame=(tick,error,result=false)=>({sim_tick:tick,time_s:tick/120,result_only:result,heading:{error_rad:error},observation:[error],stable_hold_s:tick,policy:{selected_action:tick%9},reward:{reward_total:tick}});
const rows=[frame(9,.8,true),frame(10,.5),frame(11,.3)];
const render=new Function('rows',`let live=false,manifest={streams:{}},generation=1,cache=new Map(),captured; const text=()=>{};const renderFrame=x=>captured=x; const around=async kind=>kind==='transition'?rows:kind==='trajectory'?[{x_m:0,y_m:0,z_m:1,yaw_rad:.9,sim_tick:11,time_s:11/120}]:[]; ${seek};return seek(11/120).then(()=>captured);`);
(async()=>{const actual=await render(rows);assert.equal(actual.heading.error_rad,.3);assert.deepEqual(actual.observation,[.3]);assert.equal(actual.stable_hold_s,11);assert.equal(actual.reward.reward_total,9);console.log('REPLAY_LATEST_MEASURED_STATE_PASS');})();
