from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
root=Path.cwd();out=root/'reports/ts1_altitude';s=json.loads((out/'summary.json').read_text());font=FontProperties(fname='/mnt/c/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'figure.facecolor':'#f5f7fb','axes.facecolor':'white','axes.spines.top':False,'axes.spines.right':False,'font.size':11})
fig,axes=plt.subplots(1,2,figsize=(13,5.5));colors=['#127c81','#566bc5','#bd6b22'];profiles=['clean','pose','force','combined']
for i,row in enumerate(s['models']):
    axes[0].bar([j+(i-1)*.24 for j in range(4)],[100*row['profiles'][p]['success_rate'] for p in profiles],width=.22,color=colors[i],label=str(row['seed']))
    data=json.loads((out/f"s{row['seed']}-sealed_test.json").read_text())['results']
    axes[1].scatter([r['target_height_m'] for r in data],[r['height_error_m'] for r in data],s=12,alpha=.5,color=colors[i],label=str(row['seed']))
axes[0].set_xticks(range(4),['无扰动','定位噪声/延迟','三轴外力','叠加扰动'],fontproperties=font);axes[0].set_ylim(0,105);axes[0].set_ylabel('成功率（%）',fontproperties=font);axes[0].legend(title='Seed');axes[0].grid(axis='y',alpha=.18)
for i,t in enumerate([90,85,85,85]):axes[0].plot([i-.38,i+.38],[t,t],color='#b43845',linestyle='--',linewidth=1.5)
axes[0].set_title('封存评估 · 每种子每场景75场',fontproperties=font)
axes[1].axhline(.1,color='#b43845',linestyle='--');axes[1].set_xlabel('目标高度（米）',fontproperties=font);axes[1].set_ylabel('结束时真实高度误差（米）',fontproperties=font);axes[1].set_title('误差只是验收条件之一，仍需低速稳定保持',fontproperties=font);axes[1].grid(alpha=.18)
fig.suptitle('V1 独立高度控制 · '+('门禁通过' if s['MODEL_READY_FOR_NEXT_STAGE'] else '门禁未通过'),fontproperties=font,fontsize=18)
fig.text(.02,.02,'冻结MaleCNS，仅训练独立读出；尚未验证三维位置与朝向联合任务。',fontproperties=font,color='#445166')
fig.tight_layout(rect=[0,.06,1,.92]);fig.savefig(out/'results.png',dpi=160);print('results.png generated')
