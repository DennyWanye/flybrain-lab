"""Offline scientific chart, derived only from the completed paired evaluation."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
out=Path('reports/ts1_robust');s=json.loads((out/'summary.json').read_text())
font=Path('/mnt/c/Windows/Fonts/msyh.ttc')
if font.exists():
    font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
plt.rcParams['axes.unicode_minus']=False
fig,axes=plt.subplots(2,2,figsize=(12,7.5),sharey=True)
names={'clean':'无扰动','pose':'定位噪声 + 100ms延迟','force':'短时外力','combined':'定位扰动 + 外力'}
for ax,(profile,title) in zip(axes.flat,names.items()):
    x=np.arange(3);old=[r['before_profiles'][profile]['successes'] for r in s['models']];new=[r['profiles'][profile]['successes'] for r in s['models']]
    bars1=ax.bar(x-.18,np.array(old)/75*100,.36,label='J1R 原模型',color='#8c9daf')
    bars2=ax.bar(x+.18,np.array(new)/75*100,.36,label='J2 扰动训练后',color='#218c83')
    for bars,counts in ((bars1,old),(bars2,new)):
        for bar,count in zip(bars,counts):ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1,f'{count}/75',ha='center',fontsize=9)
    gate=90 if profile=='clean' else 85;ax.axhline(gate,color='#b45b32',linestyle='--',linewidth=1,label=f'门槛 {gate}%')
    ax.set_title(f'{title}（门槛{gate}%）');ax.set_xticks(x,[f"种子 {r['seed']}" for r in s['models']]);ax.set_ylim(0,110);ax.set_yticks([0,25,50,75,100]);ax.set_ylabel('成功率 %');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
axes[0,0].legend(loc='lower left',fontsize=9)
fig.suptitle('J2 与 J1R：同场景封存测试',fontsize=17)
fig.text(.5,.02,'相同物理场景与扰动；每种子每类75场。仅模拟固定高度组合任务，非真机结果。',ha='center',fontsize=10)
fig.tight_layout(rect=[0,.045,1,.95]);fig.savefig(out/'paired-evaluation.png',dpi=150);plt.close(fig)
print('paired-evaluation.png generated from summary')
