#!/usr/bin/env python3
"""Recreate scientific figures from published counts and archived numeric data.

Uses Matplotlib's standard style, tab10 colours, labelled axes and vector
exports. Captions and interpretation belong in the accompanying documentation.
"""
import csv,json,os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[2]/'revision2/mpl_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,FancyArrowPatch,Patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1];ASSETS=ROOT/'docs/assets';DATA=ASSETS/'data';FIG=ASSETS/'figures'
plt.style.use('default')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,'axes.labelsize':10,'legend.fontsize':9,'xtick.labelsize':9,'ytick.labelsize':9,'axes.linewidth':.8,'lines.linewidth':1.4,'svg.fonttype':'none','pdf.fonttype':42,'savefig.facecolor':'white'})
BLUE,ORANGE,GREEN=plt.rcParams['axes.prop_cycle'].by_key()['color'][:3]


def save(fig,name):
    for ext in ['svg','png','pdf']:
        path=FIG/f'{name}.{ext}'
        fig.savefig(path,dpi=300,bbox_inches='tight',pad_inches=.06)
        if ext=='svg':path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    plt.close(fig)


def results():
    data=json.loads((DATA/'published_results.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(8,3.2),layout='constrained',gridspec_kw={'width_ratios':[1,1.45]})
    ax=axes[0];v=[113/140*100,0,0];bars=ax.bar(range(3),v,color=[BLUE,'0.65','0.65'],edgecolor='black',linewidth=.6,width=.62)
    ax.set(xticks=range(3),xticklabels=['Diffusion\nPolicy','IBC','LSTM-GMM'],ylabel='Success rate (%)',ylim=(0,112),yticks=[0,25,50,75,100],title='(a) Method comparison')
    for b,value,n in zip(bars,v,[113,0,0]):ax.annotate(f'{n}/140',xy=(b.get_x()+b.get_width()/2,value+2),ha='center',fontsize=9)
    ax=axes[1];rs=data['rows'][:14];bars=ax.bar(np.arange(1,15),[r['successes']*10 for r in rs],color=[BLUE]*10+[ORANGE]*4,edgecolor='black',linewidth=.5,width=.72)
    ax.set(xlabel='Carcass ID',ylabel='Success rate (%)',xticks=range(1,15),ylim=(0,112),yticks=[0,25,50,75,100],title='(b) Diffusion Policy by carcass')
    ax.axvline(10.5,color='0.4',lw=.7,ls='--');ax.legend(handles=[Patch(facecolor=BLUE,label='Seen (84/100)'),Patch(facecolor=ORANGE,label='Unseen (29/40)')],loc='lower left',fontsize=8,framealpha=1)
    save(fig,'published_results')


def training():
    inv=json.loads((DATA/'source_inventory.json').read_text());fig,axes=plt.subplots(1,3,figsize=(8,2.8),layout='constrained')
    for ax,run,title,ylabel in zip(axes,inv['training_runs'],['(a) Diffusion Policy','(b) IBC','(c) LSTM-GMM'],['Noise-prediction MSE','Classification loss','Negative log-likelihood']):
        rows=list(csv.DictReader((DATA/run['csv']).open()));x=np.array([float(r['epoch']) for r in rows])+1;y=np.array([float(r['train_loss']) for r in rows]);ax.plot(x,y,color=BLUE,lw=1)
        if run['method']=='dp':ax.set_yscale('log')
        ax.set(title=title,xlabel='Epoch',ylabel=ylabel);ax.ticklabel_format(axis='x',style='plain')
    save(fig,'training_curves')


def action_trace():
    d=json.loads((DATA/'recorded_episode.json').read_text());t=np.array(d['time']);a=np.array(d['action']);p=np.array(d['robot_eef_pose']);valid=np.any(a!=0,axis=1)
    fig,axes=plt.subplots(4,1,figsize=(7.2,6),sharex=True,layout='constrained')
    for j,ax in enumerate(axes[:3]):
        ax.plot(t[valid],a[valid,j]*1000,color=BLUE,label='Target');ax.plot(t,p[:,j]*1000,'--',color='0.25',lw=1,label='Measured')
        ax.set_ylabel('$'+['x','y','z'][j]+'$ (mm)');ax.set_xlim(0,t[-1])
    axes[0].legend(ncol=2,loc='lower right');axes[0].set_title('(a–c) End-effector translation',loc='left')
    ax=axes[3]
    for j,c,label,ls in [(6,BLUE,'Left','-'),(7,ORANGE,'Right','--')]:ax.step(t[valid],a[valid,j],where='post',color=c,label=label,ls=ls)
    ax.set(yticks=[0,1],yticklabels=['0 (closed)','1 (open)'],ylim=(-.15,1.15),xlabel='Episode time (s)',ylabel='Jaw command');ax.set_title('(d) Independent jaw commands',loc='left');ax.legend(loc='center left',ncol=2)
    save(fig,'recorded_actions')


def denoising():
    z=np.load(DATA/'denoising_trace.npz');meta=json.loads((DATA/'denoising_provenance.json').read_text());samples=[]
    for i,s in enumerate(meta['samples']):samples.append(dict(time=s['time'],normalized=z[f'normalized_{i}'].round(5).tolist(),physical=z[f'physical_{i}'].round(6).tolist()))
    (DATA/'denoising.json').write_text(json.dumps(dict(metadata={k:v for k,v in meta.items() if k!='policy_config'},samples=samples),separators=(',',':')))
    trace=z['normalized_2'];physical=z['physical_2'];fig=plt.figure(figsize=(9,4.5));gs=fig.add_gridspec(2,4,height_ratios=[1.45,1],hspace=.45,wspace=.4,left=.075,right=.94,bottom=.13,top=.91)
    for col,step in enumerate([0,10,40,100]):
        ax=fig.add_subplot(gs[0,col],projection='3d');v=trace[step,:,:3];ax.plot(*v.T,lw=.8,color='0.5');ax.scatter(*v.T,c=np.arange(16),cmap='plasma',s=12,depthshade=False)
        ax.set(xlim=(-3,3),ylim=(-3,3),zlim=(-3,3),xticks=[-2,2],yticks=[-2,2],zticks=[-2,2]);ax.set_title(f'Iteration {step}',pad=1);ax.set_xlabel('$x$',labelpad=-9);ax.set_ylabel('$y$',labelpad=-9);ax.set_zlabel('$z$',labelpad=-9);ax.tick_params(labelsize=7,pad=-2);ax.view_init(elev=25,azim=-55)
        ax=fig.add_subplot(gs[1,col]);v=physical[step];ax.plot(range(1,17),v[:,6],color=BLUE,label='Left');ax.plot(range(1,17),v[:,7],color=ORANGE,ls='--',label='Right');ax.axhline(.5,color='0.5',lw=.7,ls=':');ax.set(ylim=(-1,2),xticks=[1,8,16],yticks=[0,1],xlabel='Future action index')
        if col==0:ax.set_ylabel('Raw jaw value');ax.legend(loc='lower left',fontsize=8)
    save(fig,'diffusion_denoising')
    fig,axes=plt.subplots(2,2,figsize=(7.2,4.5),layout='constrained');v=physical[-1]
    for j,ax in enumerate(axes.flat):
        if j<3:
            ax.plot(range(1,17),v[:,j]*1000,'o-',color=BLUE,ms=3);ax.set_ylabel('$'+['x','y','z'][j]+'$ (mm)');ax.ticklabel_format(axis='y',useOffset=False)
        else:
            ax.step(range(1,17),np.clip(np.rint(v[:,6]),0,1),where='post',label='Left');ax.step(range(1,17),np.clip(np.rint(v[:,7]),0,1),where='post',ls='--',label='Right');ax.set_ylabel('Decoded jaw command');ax.set_yticks([0,1]);ax.set_ylim(-.15,1.15);ax.legend()
        ax.set_xlabel('Future action index');ax.set_xticks([1,4,8,12,16]);ax.set_title(['(a) Cartesian x','(b) Cartesian y','(c) Cartesian z','(d) Jaw commands'][j],loc='left')
    save(fig,'final_action_sequence')


def pipeline():
    fig,ax=plt.subplots(figsize=(9,2.65));ax.set(xlim=(0,10),ylim=(0,3));ax.axis('off')
    blocks=[(.05,1.45,2.15,'Observation history','3 RGB views\nPose + jaw states'),(2.7,1.45,1.8,'Visual encoder','Image features'),(5,1.45,2.1,'Diffusion model','16 future actions\n8 values per action'),(7.65,1.45,2.25,'Robot + gripper','Execute action prefix\nUpdate observations')]
    for x,y,w,title,body in blocks:
        ax.add_patch(Rectangle((x,y),w,1.05,facecolor='white',edgecolor='black',lw=.8));ax.text(x+w/2,y+.77,title,ha='center',fontsize=10);ax.text(x+w/2,y+.35,body,ha='center',va='center',fontsize=9)
    for x1,x2 in [(2.2,2.7),(4.5,5),(7.1,7.65)]:ax.add_patch(FancyArrowPatch((x1,1.97),(x2,1.97),arrowstyle='-|>',mutation_scale=10,lw=.8,color='black'))
    ax.annotate('',xy=(1.1,1.43),xytext=(8.7,1.43),arrowprops=dict(arrowstyle='->',connectionstyle='bar,fraction=-.11',lw=.8));ax.text(4.8,.69,'Receding-horizon feedback',ha='center',fontsize=9)
    ax.text(5,.12,'After grasp and lift: scripted waypoint transfer to the shackle',ha='center',fontsize=9)
    save(fig,'policy_pipeline')

if __name__=='__main__':
    FIG.mkdir(parents=True,exist_ok=True);results();training();action_trace();denoising();pipeline()
    (DATA/'bundle.js').write_text('window.CHICGRASP_DATA='+json.dumps(dict(denoising=json.loads((DATA/'denoising.json').read_text()),recorded=json.loads((DATA/'recorded_episode.json').read_text())),separators=(',',':'))+';\n')
    print('Exported standard Matplotlib figures as PNG, PDF and SVG')
