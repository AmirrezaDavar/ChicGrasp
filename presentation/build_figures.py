#!/usr/bin/env python3
"""Export publication figures (SVG/PNG/PDF), data for the browser, and source stills."""
import argparse
import csv
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/chicgrasp_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[1]
ASSETS=ROOT/'docs/assets'
DATA=ASSETS/'data'
FIG=ASSETS/'figures'
BG='#f6f4ef'; INK='#142b36'; TEAL='#067c78'; GOLD='#cb792f'; MUTED='#566874'; GRID='#dde3df'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'figure.facecolor':BG,
    'axes.facecolor':BG,'axes.edgecolor':GRID,'axes.labelcolor':INK,'text.color':INK,
    'xtick.color':MUTED,'ytick.color':MUTED,'axes.spines.top':False,'axes.spines.right':False,
    'svg.fonttype':'none','pdf.fonttype':42,'axes.titleweight':'bold'})

def save(fig,name):
    for ext in ['svg','png','pdf']:
        fig.savefig(FIG/f'{name}.{ext}',dpi=180,facecolor=BG)
    plt.close(fig)

def results():
    data=json.loads((DATA/'published_results.json').read_text())
    fig=plt.figure(figsize=(14,7.6))
    fig.text(.06,.925,'Real hardware. Measured outcomes.',fontsize=27,weight='bold')
    fig.text(.06,.875,'Published pick-and-rehang evaluation  /  140 trials per method',color=MUTED,fontsize=13)
    ax=fig.add_axes([.10,.29,.37,.46]);vals=[80.71,0,0]
    ax.barh([2,1,0],vals,color=[TEAL,'#a6b2b7','#a6b2b7'],height=.48)
    ax.set_yticks([2,1,0],['Diffusion\nPolicy','IBC','LSTM-GMM']);ax.set_xlim(0,104);ax.set_xlabel('Success rate (%)')
    ax.set_xticks([0,25,50,75,100]);ax.spines[['left','bottom']].set_visible(False);ax.tick_params(length=0)
    for y,v,n in zip([2,1,0],vals,[113,0,0]):
        ax.text(v+2,y,f'{v:g}%\n{n}/140',va='center',weight='bold',color=TEAL if n else INK)
    ax2=fig.add_axes([.58,.29,.36,.46]);rs=data['rows'][:14]
    ax2.bar(range(14),[r['successes']*10 for r in rs],color=[TEAL]*10+[GOLD]*4,width=.65)
    ax2.set_ylim(0,115);ax2.set_yticks([0,50,100]);ax2.set_xticks(range(14),range(1,15));ax2.set_xlabel('Chicken exemplar');ax2.set_ylabel('Success rate (%)')
    ax2.axvline(9.5,color=MUTED,lw=.7,ls='--');ax2.text(4.5,109,'Seen  ·  84/100',ha='center',color=TEAL,weight='bold');ax2.text(11.7,109,'Unseen  ·  29/40',ha='center',color=GOLD,weight='bold')
    ax2.spines[['left','bottom']].set_visible(False);ax2.tick_params(length=0)
    fig.text(.06,.15,'Success requires a two-leg grasp, lift, and completion of the scripted rehang.',fontsize=12,weight='bold')
    fig.text(.06,.10,'Unseen: 12 nominal + 28 disturbed trials on four held-out birds. Each bird is evaluated 10 times.',fontsize=11,color=MUTED)
    fig.text(.06,.045,'Source: Davar et al., Advanced Robotics Research (2026), Table 4  ·  DOI: 10.1002/adrr.202500149',fontsize=10,color=MUTED)
    save(fig,'published_results')

def training():
    inv=json.loads((DATA/'source_inventory.json').read_text())
    fig,axes=plt.subplots(1,3,figsize=(14,5.6));fig.subplots_adjust(left=.065,right=.975,top=.72,bottom=.27,wspace=.3)
    fig.text(.065,.9,'Learning curves from the archived runs',fontsize=25,weight='bold')
    fig.text(.065,.835,'Each objective has its own scale. Lower training loss does not establish task success.',fontsize=12,color=MUTED)
    for ax,run,color,title in zip(axes,inv['training_runs'],[TEAL,GOLD,'#526e9f'],['Diffusion Policy','IBC','LSTM-GMM']):
        rows=list(csv.DictReader((DATA/run['csv']).open()));x=np.array([float(r['epoch']) for r in rows])+1;y=np.array([float(r['train_loss']) for r in rows]);ax.plot(x,y,color=color,lw=1.5)
        if run['method']=='dp':ax.set_yscale('log');ax.set_ylabel('Noise-prediction MSE (log scale)')
        elif run['method']=='ibc':ax.set_ylabel('Energy-based classification loss')
        else:ax.set_ylabel('Negative log-likelihood')
        ax.set_title(title,loc='left',fontsize=15);ax.set_xlabel('Epoch');ax.grid(axis='y',color=GRID,lw=.6);ax.set_axisbelow(True)
    fig.text(.065,.13,'Last logged loss per epoch; repeated global steps resolved by keeping the last record. No smoothing.',fontsize=10,color=MUTED)
    fig.text(.065,.07,'DP / IBC: 450 epochs. Archived LSTM-GMM log: 470 epochs, with resume history; the paper reports 450.',fontsize=10,color=MUTED)
    save(fig,'training_curves')

def action_trace():
    d=json.loads((DATA/'recorded_episode.json').read_text());t=np.array(d['time']);a=np.array(d['action']);p=np.array(d['robot_eef_pose']);valid=np.any(a!=0,axis=1)
    fig,axes=plt.subplots(3,1,figsize=(13,7.8),sharex=True,gridspec_kw={'height_ratios':[2,1,1]});fig.subplots_adjust(left=.09,right=.95,top=.78,bottom=.17,hspace=.30)
    fig.text(.09,.925,'One grasp. Two independent jaw commands.',fontsize=25,weight='bold');fig.text(.09,.866,'Recorded evaluation episode 14  /  robot-base Cartesian targets and jaw state',fontsize=12,color=MUTED)
    for j,c,label in zip(range(3),[TEAL,GOLD,'#526e9f'],['x','y','z']):
        axes[0].plot(t[valid],a[valid,j]*1000,color=c,lw=1.6,label=f'{label} target')
    axes[0].set_ylabel('Position (mm)');axes[0].legend(ncol=3,frameon=False,loc='center right')
    for ax,j,c,label in zip(axes[1:],[6,7],[TEAL,GOLD],['Left jaw','Right jaw']):
        ax.step(t[valid],a[valid,j],where='post',color=c,lw=2,label=label+' command')
        ax.set_yticks([0,1],['Closed','Open']);ax.set_ylim(-.15,1.3);ax.set_ylabel(label);ax.axvline(21.1 if j==6 else 20.6,color=c,lw=.7,ls='--');ax.text(21.5, .5,'21.1 s' if j==6 else '20.6 s',color=c)
    for ax in axes:ax.grid(axis='y',color=GRID,lw=.6)
    axes[-1].set_xlabel('Elapsed episode time (s)')
    fig.text(.09,.075,'Source: replay_buffer_dp.zarr.zip. Jaw bits: 0 = closed, 1 = open. Initial all-zero logger rows excluded.',fontsize=10,color=MUTED)
    save(fig,'recorded_actions')

def denoising():
    z=np.load(DATA/'denoising_trace.npz');meta=json.loads((DATA/'denoising_provenance.json').read_text())
    samples=[]
    for i,s in enumerate(meta['samples']):
        samples.append(dict(time=s['time'],normalized=z[f'normalized_{i}'].round(5).tolist(),physical=z[f'physical_{i}'].round(6).tolist()))
    (DATA/'denoising.json').write_text(json.dumps(dict(metadata={k:v for k,v in meta.items() if k!='policy_config'},samples=samples),separators=(',',':')))
    fig=plt.figure(figsize=(14,8));fig.text(.06,.93,'From noise to a coordinated action sequence',fontsize=25,weight='bold');fig.text(.06,.88,'Actual checkpoint replay  /  episode 14 at 20.0 s  /  100 DDIM iterations',fontsize=12,color=MUTED)
    trace=z['normalized_2'];physical=z['physical_2'];steps=[0,10,40,100]
    for col,step in enumerate(steps):
        ax=fig.add_axes([.075+.235*col,.44,.19,.32],projection='3d');v=trace[step,:,:3];ax.plot(v[:,0],v[:,1],v[:,2],color=TEAL,lw=2,marker='o',ms=3);ax.set_title('Initial noise' if not step else f'Step {step} / 100',fontsize=13);ax.set_xlim(-3,3);ax.set_ylim(-3,3);ax.set_zlim(-3,3);ax.set_xticks([-2,0,2]);ax.set_yticks([-2,0,2]);ax.set_zticks([-2,0,2]);ax.tick_params(labelsize=7);ax.set_xlabel('x',labelpad=-8);ax.set_ylabel('y',labelpad=-8);ax.set_zlabel('z',labelpad=-8);ax.view_init(elev=22,azim=-60);ax.set_facecolor(BG)
        if step==100:
            lo=v.min(0);hi=v.max(0);center=(lo+hi)/2;radius=max(hi-lo)*.65
            ax.set_xlim(center[0]-radius,center[0]+radius);ax.set_ylim(center[1]-radius,center[1]+radius);ax.set_zlim(center[2]-radius,center[2]+radius)
            ax.set_xticks([]);ax.set_yticks([]);ax.set_zticks([]);ax.set_title('Step 100 / 100\nDetail view (zoomed)',fontsize=12)
        ax2=fig.add_axes([.08+.235*col,.22,.185,.16]);v=physical[step];ax2.plot(range(1,17),v[:,6],color=TEAL,lw=1.7,label='Left');ax2.plot(range(1,17),v[:,7],color=GOLD,lw=1.7,label='Right');ax2.axhline(.5,color=MUTED,lw=.6,ls='--');ax2.set_ylim(-1,2);ax2.set_xticks([1,6,16]);ax2.set_yticks([0,.5,1]);ax2.tick_params(labelsize=8);ax2.set_xlabel('Planned action index',fontsize=9)
        if col==0:ax2.set_ylabel('Jaw value',fontsize=10);ax2.legend(frameon=False,fontsize=8,ncol=2,loc='upper left')
    fig.text(.06,.12,'Top: normalized XYZ targets; final panel zoomed. Bottom: raw jaw values; final values are rounded and clipped for commands.',fontsize=10,color=MUTED)
    fig.text(.06,.073,'This is a new offline sample, not the original executed plan. Recorded images and states are aligned approximately by elapsed time.',fontsize=10,color=MUTED)
    fig.text(.06,.031,'EMA checkpoint latest.ckpt  ·  seed 42  ·  16-step prediction horizon  ·  2 observation steps  ·  source hash in denoising_provenance.json',fontsize=9,color=MUTED)
    save(fig,'diffusion_denoising')

def pipeline():
    svg='''<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="750" viewBox="0 0 1600 750" role="img" aria-label="ChicGrasp perception to action pipeline"><defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" fill="#067c78"/></marker></defs><rect width="1600" height="750" fill="#f6f4ef"/><g font-family="DejaVu Sans, sans-serif" fill="#142b36"><text x="70" y="90" font-size="40" font-weight="700">See. Denoise. Act. Observe again.</text><text x="70" y="135" font-size="21" fill="#566874">ChicGrasp joins Cartesian motion and independent jaw control in one policy.</text>'''
    boxes=[(70,225,330,230,'01  OBSERVE',['Three RGB camera views','End-effector pose','Left + right jaw states']),(450,225,330,230,'02  GENERATE',['Encode observation history','Iteratively remove noise','Predict an action sequence']),(830,225,330,230,'03  COMMAND',['Robot: Cartesian pose targets','Gripper: left / right jaw bits','Execute a short prefix'])]
    for x,y,w,h,title,lines in boxes:
        svg+=f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="18" fill="#fff" stroke="#dde3df"/><text x="{x+25}" y="{y+45}" font-size="21" fill="#067c78" font-weight="700">{title}</text>'
        for i,line in enumerate(lines):svg+=f'<text x="{x+25}" y="{y+102+i*39}" font-size="19">{line}</text>'
    for x in [410,790]:svg+=f'<path d="M{x},340 h30" stroke="#067c78" stroke-width="3" marker-end="url(#arrow)"/>'
    svg+='''<path d="M995,470 V525 H235 V470" stroke="#067c78" stroke-width="3" fill="none" marker-end="url(#arrow)"/><text x="447" y="565" font-size="20" fill="#067c78">Receding horizon: update the plan with new observations</text><path d="M1170,340 h45" stroke="#cb792f" stroke-width="3"/><rect x="1230" y="225" width="300" height="230" rx="18" fill="#efe5d7"/><text x="1255" y="270" font-size="21" font-weight="700">SCRIPTED REHANG</text><text x="1255" y="327" font-size="19">A fixed waypoint sequence</text><text x="1255" y="366" font-size="19">completes the transfer.</text><text x="1255" y="415" font-size="17" fill="#566874">Separate from policy inference</text><text x="70" y="649" font-size="19">Task commands: [x, y, z, left jaw, right jaw]. Stored tensor: [x, y, z, rx, ry, rz, left jaw, right jaw].</text><text x="70" y="691" font-size="17" fill="#566874">Source: archived checkpoint configuration and ChicGrasp implementation; published study: DOI 10.1002/adrr.202500149.</text></g></svg>'''
    (FIG/'policy_pipeline.svg').write_text(svg)

if __name__=='__main__':
    FIG.mkdir(parents=True,exist_ok=True)
    results();training();action_trace();denoising();pipeline()
    (DATA/'bundle.js').write_text('window.CHICGRASP_DATA='+json.dumps(dict(
        denoising=json.loads((DATA/'denoising.json').read_text()),
        recorded=json.loads((DATA/'recorded_episode.json').read_text())),separators=(',',':'))+';\n')
    print('Figures and browser data generated')
