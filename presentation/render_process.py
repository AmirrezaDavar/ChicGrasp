#!/usr/bin/env python3
"""Render reference-style denoising overlays from actual checkpoint arrays.

The source recording pauses for each offline replay, then resumes at 4x speed.
No sampled action is presented as an executed command. The affine camera overlay
is an estimated local displacement display, with its validation saved alongside.
"""
import argparse,json,os,shutil,subprocess
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[2]/'revision2/mpl_cache'))
import cv2,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-root',type=Path,required=True);ap.add_argument('--process-data',type=Path,required=True);ap.add_argument('--review',type=Path,required=True);args=ap.parse_args()
    dst=ROOT/'docs/assets/data/process';dst.mkdir(parents=True,exist_ok=True);args.review.mkdir(parents=True,exist_ok=True)
    for name in ['denoising_trace.npz','denoising_provenance.json','projection.json','projection_tracking.npz','projection_validation.png']:
        if (args.process_data/name).resolve()!=(dst/name).resolve():shutil.copy2(args.process_data/name,dst/name)
    z=np.load(dst/'denoising_trace.npz');meta=json.loads((dst/'denoising_provenance.json').read_text());projection=json.loads((dst/'projection.json').read_text());matrix=np.array(projection['matrix_xyz1_to_uv'])
    for i in range(3):
        assert z[f'physical_{i}'].shape==(17,6,16,8)
        np.testing.assert_allclose(z[f'physical_{i}'][-1],z[f'action_pred_{i}'],atol=1e-6)
        np.testing.assert_allclose(z[f'action_{i}'],z[f'action_pred_{i}'][:,1:16],atol=1e-6)
    plt.rcdefaults();plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.linewidth':.7,'svg.fonttype':'none','pdf.fonttype':42})
    fig=plt.figure(figsize=(12.8,7.2),dpi=100,facecolor='white');ax=fig.add_axes([0,0,1,1]);ax.set_axis_off();ax.set(xlim=(0,1280),ylim=(720,0))
    img=ax.imshow(np.zeros((720,1280,3),dtype=np.uint8),extent=(0,1280,720,0))
    colour=np.tile(np.arange(1,17),6);scatter=ax.scatter([],[],s=18,c=[],cmap='plasma',norm=Normalize(1,16),edgecolors='white',linewidths=.2,zorder=3)
    title=ax.text(18,29,'ChicGrasp',fontsize=13,color='black',bbox=dict(facecolor='white',alpha=.88,edgecolor='none',pad=4),zorder=5)
    status=ax.text(1262,29,'',ha='right',fontsize=11,color='black',bbox=dict(facecolor='white',alpha=.88,edgecolor='none',pad=4),zorder=5)
    note=ax.text(18,702,'Offline samples; estimated image projection',fontsize=10,color='black',bbox=dict(facecolor='white',alpha=.9,edgecolor='none',pad=4),zorder=5)
    legendpanel=fig.add_axes([.04,.07,.21,.105],facecolor='white');legendpanel.set_axis_off();legendpanel.patch.set_visible(True)
    cax=fig.add_axes([.055,.127,.17,.016]);cb=fig.colorbar(ScalarMappable(norm=Normalize(1,16),cmap='plasma'),cax=cax,orientation='horizontal',ticks=[1,8,16]);cb.ax.tick_params(labelsize=8);cb.set_label('Future action index',fontsize=9,labelpad=0)
    # Opaque white background beneath the small scientific jaw inset.
    panel=fig.add_axes([.69,.075,.29,.29],facecolor='white');panel.set_xticks([]);panel.set_yticks([])
    for sp in panel.spines.values():sp.set_visible(False)
    jaw=fig.add_axes([.73,.135,.23,.18],facecolor='white');left,=jaw.plot(range(1,17),np.ones(16),label='Left',color='tab:blue',lw=1.5);right,=jaw.plot(range(1,17),np.ones(16),label='Right',color='tab:orange',ls='--',lw=1.5)
    jaw.set(xlim=(1,16),ylim=(-1,2.5),xticks=[1,8,16],yticks=[0,1,2],xlabel='Future action index',ylabel='Raw jaw value');jaw.tick_params(labelsize=8);jaw.xaxis.label.set_size(9);jaw.yaxis.label.set_size(9);jaw.legend(loc='upper right',fontsize=8,ncol=2,frameon=False);jaw_title=jaw.set_title('Candidate 1',fontsize=9)
    cap=cv2.VideoCapture(str(args.data_root/'videos/dp_eval/14/2.mp4'));fps_source=cap.get(cv2.CAP_PROP_FPS);last=-1;rgb=None
    def frame_at(seconds):
        nonlocal last,rgb
        index=round(seconds*fps_source)
        if index!=last:
            if index!=last+1:cap.set(cv2.CAP_PROP_POS_FRAMES,index)
            ok,bgr=cap.read();assert ok,(seconds,index);rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);last=index
        return rgb
    # Every one of the 16 captured scheduler iterations is displayed for two
    # frames. Noise and final states receive additional holds; no interpolated
    # or fabricated trajectory is inserted between scheduler states.
    timeline=[];start=0
    for sample,s in enumerate(meta['samples']):
        t=s['time'];n=round((t-start)/4*30)
        for j in range(n):timeline.append(dict(source_time=start+(t-start)*j/n,sample=None,iteration=None,phase='recorded_motion'))
        for step in range(17):
            hold=6 if step==0 else 12 if step==16 else 2
            for _ in range(hold):timeline.append(dict(source_time=t,sample=sample,iteration=step,phase='offline_denoising'))
        start=t
    end=24.5;n=round((end-start)/4*30)
    for j in range(n):timeline.append(dict(source_time=start+(end-start)*j/n,sample=None,iteration=None,phase='recorded_motion'))
    media=ROOT/'docs/assets/media';media.mkdir(exist_ok=True);encoders={}
    for variant in ['','_clean']:
        path=media/f'chicgrasp_action_process{variant}.mp4'
        command=['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','1280x720','-r','30','-i','-','-an','-c:v','libx264','-crf','18','-preset','medium','-pix_fmt','yuv420p','-movflags','+faststart',str(path)]
        encoders[variant]=subprocess.Popen(command,stdin=subprocess.PIPE)
    reviews=[];seen=set()
    for i,item in enumerate(timeline):
        img.set_data(frame_at(item['source_time']));active=item['sample'] is not None
        if active:
            sample,step=item['sample'],item['iteration'];v=z[f'physical_{sample}'][step];pts=v[:,:,:3].reshape(-1,3);uv=np.c_[pts,np.ones(len(pts))]@matrix;scatter.set_offsets(uv);scatter.set_array(colour);scatter.set_visible(True)
            status.set_text(f'Denoising {step:02d}/16  |  observation {item["source_time"]:.1f} s');left.set_ydata(v[0,:,6]);right.set_ydata(v[0,:,7]);jaw_title.set_text('Candidate 1: predicted jaw values');note.set_text('Offline samples; estimated image projection')
        else:
            scatter.set_visible(False);status.set_text(f'Recorded motion  |  4×  |  {item["source_time"]:.1f} s');note.set_text('Recorded hardware evaluation · episode 14')
        cax.set_visible(active)
        for variant,enc in encoders.items():
            visible=active and not variant;panel.set_visible(visible);jaw.set_visible(visible);cax.set_visible(visible);legendpanel.set_visible(visible);title.set_visible(not variant);status.set_visible(not variant)
            fig.canvas.draw();out=np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy();enc.stdin.write(out.tobytes())
            if not variant and active and step in [0,4,8,16] and (sample,step) not in seen:
                seen.add((sample,step));im=Image.fromarray(out);im.save(args.review/f'process_{sample}_{step}.jpg',quality=95);reviews.append((sample,step,im))
                if sample==0 and step==8:im.save(ROOT/'docs/assets/figures/action_process_poster.jpg',quality=95)
        if i%90==0:print(f'Rendered {i}/{len(timeline)} frames',flush=True)
    for enc in encoders.values():enc.stdin.close();assert enc.wait()==0
    cap.release();plt.close(fig)
    sheet=Image.new('RGB',(1600,960),'white');draw=ImageDraw.Draw(sheet)
    for index,(sample,step,im) in enumerate(reviews):
        im.thumbnail((400,225));x=(index%4)*400;y=(index//4)*320;sheet.paste(im,(x,y));draw.text((x+8,y+230),f'Observation {meta["samples"][sample]["time"]:.0f}s; iteration {step}/16',fill='black')
    sheet.save(args.review/'process_contact_sheet.jpg',quality=95)
    report=dict(reference='https://diffusion-policy.cs.columbia.edu/',reference_local='/media/wanglab22/Expansion/highlight_pusht_process.mp4',source_video='videos/dp_eval/14/2.mp4',width=1280,height=720,fps=30,frames=len(timeline),duration_seconds=len(timeline)/30,playback_speed=4,candidate_count=6,inference_steps=16,prediction_horizon=16,returned_action_steps=15,renderer='Matplotlib Agg, plasma colormap; FFmpeg H.264',interpolation='None between model states. Each captured iteration is held for at least two frames.',interpretation='Actual offline samples from the supplied checkpoint. Playback after pauses is the original recording, not execution of the displayed samples. Frame/state alignment uses elapsed time and is approximate. Projected positions use a local motion estimate, not a measured camera calibration.',jaw_display='Raw values from candidate 1; 0 closed, 1 open after rounding and clipping. Other candidates are displayed in the image overlay only.',timeline=timeline)
    (dst/'render_provenance.json').write_text(json.dumps(report,indent=2));print(f'Saved {len(timeline)/30:.2f}s reference-style videos',flush=True)

if __name__=='__main__':main()
