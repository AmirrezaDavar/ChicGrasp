#!/usr/bin/env python3
"""Render an 80-second research overview with conventional figures and captions."""
import argparse,json,subprocess
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import render_media as legacy

ROOT=Path(__file__).resolve().parents[1];A=ROOT/'docs/assets';FPS=24;W,H=1920,1080
F='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf';B='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
fonts={}
def txt(im,xy,s,size=30,bold=False,fill='black'):
    key=(size,bold)
    if key not in fonts:fonts[key]=ImageFont.truetype(B if bold else F,size)
    ImageDraw.Draw(im).text(xy,s,font=fonts[key],fill=fill,spacing=14)
def fit(im,source,box):
    x,y,w,h=box;scale=min(w/source.width,h/source.height);source=source.resize((round(source.width*scale),round(source.height*scale)),Image.Resampling.LANCZOS);im.paste(source,(int(x+(w-source.width)/2),int(y+(h-source.height)/2)))
def base(title,caption):
    im=Image.new('RGB',(W,H),'white');txt(im,(80,40),title,36);txt(im,(80,1020),caption,22,fill='#444444');return im

def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True);p.add_argument('--review',type=Path,required=True);a=p.parse_args();a.review.mkdir(parents=True,exist_ok=True)
    legacy.WHITE='#ffffff';cad=legacy.CAD();cad_img=cad.render((1400,1100));cad_img.save(A/'figures/gripper_cad.png')
    videos=legacy.Videos();results=Image.open(A/'figures/published_results.png').convert('RGB');pipe=Image.open(A/'figures/policy_pipeline.png').convert('RGB');record=json.loads((A/'data/recorded_episode.json').read_text());rt=np.array(record['time']);ra=np.array(record['action'])
    media=A/'media';process=media/'chicgrasp_action_process.mp4';proc_duration=328/30
    def frame(t):
        if t<5:
            im=base('ChicGrasp','Davar et al. · Advanced Robotics Research (2026) · DOI: 10.1002/adrr.202500149')
            txt(im,(80,160),'Imitation learning with a customized\ndual-jaw gripper',51,bold=True)
            txt(im,(80,355),'Amirreza Davar',30)
            fit(im,videos.get(a.data_root/'videos/dp_eval/14/2.mp4',t),(790,430,1050,565))
            txt(im,(80,540),'UR10e robot\nIndependent pneumatic jaws\nDiffusion-policy grasping\nScripted rehang',30)
        elif t<13:
            im=base('Customized dual-jaw gripper','Local CAD assembly; the camera holder is omitted. Geometry is preserved.')
            fit(im,cad_img,(70,130,970,820));fit(im,videos.get(a.data_root/'videos/dp_eval/14/0.mp4',18+(t-5)*.5),(1050,175,790,445));txt(im,(1065,700),'Two independently commanded jaw groups\nallow different closure times for the legs.',28)
        elif t<25:
            im=base('Observation and action representation','Checkpoint inputs: three RGB views, end-effector pose, and the two jaw states.')
            for c in range(3):
                fit(im,videos.get(a.data_root/f'videos/dp_eval/14/{c}.mp4',2+t-13),(80+c*610,150,590,332));txt(im,(80+c*610,505),['Wrist camera','Side camera 1','Side camera 2'][c],25)
            fit(im,pipe,(80,580,1760,350))
        elif t<36:
            im=Image.new('RGB',(W,H),'white');fit(im,videos.get(process,min(t-25,proc_duration-1/30)),(0,0,W,H))
        elif t<48:
            sec=18+(t-36)*.5;im=base('Independent jaw commands','Recorded episode 14 · 0.5× playback · Command values, not force or contact measurements')
            fit(im,videos.get(a.data_root/'videos/dp_eval/14/0.mp4',sec),(80,145,1280,720));i=min(np.searchsorted(rt,sec,side='right')-1,len(rt)-1)
            txt(im,(1415,245),f'Time: {sec:.1f} s',29)
            txt(im,(1415,365),'Left jaw',29);txt(im,(1415,415),'OPEN' if ra[i,6]>=.5 else 'CLOSED',34,bold=True)
            txt(im,(1415,540),'Right jaw',29);txt(im,(1415,590),'OPEN' if ra[i,7]>=.5 else 'CLOSED',34,bold=True)
            txt(im,(80,915),'Right closure: 20.6 s       Left closure: 21.1 s',28)
        elif t<62:
            im=base('Published evaluation','Davar et al. (2026), Table 4 · 10 trials per carcass, 14 carcasses per method')
            fit(im,results,(70,125,1780,710));txt(im,(95,875),'Success: two-leg grasp, lift, and completed scripted rehang.',30);txt(im,(95,932),'113 / 140 successful trials (80.71%). Laboratory prototype evaluation.',28)
        elif t<72:
            im=base('Ongoing work: Isaac Lab','Separate simulation work · Held practice configuration; released two-hock hang remains unverified')
            fit(im,videos.get(media/'simulation_workbench.mp4',t-62),(80,120,1760,870))
        else:
            im=base('ChicGrasp','Full author list and citation are available in the paper and repository.')
            txt(im,(80,175),'Amirreza Davar',46,bold=True)
            txt(im,(80,305),'Mechanical design and fabrication\nDemonstration collection and dataset integration\nPolicy training and baseline comparisons\nRobot deployment and experimental evaluation',32)
            txt(im,(80,665),'github.com/AmirrezaDavar/ChicGrasp',32)
            txt(im,(80,735),'doi.org/10.1002/adrr.202500149',28)
            fit(im,cad_img,(1320,230,510,635))
        return im
    path=media/'chicgrasp_overview_80s.mp4';enc=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','1920x1080','-r',str(FPS),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
    samples=[]
    for i in range(80*FPS):
        t=i/FPS;im=frame(t);enc.stdin.write(im.tobytes())
        if i in [3*FPS,8*FPS,18*FPS,28*FPS,34*FPS,43*FPS,54*FPS,67*FPS,76*FPS]:
            im.save(a.review/f'overview_{int(t)}.jpg',quality=95);samples.append((int(t),im.copy()))
            if t==67:im.save(A/'figures/simulation_preview.jpg',quality=95)
        if i%(10*FPS)==0:print(f'Rendered {int(t)}/80 seconds',flush=True)
    enc.stdin.close();assert enc.wait()==0;videos.close()
    sheet=Image.new('RGB',(1440,900),'white');d=ImageDraw.Draw(sheet)
    for j,(t,im) in enumerate(samples):
        im.thumbnail((480,270));x=j%3*480;y=j//3*300;sheet.paste(im,(x,y));d.text((x+8,y+275),f'{t} s',fill='black')
    sheet.save(a.review/'overview_contact_sheet.jpg',quality=95)
    segments=[(0,5,'Title and hardware'),(5,13,'Custom gripper'),(13,25,'Observation and action representation'),(25,36,'Offline diffusion replay, estimated image projection'),(36,48,'Recorded jaw closure at half speed'),(48,62,'Published evaluation'),(62,72,'Separate ongoing simulation'),(72,80,'Technical contributions and project links')]
    (A/'data/overview_edit.json').write_text(json.dumps(dict(duration=80,fps=24,segments=segments,source='Supplied hardware recordings, actual offline checkpoint traces, local CAD, published counts, and existing simulation cinematic'),indent=2))
    vtt='WEBVTT\n\n'
    def stamp(s):return f'{int(s)//60:02d}:{int(s)%60:02d}.000'
    for st,en,label in segments:vtt+=f'{stamp(st)} --> {stamp(en)}\n{label}\n\n'
    (media/'chicgrasp_overview_80s.vtt').write_text(vtt.rstrip()+'\n');print('Saved 80-second overview')

if __name__=='__main__':main()
