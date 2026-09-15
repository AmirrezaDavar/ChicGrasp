#!/usr/bin/env python3
"""Create the 84-second film, short clips, CAD still, and review contact sheet.

All visual motion comes from source recordings, camera orbit around actual CAD,
or captured model outputs. This renderer performs no robot or physics actions.
"""
from pathlib import Path
import argparse
import json
import math
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]
ASSETS=ROOT/'docs/assets';DATA=ASSETS/'data';MEDIA=ASSETS/'media';FIG=ASSETS/'figures'
W,H=1920,1080
BG='#10232e'; PANEL='#193440'; WHITE='#f6f4ef'; DIM='#acc0c6'; TEAL='#65d6c5'; GOLD='#f6b66d'; BLUE='#80b7f4'
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
BOLD='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
FONTS={}


def font(size,bold=False):
    key=(size,bold)
    if key not in FONTS:FONTS[key]=ImageFont.truetype(BOLD if bold else FONT,size)
    return FONTS[key]


def text(im,xy,value,size=30,fill=WHITE,bold=False):
    ImageDraw.Draw(im).text(xy,str(value),font=font(size,bold),fill=fill,spacing=10)


def line(im,points,color=TEAL,width=3):
    ImageDraw.Draw(im).line([tuple(map(float,p)) for p in points],fill=color,width=width)


def panel(im,box,fill=PANEL,r=24):
    ImageDraw.Draw(im).rounded_rectangle(box,radius=r,fill=fill)


def base(section,number=None):
    im=Image.new('RGB',(W,H),BG)
    text(im,(75,42),'CHICGRASP',27,TEAL,True)
    text(im,(75,100),section,49,bold=True)
    text(im,(75,1024),'AMIRREZA DAVAR  /  ROBOT LEARNING + HARDWARE',19,DIM)
    if number is not None:text(im,(1765,1024),f'{number:02d} / 07',19,DIM)
    return im


def fit(im,img,box):
    x,y,w,h=box
    copy=img.copy();copy.thumbnail((int(w),int(h)),Image.Resampling.LANCZOS)
    im.paste(copy,(int(x+(w-copy.width)/2),int(y+(h-copy.height)/2)))


class Videos:
    def __init__(self):self.caps={};self.indices={};self.frames={}
    def get(self,path,seconds):
        key=str(path)
        if key not in self.caps:
            self.caps[key]=cv2.VideoCapture(key);self.indices[key]=-100
        cap=self.caps[key];fps=cap.get(cv2.CAP_PROP_FPS)
        index=min(max(0,int(seconds*fps)),int(cap.get(cv2.CAP_PROP_FRAME_COUNT))-1)
        if index!=self.indices[key]:
            if index!=self.indices[key]+1:cap.set(cv2.CAP_PROP_POS_FRAMES,index)
            ok,bgr=cap.read()
            if not ok:raise RuntimeError(f'Cannot decode {path} frame {index}')
            self.frames[key]=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));self.indices[key]=index
        return self.frames[key]
    def close(self):
        for cap in self.caps.values():cap.release()


class CAD:
    def __init__(self):
        data=np.load(DATA/'gripper_mesh.npz');t=data['triangles'];self.groups=data['groups']
        center=(t.min(axis=(0,1))+t.max(axis=(0,1)))/2
        self.triangles=t-center
        normals=np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]);length=np.linalg.norm(normals,axis=1)
        self.normals=normals/np.maximum(length[:,None],1e-10)
    def render(self,size=(1100,820),az=-27,el=16,light=True):
        # Orthographic view of the intact assembly. Only the camera rotates.
        az,el=np.radians([az,el]);eye=np.array([np.sin(az)*np.cos(el),-np.cos(az)*np.cos(el),np.sin(el)])
        right=np.cross(eye,[0,0,1]);right/=np.linalg.norm(right);up=np.cross(right,eye)
        matrix=np.array([right,up,eye]).T
        xyz=self.triangles@matrix
        w,h=size;scale=min(w,h)*3.4
        uv=xyz[:,:,:2]*[scale,-scale]+[w*.5,h*.49]
        im=Image.new('RGB',size,WHITE if light else BG);d=ImageDraw.Draw(im)
        # Soft presentation shadow, with no claim of physical contact.
        if light:
            from PIL import ImageFilter
            sh=Image.new('RGBA',size);sd=ImageDraw.Draw(sh);sd.ellipse((w*.23,h*.83,w*.78,h*.9),fill=(20,43,54,30));sh=sh.filter(ImageFilter.GaussianBlur(17));im.paste(sh,(0,0),sh)
        colors=np.array([[153,165,172],[27,138,132],[209,145,77]])[self.groups]
        lambert=np.abs(self.normals@np.array([-.3,-.6,.74]));intensity=.48+.52*lambert
        rgb=np.clip(colors*intensity[:,None]+18,0,255).astype(np.uint8)
        order=np.argsort(xyz[:,:,2].mean(axis=1))
        for i in order:d.polygon([tuple(p) for p in uv[i]],fill=tuple(int(v) for v in rgb[i]))
        return im


def plan_projection(v,box):
    x,y,w,h=box
    # Equal normalized unit lengths on all three Cartesian axes.
    q=np.array([[.78,.3],[-.62,.4],[0,-.9]])
    return v[:,:3]@q*min(w,h)*.17+[x+w*.5,y+h*.55]


def denoise_frame(sample_index,step,small=False):
    im=base('How the policy generates an action',4)
    time=[5,12,20][sample_index];p=TRACES[f'physical_{sample_index}'][step];n=TRACES[f'normalized_{sample_index}'][step]
    text(im,(75,168),f'OFFLINE CHECKPOINT REPLAY   /   Observation at {time:.1f} s',23,TEAL)
    for cam in range(3):
        img=OBS[(time,cam)];fit(im,img,(75+cam*310,222,290,175));text(im,(75+cam*310,404),['Wrist RGB','Side RGB 1','Side RGB 2'][cam],20,DIM)
    panel(im,(1050,218,1835,444))
    text(im,(1085,246),'DENOISING ITERATION',23,DIM)
    text(im,(1085,282),f'{step:03d} / 100',65,TEAL,True)
    text(im,(1085,369),'16 planned actions  /  2 observations',23,DIM)
    panel(im,(75,477,959,958));panel(im,(993,477,1835,958))
    text(im,(106,502),'CARTESIAN TARGET SEQUENCE',24,WHITE,True)
    box=(115,536,790,370)
    origin=plan_projection(np.zeros((1,3)),box)[0]
    for j,c,label in zip(range(3),[TEAL,GOLD,BLUE],['x','y','z']):
        end=plan_projection(np.eye(3)[j:j+1]*2.5,box)[0];line(im,[origin,end],DIM,2);text(im,tuple(end+[8,-12]),label,22,c)
    uv=plan_projection(n,box);line(im,uv,TEAL,4)
    draw=ImageDraw.Draw(im)
    for j,(x,y) in enumerate(uv):draw.ellipse((x-5,y-5,x+5,y+5),fill=TEAL if 1<=j<=6 else WHITE)
    if step>=95:
        panel(im,(630,661,929,896),fill='#264652',r=12)
        text(im,(648,675),'Final XYZ detail',20,DIM)
        vv=p[:,:3];vv=vv-(vv.min(0)+vv.max(0))/2
        vv=vv/max(np.ptp(vv,axis=0).max(),1e-6)*3
        detail=plan_projection(vv,(655,711,242,150));line(im,detail,TEAL,3)
        for x,y in detail:draw.ellipse((x-3,y-3,x+3,y+3),fill=WHITE)
        text(im,(648,864),'Zoomed; equal axis scale',16,DIM)
    text(im,(106,912),'Normalized coordinates; points are future targets',20,DIM)
    text(im,(1025,502),'LEFT + RIGHT JAW VALUES',24,WHITE,True)
    x0,y0,ww,hh=1080,596,690,230
    for value in [0,.5,1]:
        yy=y0+hh*(1-value);line(im,[(x0,yy),(x0+ww,yy)],'#46616a',1);text(im,(1030,yy-12),str(value),18,DIM)
    # Raw diffusion iterates can exceed the [0,1] actuator domain. Clip only
    # the viewport to [0,1], and explicitly label it; export keeps all values.
    for idx,c in [(6,TEAL),(7,GOLD)]:
        xy=np.c_[np.linspace(x0,x0+ww,16),y0+hh*(1-np.clip(p[:,idx],0,1))];line(im,xy,c,4)
    for j in [1,6,16]:text(im,(x0+(j-1)/15*ww-10,y0+hh+10),j,18,DIM)
    text(im,(1080,548),'Left',23,TEAL,True);text(im,(1200,548),'Right',23,GOLD,True)
    text(im,(1370,548),'0 closed  /  1 open',23,DIM)
    text(im,(1080,879),'Planned action index  /  view clipped to [0, 1]',20,DIM)
    text(im,(1080,912),'At the final step: round, then clip to binary commands',19,DIM)
    text(im,(75,977),'New model sample on recorded inputs. These predictions were not executed. Camera alignment is approximate.',21,DIM)
    return im


def recorded_frame(seconds,videos,data_root):
    im=base('The two jaws close independently',5)
    text(im,(75,168),f'RECORDED EVALUATION   /   Episode 14   /   {seconds:04.1f} s',25,TEAL)
    fit(im,videos.get(data_root/'videos/dp_eval/14/0.mp4',seconds),(75,239,1115,645))
    text(im,(75,916),'Wrist camera + recorded commands  /  approximate elapsed-time alignment',22,DIM)
    t=np.array(REC['time']);a=np.array(REC['action']);idx=max(2,int(np.argmin(abs(t-seconds))))
    for offset,j,c,name in [(0,6,TEAL,'LEFT JAW'),(1,7,GOLD,'RIGHT JAW')]:
        y=252+offset*310;panel(im,(1230,y,1835,y+280));text(im,(1260,y+22),name,23,c,True)
        text(im,(1260,y+64),'OPEN' if a[idx,j]>=.5 else 'CLOSED',42,c,True)
        x0,y0,ww,hh=1270,y+146,510,78
        valid=(t>=18)&(t<=23);tt=t[valid];vv=a[valid,j]
        line(im,np.c_[x0+(tt-18)/5*ww,y0+(1-vv)*hh],c,3)
        xx=x0+(np.clip(seconds,18,23)-18)/5*ww;line(im,[(xx,y0-8),(xx,y0+hh+6)],WHITE,2)
        text(im,(1260,y+236),'18 s                                23 s',18,DIM)
    return im


def film_frame(t,videos,cad,data_root,cinematic):
    if t<8:
        im=base('Robot learning for delicate handling',1)
        text(im,(75,245),'ChicGrasp',94,bold=True)
        text(im,(80,380),'A custom dual-jaw gripper.\nA diffusion policy.\nOne coordinated grasp.',35,DIM)
        text(im,(80,605),'113 / 140',76,TEAL,True)
        text(im,(80,705),'successful published trials',27,DIM)
        text(im,(80,807),'FIRST-AUTHOR RESEARCH',22,GOLD,True)
        text(im,(80,854),'Advanced Robotics Research, 2026',27,WHITE)
        for row,cam in enumerate([1,0]):fit(im,videos.get(data_root/f'videos/dp_eval/14/{cam}.mp4',16+t),(925,225+row*367,910,342))
        text(im,(925,972),'Hardware grasp footage  /  1× playback',20,DIM)
        return im
    if t<16:
        im=base('Designed around two independently actuated jaws',2)
        fit(im,cad.render((1080,760),az=-28+(t-8)*9),(75,220,1080,760))
        text(im,(1200,282),'HARDWARE + CONTROL',24,TEAL,True)
        text(im,(1200,355),'Custom mechanical design\nPneumatic actuation\nIndependent jaw commands\nUR10e integration',33)
        text(im,(1200,643),'Align each leg.\nClose each jaw separately.',37,GOLD,True)
        text(im,(1200,823),'Local CAD assembly\nPresentation colors added',23,DIM)
        return im
    if t<24:
        im=base('Learning from multiview demonstrations',3)
        for cam in range(3):
            fit(im,videos.get(data_root/f'videos/dp_eval/14/{cam}.mp4',2+t-16),(75+cam*590,235,560,330));text(im,(75+cam*590,584),['Wrist RGB','Side RGB 1','Side RGB 2'][cam],25,DIM)
        for x,title,sub in [(75,'100 demonstrations','Reported training set'),(680,'3 camera views','Synchronized with robot + jaws'),(1285,'5 task commands','XYZ + two binary jaw commands')]:
            panel(im,(x,677,x+555,936));text(im,(x+30,725),title,32,TEAL,True);text(im,(x+30,799),sub,23,DIM)
        return im
    if t<42:
        local=t-24;sample=min(2,int(local/6));phase=local%6;step=min(100,int(phase/4.7*100))
        return denoise_frame(sample,step)
    if t<52:
        im=recorded_frame(18+(t-42)*.5,videos,data_root)
        text(im,(1260,910),'0.5× playback',25,GOLD)
        return im
    if t<65:
        im=base('Published benchmark',6)
        fit(im,Image.open(FIG/'published_results.png'),(65,220,1790,710))
        text(im,(75,949),'Policy-controlled grasping followed by scripted rehang  /  reported total cycle: approximately 38 s',23,DIM)
        return im
    if t<75:
        im=base('Ongoing: an Isaac Lab workbench',7)
        fit(im,videos.get(cinematic,10+(t-65)*1.7),(75,220,1260,710))
        text(im,(1390,285),'SIMULATION',25,GOLD,True)
        text(im,(1390,355),'UR10e + gripper\nArticulated anatomy\nContact modeling\nShackle interaction',30)
        text(im,(1390,652),'Held-pose cinematic\nFull released hang\nis not yet verified',24,DIM)
        text(im,(75,957),'Separate ongoing project  /  not part of the published hardware benchmark',23,DIM)
        return im
    im=base('From mechanical design to learned control')
    text(im,(95,278),'Amirreza Davar',80,bold=True)
    text(im,(100,412),'Gripper design  /  Data collection  /  Policy training\nReal-robot integration  /  Experimental evaluation',33,DIM)
    text(im,(100,628),'Explore the code, CAD, results, and action visualizations.',31,TEAL)
    text(im,(100,724),'github.com/AmirrezaDavar/ChicGrasp',45,WHITE,True)
    text(im,(100,845),'Paper: 10.1002/adrr.202500149',28,DIM)
    return im


def encode(path,frames,fps=24):
    path.parent.mkdir(parents=True,exist_ok=True)
    p=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24',
        '-s',f'{W}x{H}','-r',str(fps),'-i','-','-an','-c:v','libx264','-threads','4',
        '-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
    try:
        for f in frames:p.stdin.write(f.tobytes())
    finally:
        p.stdin.close()
        if p.wait()!=0:raise RuntimeError('ffmpeg encode failed')


def main():
    global TRACES,OBS,REC
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--data-root',type=Path,required=True)
    ap.add_argument('--cinematic',type=Path,required=True);ap.add_argument('--stills-only',action='store_true')
    args=ap.parse_args();MEDIA.mkdir(parents=True,exist_ok=True);FIG.mkdir(parents=True,exist_ok=True)
    TRACES=np.load(DATA/'denoising_trace.npz');REC=json.loads((DATA/'recorded_episode.json').read_text())
    OBS={(t,c):Image.open(DATA/f't{t}_camera_{c}.jpg').convert('RGB') for t in [5,12,20] for c in range(3)}
    videos=Videos();cad=CAD()
    cadim=Image.new('RGB',(1600,1000),WHITE);fit(cadim,cad.render((1140,810)),(20,145,1140,810))
    text(cadim,(70,48),'Independent jaws. Coordinated control.',44,'#142b36',True)
    text(cadim,(1140,338),'LOCAL CAD',24,'#067c78',True);text(cadim,(1140,408),'Custom gripper\nassembly',32,'#142b36',True)
    text(cadim,(1140,562),'Left jaw group',26,'#067c78');text(cadim,(1140,619),'Right jaw group',26,'#b67729')
    text(cadim,(70,948),'Composed geometry preserved. Colors added for readability. Camera holder is not included in this local assembly.',19,'#566874')
    cadim.save(FIG/'gripper_cad.png')
    film_frame(5,videos,cad,args.data_root,args.cinematic).save(FIG/'hero.jpg',quality=94)
    times=[4,12,19,26,29,35,40,46,50,59,70,79]
    sheet=Image.new('RGB',(1280,4*203),BG)
    for i,t in enumerate(times):
        im=film_frame(t,videos,cad,args.data_root,args.cinematic)
        if t in [12,40,46,70]:im.save(FIG/f'film_review_{t}.jpg',quality=94)
        im.thumbnail((426,180));sheet.paste(im,(i%3*426,i//3*203));text(sheet,(i%3*426+8,i//3*203+181),f'{t:02d} s',14,DIM)
    sheet.save(FIG/'film_contact_sheet.jpg',quality=92)
    if args.stills_only:return
    def frames():
        for i in range(84*24):
            if i%240==0:print(f'Film {i/24:.0f} / 84 seconds',flush=True)
            yield film_frame(i/24,videos,cad,args.data_root,args.cinematic)
    encode(MEDIA/'chicgrasp_84s.mp4',frames())
    print('Encoding standalone denoising loop',flush=True)
    encode(MEDIA/'diffusion_action_generation.mp4',(denoise_frame(2,min(100,int(i/24/6*100))) for i in range(8*24)))
    encode(MEDIA/'gripper_cad_orbit.mp4',(film_frame(8+i/24,videos,cad,args.data_root,args.cinematic) for i in range(8*24)))
    # Original evaluation footage is kept at native speed and with no retiming.
    for cam in [0,1,2]:
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(args.data_root/f'videos/dp_eval/14/{cam}.mp4'),
                        '-an','-vf','scale=960:-2','-c:v','libx264','-threads','2','-preset','fast','-crf','22','-pix_fmt','yuv420p','-movflags','+faststart',str(MEDIA/f'episode14_camera{cam}.mp4')],check=True)
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-ss','10','-t','10','-i',str(args.cinematic),'-an',
                    '-vf','scale=1280:-2','-c:v','libx264','-threads','2','-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',str(MEDIA/'simulation_workbench.mp4')],check=True)
    # Small GitHub-compatible preview with a direct link to the full film.
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-ss','2','-t','5','-i',str(MEDIA/'chicgrasp_84s.mp4'),
                    '-vf','fps=8,scale=640:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=96[p];[s1][p]paletteuse',str(MEDIA/'preview.gif')],check=True)
    videos.close()
    shots=[(0,8,'Introduction', 'dp_eval/14 cameras 0 and 1 at elapsed 16–24 s, 1×'),
           (8,16,'CAD', 'gripper_customized_1.usd composed mesh, camera orbit only'),
           (16,24,'Observations','dp_eval/14 all cameras at elapsed 2–10 s, 1×; paper demonstration count'),
           (24,42,'Action generation','Actual EMA checkpoint replay; samples at 5, 12, 20 s; 100 DDIM steps each'),
           (42,52,'Jaw commands','dp_eval/14 elapsed 18–23 s at 0.5×, approximate video/log alignment'),
           (52,65,'Results','Paper Table 4 transcription and calculated totals'),
           (65,75,'Ongoing simulation','Existing validated cinematic elapsed 10–27 s at 1.7×'),
           (75,84,'Credits','User-confirmed personal technical ownership; paper retains all coauthors')]
    (DATA/'edit_decision_list.json').write_text(json.dumps(dict(duration_seconds=84,fps=24,resolution=[W,H],audio='none; designed for silent playback',shots=shots),indent=2))
    print('All media encoded',flush=True)


if __name__=='__main__':main()
