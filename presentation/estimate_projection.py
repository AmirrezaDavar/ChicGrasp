#!/usr/bin/env python3
"""Estimate a local display projection from tracked gripper translation.

This is not a camera calibration or a TCP measurement. The image origin is a
manually selected midpoint of the two jaws. The linear displacement map is fit
to optical flow of rigid gripper features and logged end-effector translation.
Do not use this map for robot control or metric scene reconstruction.
"""
import argparse,json
from pathlib import Path
import cv2,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    root=Path(__file__).resolve().parents[1]
    record=json.loads((root/'docs/assets/data/recorded_episode.json').read_text());t=np.array(record['time']);xyz=np.array(record['robot_eef_pose'])[:,:3]
    path=a.data_root/'videos/dp_eval/14/2.mp4';cap=cv2.VideoCapture(str(path));fps=cap.get(cv2.CAP_PROP_FPS)
    ok,frame=cap.read();assert ok
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY);mask=np.zeros_like(gray);mask[112:184,556:630]=255
    points=cv2.goodFeaturesToTrack(gray,maxCorners=60,qualityLevel=.015,minDistance=3,mask=mask)
    initial=points.copy();trajectory=[np.zeros(2)];tracked=[len(points)];origin=np.array([535.,190.]);review={0:frame.copy()};displacement=np.zeros(2);idx=0
    while True:
        ok,frame=cap.read()
        if not ok:break
        idx+=1;newgray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        nxt,status,error=cv2.calcOpticalFlowPyrLK(gray,newgray,points,None,winSize=(31,31),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,40,.001))
        back,status2,_=cv2.calcOpticalFlowPyrLK(newgray,gray,nxt,None,winSize=(31,31),maxLevel=3)
        good=(status[:,0]>0)&(status2[:,0]>0)&(np.linalg.norm(back[:,0]-points[:,0],axis=1)<.5)
        assert good.sum()>=8,(idx,good.sum())
        # Track initial feature identities; use median total displacement.
        initial=initial[good];points=nxt[good]
        displacement=np.median(points[:,0]-initial[:,0],axis=0)
        trajectory.append(displacement);tracked.append(len(points));gray=newgray
        if idx in [int(v*fps) for v in [5,12,20,24]]:review[idx]=frame.copy()
    cap.release();vt=np.arange(len(trajectory))/fps;uv=np.array(trajectory)+origin
    # Entire alternating 2 s blocks are withheld; no pointwise random split.
    train=(np.floor(vt/2).astype(int)%2)==0
    best=None
    for offset in np.arange(-.35,.351,.01):
        valid=(vt+offset>=t[0])&(vt+offset<=t[-1]);x=np.c_[np.column_stack([np.interp(vt+offset,t,xyz[:,j]) for j in range(3)]),np.ones(len(vt))]
        m=np.linalg.lstsq(x[train&valid],uv[train&valid],rcond=None)[0];err=np.linalg.norm(x@m-uv,axis=1);score=np.mean(err[train&valid]**2)
        if best is None or score<best[0]:best=(score,offset,m,x,err,valid)
    _,offset,m,x,err,valid=best;held=(~train)&valid
    assert np.sqrt(np.mean(err[held]**2))<5,'Local projection failed held-out validation'
    # Retain the training-only matrix, so the reported held-out error remains honest.
    report=dict(kind='estimated_local_displacement_projection',episode=14,camera=2,image_size=[1280,720],matrix_xyz1_to_uv=m.tolist(),video_to_state_offset_seconds=float(offset),anchor_image_px=origin.tolist(),anchor_description='Manually selected midpoint between visible jaws in first frame, not a calibrated TCP location.',tracking='Pyramidal Lucas-Kanade flow on rigid pneumatic-body corners; median displacement of persistent features; forward-backward error <0.5 px.',initial_features=int(tracked[0]),final_features=int(tracked[-1]),split='Fit on even-numbered two-second video blocks; validate on odd blocks. Offset selected on training blocks only.',held_out_rmse_px=float(np.sqrt(np.mean(err[held]**2))),held_out_p95_px=float(np.percentile(err[held],95)),held_out_max_px=float(err[held].max()),training_rmse_px=float(np.sqrt(np.mean(err[train&valid]**2))),fitted_xyz_range_m=[x[train&valid,:3].min(0).tolist(),x[train&valid,:3].max(0).tolist()],limitations='Local affine displacement estimate for this fixed view and nearly fixed gripper orientation. The origin is illustrative. No measured intrinsics/extrinsics are available. Intermediate noisy positions extend beyond the fitted motion volume and are extrapolated. Not a camera calibration, executed-plan reconstruction, or metric validation of predicted targets.')
    (a.output/'projection.json').write_text(json.dumps(report,indent=2));np.savez_compressed(a.output/'projection_tracking.npz',video_time=vt,tracked_uv=uv,projected_uv=x@m,held_out=held,valid=valid,feature_count=tracked)
    plt.rcdefaults();fig,axes=plt.subplots(1,2,figsize=(8,3),layout='constrained');axes[0].plot(uv[:,0],uv[:,1],label='Tracked displacement',lw=1);axes[0].plot((x@m)[:,0],(x@m)[:,1],'--',label='Estimated projection',lw=1);axes[0].invert_yaxis();axes[0].set(xlabel='Image u (px)',ylabel='Image v (px)');axes[0].legend(fontsize=8)
    axes[1].plot(vt[held],err[held],'.',ms=2);axes[1].set(xlabel='Video time (s)',ylabel='Held-out error (px)');fig.savefig(a.output/'projection_validation.png',dpi=200);plt.close(fig)
    sheet=Image.new('RGB',(1280,390*len(review)),'white');d=ImageDraw.Draw(sheet)
    for row,(i,bgr) in enumerate(review.items()):
        q=np.round(uv[i]).astype(int);cv2.drawMarker(bgr,tuple(q),(255,255,255),cv2.MARKER_CROSS,20,2)
        for p in [q]:cv2.circle(bgr,tuple(p),8,(0,0,0),1)
        im=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));im.thumbnail((640,360));sheet.paste(im,(0,row*390));d.text((660,row*390+20),f'{i/fps:.1f} s; tracked image origin',fill='black')
    sheet.save(a.output/'projection_anchor_review.jpg');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
