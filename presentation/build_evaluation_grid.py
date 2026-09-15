#!/usr/bin/env python3
"""Export every DP recording, then compose a synchronized 12-by-12 video grid."""
import argparse, concurrent.futures, hashlib, json, math, subprocess
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

def run(args):
    subprocess.run(args, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-root',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--workers',type=int,default=4); ap.add_argument('--camera',type=int,choices=[0,1,2],default=0); args=ap.parse_args()
    root=args.data_root; camera=str(args.camera); out=args.output; out.mkdir(parents=True,exist_ok=True)
    clips=out/'episodes'; clips.mkdir(exist_ok=True); rows=out/'work'; rows.mkdir(exist_ok=True)
    inventory=json.loads((root/'individual_action_videos_20260915/metadata/inventory.json').read_text())
    episodes=sorted((x for x in inventory if x['method']=='dp'),key=lambda x:x['episode'])
    assert [x['episode'] for x in episodes]==list(range(142))
    fps=24; speed=4; duration=math.ceil(max(x['cameras'][camera]['duration'] for x in episodes)/speed*fps)/fps+1
    font='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    def encode(x):
        ep=x['episode']; src=root/f'videos/dp_eval/{ep}/{camera}.mp4'; dst=clips/f'ep{ep:03d}.mp4'; poster=clips/f'ep{ep:03d}.jpg'
        source_duration=x['cameras'][camera]['duration']
        if not dst.exists():
            vf=f'setpts=(PTS-STARTPTS)/{speed},fps={fps},scale=640:360:flags=lanczos,setsar=1,drawtext=fontfile={font}:text=EP {ep:03d} | 4x:x=8:y=8:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.65:boxborderw=4'
            run(['ffmpeg','-v','error','-nostdin','-threads','1','-i',str(src),'-an','-vf',vf,'-c:v','libx264','-threads','1','-preset','fast','-crf','25','-pix_fmt','yuv420p','-movflags','+faststart',str(dst)])
        if not poster.exists():
            run(['ffmpeg','-v','error','-nostdin','-threads','1','-ss',str(min(3,source_duration/8)),'-i',str(dst),'-frames:v','1','-q:v','3',str(poster)])
        meta={'episode':ep,'source':f'videos/dp_eval/{ep}/{camera}.mp4','source_seconds':source_duration,'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'clip':f'episodes/ep{ep:03d}.mp4','poster':f'episodes/ep{ep:03d}.jpg','row':ep//12,'column':ep%12,'outcome':None}
        print(f'EP {ep:03d} exported',flush=True); return meta
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        items=sorted(pool.map(encode,episodes),key=lambda x:x['episode'])
    for row in range(12):
        dst=rows/f'row{row:02d}.mp4'
        if dst.exists(): continue
        cmd=['ffmpeg','-v','error','-nostdin']; filters=[]; inputs=[]
        for col in range(12):
            ep=row*12+col
            if ep<142:
                cmd+=['-threads','1','-i',str(clips/f'ep{ep:03d}.mp4')]
                filters.append(f'[{col}:v]scale=320:180:flags=lanczos,tpad=stop_mode=clone:stop_duration={duration},trim=duration={duration},setpts=PTS-STARTPTS[v{col}]')
            else:
                tile=rows/f'legend{ep}.png'
                im=Image.new('RGB',(320,180),'white');d=ImageDraw.Draw(im)
                title=ImageFont.truetype(font,18);small=ImageFont.truetype(font,13)
                lines=['ChicGrasp','142 DP recordings','Wrist view | 4x' if camera=='0' else 'Grasp phase | 4x'] if ep==142 else ['EP 000 - EP 141','Ends held in grid',f'Camera {camera}']
                for i,line in enumerate(lines):d.text((16,42+i*31),line,font=title if i==0 else small,fill='#333333')
                im.save(tile)
                cmd+=['-loop','1','-framerate',str(fps),'-i',str(tile)]
                filters.append(f'[{col}:v]trim=duration={duration},setpts=PTS-STARTPTS[v{col}]')
            inputs.append(f'[v{col}]')
        filters.append(''.join(inputs)+'hstack=inputs=12:shortest=1[out]')
        cmd+=['-filter_complex_threads','1','-filter_complex',';'.join(filters),'-map','[out]','-an','-t',str(duration),'-r',str(fps),'-c:v','libx264','-threads','2','-crf','19','-preset','fast','-pix_fmt','yuv420p',str(dst)]
        run(cmd);print(f'ROW {row} composed',flush=True)
    grid=out/'dp_all_142_grid_4k.mp4'
    if not grid.exists():
        cmd=['ffmpeg','-v','error','-nostdin']
        for row in range(12):cmd+=['-threads','1','-i',str(rows/f'row{row:02d}.mp4')]
        cmd+=['-filter_complex_threads','1','-filter_complex',''.join(f'[{i}:v]' for i in range(12))+'vstack=inputs=12:shortest=1[out]','-map','[out]','-an','-c:v','libx264','-threads','4','-crf','20','-preset','medium','-pix_fmt','yuv420p','-movflags','+faststart',str(grid)]
        run(cmd)
    web=out/'dp_all_142_grid_web.mp4'
    if not web.exists():run(['ffmpeg','-v','error','-nostdin','-threads','2','-i',str(grid),'-vf','scale=1920:1080:flags=lanczos','-an','-c:v','libx264','-threads','4','-crf','23','-preset','medium','-pix_fmt','yuv420p','-movflags','+faststart',str(web)])
    for name,src in [('dp_all_142_grid.jpg',grid),('dp_all_142_grid_web.jpg',web)]:
        if not (out/name).exists():run(['ffmpeg','-v','error','-nostdin','-ss','3','-i',str(src),'-frames:v','1','-q:v','2',str(out/name)])
    manifest={'count':142,'camera':int(camera),'columns':12,'rows':12,'fps':fps,'playback_speed':speed,'duration_seconds':duration,'scope':'All 142 supplied DP recordings, sorted by archive episode ID. These grasp-phase recordings stop before lift and scripted rehang. No per-episode published outcome mapping is available.','timing':'All recordings start together at uniform 4x speed. Each ending is held until the longest recording finishes; no episode loops independently.','episodes':items}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (out/'episodes.js').write_text('window.CHICGRASP_EPISODES='+json.dumps(manifest,separators=(',',':'))+';\n')
    print('ALL 142 EPISODES AND BOTH GRIDS COMPLETE',flush=True)
if __name__=='__main__':main()
