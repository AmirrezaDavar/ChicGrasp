#!/usr/bin/env python3
"""Render synchronized wrist-camera grids for the archived IBC/LSTM-GMM videos."""
import argparse
import concurrent.futures
import hashlib
import json
import math
import subprocess
from pathlib import Path

LABELS = {'ibc': 'IBC', 'lstm-gmm': 'LSTM-GMM'}
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
FPS = 24
SPEED = 4


def run(command):
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors='replace'))


def probe(path):
    return json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
        'stream=width,height,r_frame_rate:format=duration', '-of', 'json', str(path)
    ]))


def build(root, output, method):
    source_dir = root / 'videos' / f'{method}_eval'
    sources = sorted(source_dir.glob('*/0.mp4'), key=lambda p: int(p.parent.name))
    if not sources:
        raise ValueError(f'No wrist-camera recordings in {source_dir}')
    if len(sources) > 3:
        raise ValueError('Update the three-window layout before adding more recordings.')
    target = output / method
    clips = target / 'episodes'
    clips.mkdir(parents=True, exist_ok=True)
    items = []
    for index, source in enumerate(sources):
        episode = int(source.parent.name)
        duration = float(probe(source)['format']['duration'])
        clip = clips / f'ep{episode:03d}.mp4'
        poster = clips / f'ep{episode:03d}.jpg'
        if clip.exists() or poster.exists():
            raise FileExistsError(f'Use a new output directory to preserve {clip}')
        filters = (
            f'setpts=(PTS-STARTPTS)/{SPEED},fps={FPS},scale=1280:720:flags=lanczos,'
            f'setsar=1,drawtext=fontfile={FONT}:text=EP {episode:03d} | 4x:'
            'x=16:y=16:fontsize=30:fontcolor=white:box=1:boxcolor=black@0.65:boxborderw=6'
        )
        run(['ffmpeg', '-v', 'error', '-nostdin', '-threads', '1', '-i', str(source),
             '-an', '-vf', filters, '-c:v', 'libx264', '-threads', '2', '-preset', 'fast',
             '-crf', '22', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(clip)])
        run(['ffmpeg', '-v', 'error', '-nostdin', '-ss', str(min(1.5, duration / 8)),
             '-i', str(clip), '-vf', 'scale=640:360', '-frames:v', '1', '-q:v', '2', str(poster)])
        items.append({
            'episode': episode, 'source': str(source.relative_to(root)),
            'source_seconds': duration, 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'clip': f'episodes/{clip.name}', 'poster': f'episodes/{poster.name}',
            'row': 0, 'column': index, 'outcome': None
        })
        print(f'{LABELS[method]} episode {episode:03d} exported', flush=True)
    duration = math.ceil(max(x['source_seconds'] for x in items) / SPEED * FPS) / FPS + 1
    master = target / f'{method}_all_{len(items)}_grid_full.mp4'
    web = target / f'{method}_all_{len(items)}_grid_web.mp4'
    command = ['ffmpeg', '-v', 'error', '-nostdin']
    filters = []
    for index, item in enumerate(items):
        command.extend(['-threads', '1', '-i', str(target / item['clip'])])
        filters.append(
            f'[{index}:v]tpad=stop_mode=clone:stop_duration={duration},'
            f'trim=duration={duration},setpts=PTS-STARTPTS[v{index}]'
        )
    filters.append(''.join(f'[v{i}]' for i in range(len(items))) + f'hstack=inputs={len(items)}:shortest=1[out]')
    command.extend(['-filter_complex_threads', '1', '-filter_complex', ';'.join(filters),
                    '-map', '[out]', '-an', '-c:v', 'libx264', '-threads', '2', '-crf', '21',
                    '-preset', 'medium', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(master)])
    run(command)
    run(['ffmpeg', '-v', 'error', '-nostdin', '-threads', '1', '-i', str(master),
         '-vf', f'scale={640 * len(items)}:360:flags=lanczos', '-an', '-c:v', 'libx264',
         '-threads', '2', '-crf', '23', '-preset', 'medium', '-pix_fmt', 'yuv420p',
         '-movflags', '+faststart', str(web)])
    for video in [master, web]:
        run(['ffmpeg', '-v', 'error', '-nostdin', '-ss', '1.5', '-i', str(video),
             '-frames:v', '1', '-q:v', '2', str(video.with_suffix('.jpg'))])
    manifest = {
        'method': method, 'method_label': LABELS[method], 'count': len(items), 'camera': 0,
        'columns': len(items), 'rows': 1, 'fps': FPS, 'playback_speed': SPEED,
        'duration_seconds': duration,
        'scope': f'All {len(items)} available {LABELS[method]} wrist-camera recordings. '
                 'These are grasp-phase recordings, not the full published 140-trial evaluation. '
                 'No per-episode published outcome mapping is available.',
        'timing': 'Common starts, uniform 4x speed, and final frames held to a common ending.',
        'grid_full': master.name, 'grid_web': web.name, 'episodes': items
    }
    (target / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (target / 'episodes.js').write_text('window.CHICGRASP_EPISODES=' + json.dumps(manifest) + ';\n')
    print(f'{LABELS[method]} grid complete: {duration:.2f} s', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda method: build(args.data_root, args.output, method), LABELS))


if __name__ == '__main__':
    main()
