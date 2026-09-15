#!/usr/bin/env python3
"""Rebuild presentation tables from the paper transcription and archived logs.

No trial success is inferred from a filename, jaw bit, or selected video.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import zipfile

import numpy as np
import yaml
import zarr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'docs/assets/data'


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-root', type=Path, required=True)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    results = list(csv.DictReader((ROOT/'presentation/published_results.csv').open()))
    for r in results:
        for k in ['successes','trials','nominal_successes','nominal_trials','disturbed_successes','disturbed_trials']:
            r[k] = int(r[k])
        assert r['successes'] == r['nominal_successes']+r['disturbed_successes']
        assert r['trials'] == r['nominal_trials']+r['disturbed_trials']
    summary = []
    for method in ['Diffusion Policy','IBC','LSTM-GMM']:
        for split in ['all','seen','unseen']:
            rows = [r for r in results if r['method']==method and (split=='all' or r['split']==split)]
            n = sum(r['trials'] for r in rows)
            s = sum(r['successes'] for r in rows)
            summary.append(dict(method=method,split=split,successes=s,trials=n,success_percent=round(100*s/n,2)))
    assert [r['successes'] for r in summary[:3]] == [113,84,29]
    write_csv(OUT/'published_summary.csv',summary)
    write_csv(OUT/'published_results.csv',results)
    (OUT/'published_results.json').write_text(json.dumps(dict(rows=results,summary=summary),indent=2))
    inventory = dict(archives=[], videos={}, replay_buffers={}, training_runs=[])
    for p in sorted(args.data_root.glob('*/*.zip')):
        with zipfile.ZipFile(p) as z:
            members = z.infolist()
            inventory['archives'].append(dict(path=str(p.relative_to(args.data_root)),bytes=p.stat().st_size,
                                               members=len(members),uncompressed_bytes=sum(i.file_size for i in members)))
            if p.parent.name != 'training':
                continue
            for member in members:
                if not member.filename.endswith('logs.json.txt') or not member.file_size:
                    continue
                # Resumed logs can contain repeated epochs/global steps. Retain
                # the last record for each global step, then prefer the explicitly
                # logged epoch summary (has a learning-rate/sample/validation key).
                records = {}
                invalid = 0
                with z.open(member) as f:
                    for line in f:
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            invalid += 1
                            continue
                        records[record['global_step']] = record
                epoch_groups = {}
                for rec in records.values():
                    epoch_groups.setdefault(rec['epoch'],[]).append(rec)
                epoch_rows = []
                for epoch, group in sorted(epoch_groups.items()):
                    # Workspace emits an epoch-mean loss in the terminal record;
                    # train_action_mse_error is measured less frequently.
                    last = max(group,key=lambda x:x['global_step'])
                    row = dict(epoch=epoch,global_step=last['global_step'],train_loss=last['train_loss'])
                    epoch_rows.append(row)
                method = p.stem.removeprefix('checkpoint_')
                tag = method+'_'+str(len(inventory['training_runs']))
                write_csv(OUT/f'training_{tag}.csv',epoch_rows)
                prefix = member.filename.rsplit('/',1)[0]+'/'
                configs = [i.filename for i in members if i.filename.startswith(prefix) and i.filename.endswith('/files/config.yaml')]
                config_brief = []
                for name in configs:
                    cfg = yaml.safe_load(z.read(name))
                    cfg = {k:v.get('value',v) if isinstance(v,dict) else v for k,v in cfg.items()}
                    config_brief.append(dict(member=name,horizon=cfg.get('horizon'),n_obs_steps=cfg.get('n_obs_steps'),
                                             n_action_steps=cfg.get('n_action_steps'),training=cfg.get('training'),
                                             action_shape=cfg.get('task',{}).get('shape_meta',{}).get('action')))
                inventory['training_runs'].append(dict(method=method,archive=p.name,log_member=member.filename,
                    unique_steps=len(records),invalid_lines=invalid,epochs=len(epoch_rows),
                    first_epoch=epoch_rows[0]['epoch'],last_epoch=epoch_rows[-1]['epoch'],
                    first_loss=epoch_rows[0]['train_loss'],last_loss=epoch_rows[-1]['train_loss'],
                    csv=f'training_{tag}.csv',configs=config_brief))
    for folder in sorted((args.data_root/'videos').iterdir()):
        files = list(folder.glob('*/*.mp4'))
        inventory['videos'][folder.name] = dict(files=len(files),episodes=len({p.parent.name for p in files}),
                                               cameras=dict(Counter(p.stem for p in files)))
    for method in ['dp','ibc','lstm-gmm']:
        p = args.data_root/f'experiments/replay_buffer_{method}.zarr.zip'
        z = zarr.open_group(zarr.ZipStore(str(p),mode='r'),path=f'replay_buffer_{method}.zarr',mode='r')
        ends = z['meta/episode_ends'][:]
        inventory['replay_buffers'][method] = dict(episodes=len(ends),steps=int(ends[-1]),
            arrays={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in z['data'].items()},
            sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        if method=='dp':
            ep = 14
            start,end = int(ends[ep-1]),int(ends[ep])
            ts = z['data/timestamp'][start:end]
            ts = ts-ts[0]
            action = z['data/action'][start:end]
            pose = z['data/robot_eef_pose'][start:end]
            trace = dict(episode=ep,time=ts.round(4).tolist(),action=action.round(6).tolist(),
                robot_eef_pose=pose.round(6).tolist(),left_jaw=z['data/left_jaw'][start:end,0].tolist(),
                right_jaw=z['data/right_jaw'][start:end,0].tolist(),
                action_order=['x','y','z','rx','ry','rz','left_jaw','right_jaw'],
                jaw_convention='0 = closed, 1 = open',
                note='Recorded command/state log, not a diffusion denoising trace. Leading all-zero action rows are logger initialization and excluded from plots. Camera alignment uses elapsed episode time and is approximate.')
            (OUT/'recorded_episode.json').write_text(json.dumps(trace,separators=(',',':')))
    (OUT/'source_inventory.json').write_text(json.dumps(inventory,indent=2))
    print(json.dumps(dict(results=summary, videos=inventory['videos'],
                         runs=[{k:v for k,v in x.items() if k!='configs'} for x in inventory['training_runs']]),indent=2))


if __name__ == '__main__':
    main()
