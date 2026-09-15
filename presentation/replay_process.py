#!/usr/bin/env python3
"""Record actual DDIM iterates from a supplied checkpoint and recorded observations.

Offline only: this module never imports the hardware environment or sends commands.
Run with the original robodiff environment. The checkpoint is read from the unchanged ZIP into memory. Each batch item
uses independent Gaussian noise; all candidates and scheduler states are saved.
"""
from pathlib import Path
import argparse
import gc
import hashlib
import io
import json
import sys
import time
import zipfile

import cv2
import dill
import hydra
import numpy as np
import torch
import zarr
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from diffusion_policy.real_world.real_inference_util import get_real_obs_dict


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-root', type=Path, required=True)
    ap.add_argument('--episode', type=int, default=14)
    ap.add_argument('--times', type=float, nargs='+', default=[5, 12, 20])
    ap.add_argument('--steps', type=int, default=16)
    ap.add_argument('--candidates', type=int, default=6)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--output', type=Path, default=ROOT/'docs/assets/data/process')
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    archive = args.data_root/'training/checkpoint_dp.zip'
    member = 'outputs/blah/checkpoints/latest.ckpt'
    print('Reading checkpoint into memory', flush=True)
    with zipfile.ZipFile(archive) as z:
        raw = io.BytesIO(z.read(member))
    digest = hashlib.sha256(raw.getbuffer()).hexdigest()
    payload = torch.load(raw, map_location='cpu', pickle_module=dill)
    raw.close()
    cfg = payload['cfg']
    print('Checkpoint loaded; constructing policy', flush=True)
    policy = hydra.utils.instantiate(cfg.policy)
    state_key = 'ema_model' if cfg.training.use_ema else 'model'
    policy.load_state_dict(payload['state_dicts'][state_key], strict=True)
    del payload
    gc.collect()
    policy.eval().to(args.device)
    policy.num_inference_steps = args.steps
    policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1
    # Use the public eval_real_robot.py inference overrides: 16 iterations,
    # horizon minus observation history plus one returned actions.
    z = zarr.open_group(zarr.ZipStore(str(args.data_root/'experiments/replay_buffer_dp.zarr.zip'), mode='r'),
                        path='replay_buffer_dp.zarr', mode='r')
    ends = z['meta/episode_ends'][:]
    start = int(ends[args.episode-1]) if args.episode else 0
    end = int(ends[args.episode])
    timestamps = z['data/timestamp'][start:end]
    relative = timestamps-timestamps[0]
    traces = []
    original_step = policy.noise_scheduler.step
    frames = {}
    current = []

    def recorded_step(model_output, timestep, sample, **kwargs):
        if not current:
            current.append(sample.detach().cpu().numpy().copy())
        result = original_step(model_output, timestep, sample, **kwargs)
        current.append(result.prev_sample.detach().cpu().numpy().copy())
        return result

    policy.noise_scheduler.step = recorded_step
    for requested in args.times:
        i = max(policy.n_obs_steps-1, int(np.argmin(abs(relative-requested))))
        inds = np.arange(i-policy.n_obs_steps+1, i+1)
        obs = {k: z['data/'+k][start+inds[0]:start+inds[-1]+1]
               for k,v in cfg.task.shape_meta.obs.items() if v.type != 'rgb'}
        for k,v in cfg.task.shape_meta.obs.items():
            if v.type != 'rgb':
                continue
            cam = int(k.split('_')[-1])
            cap = cv2.VideoCapture(str(args.data_root/f'videos/dp_eval/{args.episode}/{cam}.mp4'))
            images = []
            for j in inds:
                # Archived MP4s lack original per-frame wall-clock timestamps.
                # Align by elapsed time from the episode start, explicitly
                # recorded as an approximation in provenance.
                cap.set(cv2.CAP_PROP_POS_MSEC, float(relative[j])*1000)
                ok, bgr = cap.read()
                if not ok:
                    raise RuntimeError(f'Cannot decode {k} at {relative[j]}')
                images.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            cap.release()
            obs[k] = np.stack(images)
            frames[f't{requested:g}_{k}'] = images[-1]
        obs_np = get_real_obs_dict(obs, cfg.task.shape_meta)
        tensors = {k: torch.from_numpy(v).unsqueeze(0).repeat(args.candidates, *([1]*v.ndim)).to(args.device) for k,v in obs_np.items()}
        current.clear()
        began = time.monotonic()
        with torch.no_grad():
            result = policy.predict_action(tensors)
            normalized = np.stack(current)
            physical = policy.normalizer['action'].unnormalize(
                torch.from_numpy(normalized).to(args.device)).cpu().numpy()
        traces.append(dict(time=float(relative[i]), sample_index=int(i),
                           normalized=normalized, physical=physical,
                           action=result['action'].cpu().numpy(),
                           action_pred=result['action_pred'].cpu().numpy()))
        print(f'Captured {len(current)-1} DDIM steps at t={relative[i]:.1f}s in {time.monotonic()-began:.1f}s', flush=True)
    arrays = {}
    for i, trace in enumerate(traces):
        arrays.update({f'{k}_{i}':v for k,v in trace.items() if isinstance(v,np.ndarray)})
    np.savez_compressed(args.output/'denoising_trace.npz', **arrays)
    for name, rgb in frames.items():
        cv2.imwrite(str(args.output/f'{name}.jpg'), cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
    metadata = dict(kind='offline_checkpoint_replay', checkpoint_archive=archive.name,
                    checkpoint_member=member, checkpoint_sha256=digest, state_key=state_key,
                    episode=args.episode, seed=42, inference_steps=args.steps, candidates=args.candidates,
                    horizon=int(policy.horizon), n_obs_steps=int(policy.n_obs_steps),
                    n_action_steps=int(policy.n_action_steps), action_order=['x','y','z','rx','ry','rz','left_jaw','right_jaw'],
                    alignment='Approximate: video elapsed time matched to replay-buffer episode elapsed time; original camera timestamps unavailable.',
                    interpretation='Actual model denoising iterates on recorded observations; these newly sampled actions were not executed on the robot and are not the original deployment predictions.',
                    samples=[{k:v for k,v in x.items() if not isinstance(v,np.ndarray)} for x in traces],
                    torch_version=torch.__version__, policy_config=OmegaConf.to_container(cfg.policy,resolve=True))
    (args.output/'denoising_provenance.json').write_text(json.dumps(metadata,indent=2))
    print('Saved actual denoising traces', flush=True)


if __name__ == '__main__':
    main()
