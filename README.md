# ChicGrasp

[![ChicGrasp Video Demo](https://img.youtube.com/vi/YOUR_VIDEO_ID/maxresdefault.jpg)](https://www.youtube.com/watch?v=YOUR_VIDEO_ID "Watch the ChicGrasp demo")

ChicGrasp extends the official
[Diffusion Policy](https://github.com/real-stanford/diffusion_policy)
implementation to control a UR10e robot with a **dual-jaw pneumatic gripper**.

It includes:

- Training scripts based on Diffusion Policy (`train.py`)
- Evaluation scripts for simulation and real robot (`eval.py`, `eval_real_robot.py`)
- Demo scripts (`demo_pusht.py`, `demo_real_robot.py`)
- Conda environment files for Linux, macOS, and real-robot setups

---

## 🔗 Relation to Diffusion Policy

This repository is **built on top of** the original
[Diffusion Policy](https://github.com/real-stanford/diffusion_policy) code
(kept in the `diffusion_policy/` folder).

ChicGrasp mainly adds:

- An 5D action space (3 for robot, 2 for gripper)
  \[x, y, z, left_jaw, right_jaw\]
- Gripper logging and replay in the dataset
- Real-robot scripts for UR10e + Arduino-controlled pneumatic gripper

If you know Diffusion Policy, this repo will feel familiar.

---

## 📁 Repository Structure

```text
ChicGrasp/
├─ diffusion_policy/           # Core DP implementation (models, training, envs)
├─ plots/                      # Plotting / analysis scripts
├─ tests/                      # Unit tests (if any)
├─ conda_environment.yaml      # Default environment (Linux)
├─ conda_environment_macos.yaml# Environment for macOS
├─ conda_environment_real.yaml # Environment for real UR10e + gripper
├─ demo_pusht.py               # Example pusht demo
├─ demo_real_robot.py          # Example real-robot demo
├─ eval.py                     # Evaluation in sim
├─ eval_real_robot.py          # Evaluation on real robot
├─ multirun_metrics.py         # Helper for multi-run analysis
├─ ray_exec.py                 # Ray launcher
├─ ray_train_multirun.py       # Multi-run training
├─ setup.py                    # Package install script
├─ train.py                    # Main training entry point
└─ README.md

1. close the repo:
git clone https://github.com/AmirrezaDavar/ChicGrasp.git
cd ChicGrasp
