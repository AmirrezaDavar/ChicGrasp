# Setup and real-robot usage

ChicGrasp extends the original [Diffusion Policy](https://github.com/real-stanford/diffusion_policy) environment. These are the existing project entry points. The presentation refresh did not perform a fresh dependency installation, retrain a policy, or run hardware.

## Install

```bash
git clone https://github.com/AmirrezaDavar/ChicGrasp.git
cd ChicGrasp
conda env create -f conda_environment_real.yaml -n chicgrasp_real
conda activate chicgrasp_real
pip install -e .
```

The Linux and macOS environment files from the upstream base are also retained. Review CUDA compatibility before changing dependencies.

## Hardware

- UR10e with RTDE, connected by Ethernet.
- Three Intel RealSense RGB views. The old README says D415, while the paper says D435; confirm the installed units when reproducing the setup.
- SpaceMouse for teleoperation, with `spacenavd` running.
- Arduino-controlled pneumatic valves and the custom dual-jaw gripper.

The gripper constructor defaults to `/dev/ttyACM0`. Set the actual port in `diffusion_policy/real_world/gripper_diff.py` or pass it when constructing the controller. The scripts do **not** read an `ARDUINO_PORT` environment variable. Robot IP is passed using `--robot_ip`.

## Collect demonstrations

After checking the robot workspace, emergency stop, camera feeds, serial port, and SpaceMouse:

```bash
python demo_real_robot.py \
  --output data/demo_pusht_real \
  --robot_ip 192.168.0.204
```

Use your actual robot IP. With the OpenCV window focused, press `C` to record, `S` to stop, and `Q` to exit. Review the source key mappings before operating the robot and gripper.

## Train

```bash
python train.py \
  --config-name=train_diffusion_unet_real_image_workspace \
  task.dataset_path=data/demo_pusht_real
```

`diffusion_policy/config/task/real_pusht_image.yaml` defines observation shapes and the eight-value stored action tensor. A complete demonstration dataset is needed; the supplied `training/checkpoint_*.zip` archives contain models and logs.

## Evaluate

```bash
python eval_real_robot.py \
  --input path/to/checkpoints/latest.ckpt \
  --output data/eval_chicgrasp \
  --robot_ip 192.168.0.204
```

This command can move real hardware. The archived script contains deployment-specific bounds and runtime overrides. Check those settings, the intended checkpoint, camera ordering, and jaw convention before use. See [the evidence page](evidence.html#differences) for differences between the paper, checkpoint, and runtime script.

## Preview the presentation

```bash
python -m http.server 8000 --directory docs
```

Visit `http://localhost:8000`. No robot, checkpoint download, or GPU is needed to view the page or its recorded action explorer.
