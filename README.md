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

- A 5D action space (3 for robot, 2 for gripper):  
  \[x, y, z, left_jaw, right_jaw\]
- Gripper logging and replay in the dataset
- Real-robot scripts for UR10e + Arduino-controlled pneumatic gripper

If you know Diffusion Policy, this repo will feel familiar.

---

## Requirements & Installation

1. Clone the repo
```text
git clone https://github.com/AmirrezaDavar/ChicGrasp.git
cd ChicGrasp
```
2. Create a conda environment
```text
conda env create -f conda_environment_real.yaml -n chicgrasp_real
conda activate chicgrasp_real
```
These environments are based on the original Diffusion Policy requirements. For GPU / CUDA versions, please also refer to the official DP repo:
https://github.com/real-stanford/diffusion_policy#installation

3. Install the ChicGrasp package
```text
pip install -e .
```

## 📦 ChicGrasp data, checkpoints, and videos

Large files are hosted on Box in the **ChicGrasp-data** folder, separate from this code:

[ChicGrasp-data](https://uark.box.com/s/c9bnzfpy6shej765x8g0z3fzt3q8ay7p)

Locally, we mirror the same layout under `data/`:

```text
data/
  training/       # demonstration datasets (e.g., chicgrasp_real_image.zarr.zip)
  experiments/    # configs, logs, checkpoints for each method
  videos/         # evaluation videos (seen + unseen)
```


## 1) Training data
```bash
bash scripts/download_chicgrasp_training.sh
```


## 2) Experiment logs + checkpoints (Diffusion Policy CNN, IBC, LSTM-GMM)
```bash
bash scripts/download_chicgrasp_experiments.sh
```


## 3) Evaluation videos
```bash
bash scripts/download_chicgrasp_videos.sh
```


## 🤖 Hardware Setup (Real Robot)
ChicGrasp was developed with:

- UR10e robot arm RTDE interface is required: https://docs.universal-robots.com/tutorials/communication-protocol-tutorials/rtde-guide.html
- Three Intel RealSense D415 cameras https://www.intelrealsense.com/depth-camera-d415/
- One SpaceMouse https://3dconnexion.com/us/product/spacemouse-wireless/
- Arduino UNO R4 WiFi for valve control https://store-usa.arduino.cc/products/uno-r4-wifi
- Screws for gripper ()
- Dual-jaw pneumatic gripper (Airtac HFZ + 3D-printed jaws)
  - Airtac HFZ16: https://www.airtacs.com/products/airtac-hfz-air-fingerdouble-acting-hfz16-1
  - TAILONZ PNEUMATIC 1/4" NPT solenoid valve (2-position, 5-port): https://www.amazon.com/Pneumatic-Solenoid-4V210-08-Pilot-Operated-Connection/dp/B081PTW87K

- Linux PC connected:
  - via Ethernet to the UR10e
  - via USB to the Arduino

Set these environment variables (example):
```text
export UR_IP=192.168.0.10        # TODO: set to your UR10e IP
export ARDUINO_PORT=/dev/ttyACM0 # TODO: set to your Arduino serial port
```
## Demonstration, Training, and Evaluation 

Before running anything on the real system, check that:

- The UR10e controller is powered on and reachable over the network (keep the emergency stop within easy reach at all times).
- Your RealSense camera(s) are plugged into the workstation and working (you can verify with `realsense-viewer`).
- The SpaceMouse is connected and the `spacenavd` daemon is running (check with `systemctl status spacenavd`).

Start the demonstration-collection script. Press **`C`** to begin recording, use the SpaceMouse to teleoperate the robot, and press **`S`** to stop recording:

```bash
(chicgrasp_real)$ python demo_real_robot.py -o data/demo_pusht_real --robot_ip 192.168.0.204
```

This command will create a demonstration dataset in data/demo_pusht_real that follows the same structure

To train a Diffusion Policy model on this data, run:

```bash
(chicgrasp_real)$ python train.py --config-name=train_diffusion_unet_real_image_workspace task.dataset_path=data/demo_pusht_real
```
If your camera configuration differs from the default, update diffusion_policy/config/task/real_pusht_image.yaml accordingly.

After training finishes and you have a checkpoint at data/outputs/blah/checkpoints/latest.ckpt, you can evaluate the policy with:

```bash
(chicgrasp_real)$ python eval_real_robot.py -i data/outputs/blah/checkpoints/latest.ckpt -o data/eval_pusht_real --robot_ip 192.168.0.204
```

## 📥 CAD Models

The main ChicGrasp CAD models can be downloaded here:

- **[ChicGrasp Dual-Jaw Gripper Assembly](https://cad.onshape.com/documents/59651785fc8216c351878e9e/w/96ccd4d006239148f4499ab2/e/caa066b5de7d7eefbd6d0296)**
- **[Jaw Finger](https://cad.onshape.com/documents/f9923c0a774e494641001547/w/5449e3aa08570ad5e1bd725d/e/4c4594f9492c86e876574dad)**
- **[Flenge](https://cad.onshape.com/documents/2ba3c05d30474bc51e5caf05/w/b2a50accccf850db7a426ca7/e/8b4a64de694fb9754597638f)**
- **[Camera Holder](https://cad.onshape.com/documents/1ce782597a880b6af038303f/w/750def440809f62fce0fa768/e/c56676c9f0502747cf60a721)**

Open the links in a browser, then right-click on a part or the Part Studio tab in Onshape and select Export… to download STEP/STL files.

## 🧾 Citation

If you use ChicGrasp in your research, please cite this work:

```text
@article{davar2025chicgrasp,
  title   = {ChicGrasp: Imitation-Learning based Customized Dual-Jaw Gripper Control for Delicate, Irregular Bio-products Manipulation},
  author  = {Davar, Amirreza and Others},
  journal = {arXiv preprint arXiv:XXXXX},
  year    = {2025}
}
```
















