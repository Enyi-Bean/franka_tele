# U-Arm → Franka FR3 Teleoperation and Visual-Tactile Data Collection

[简体中文](README.md) | **English**

Control a single Franka FR3 directly with **franky/libfranka, without ROS2**. Supports teleoperation, visual demonstration recording, and visual recording with tactile sensors on both fingers. Recordings are written directly in **LeRobot v2.1** format: no separate conversion, lerobot package, or PyTorch installation is needed. A remote pi0.5 inference example is included below.

## 1. Hardware and entry points

| Component | Original setup / entry point |
|---|---|
| Follower | Franka Research 3 + Franka Hand, FCI IP `192.168.3.1` |
| Leader | U-Arm, 8 servos (joint IDs 0–6, trigger ID 7), Zhongling serial protocol, 115200 baud |
| Leader serial port | `/dev/ttyUSB0`; override with `--port` |
| Wrist camera | D435i, `233622077251` |
| Third-person camera | D435i, `135122077750` |
| Optional tactile sensors | Xense, left `OG001273`, right `OG001275` |
| Teleoperation only | `teleop.py` |
| Teleoperation and recording | `record.py` |
| Sensor streaming / dataset writing | `camera_d435i.py`, `tactile_xense.py`, `lerobot_writer.py` |
| Setup and diagnostics | `enable_fci.py`, `home_robot.py`, `gripper_init.py`, `uarm_calibrate.py`, `snap_base.py` |
| Remote policy client | `deploy_pi05.py`, `test_infer.py` |

Serial numbers are defaults for the original hardware. Override them when replacing devices. `uarm_calib.json` contains the original leader joint directions, trigger range, and ready pose. Recalibrate when replacing the leader. Its zero position is sampled again each time following starts.

## 2. Environment setup

Original environment: Ubuntu 22.04, Python 3.11, Linux x86_64, and a PREEMPT_RT kernel. Real robot control requires a real-time kernel and appropriate permissions; a standard kernel has not been validated for this handoff. The control laptop does not need a GPU.

```bash
conda create -n franka_tele python=3.11
conda activate franka_tele
cd /path/to/franka_tele
sudo apt-get install ffmpeg
python -m pip install -r requirements.txt
sudo bash setup_realtime_perms.sh
sudo usermod -aG dialout "$USER"
# Log out and back in, or reboot, to apply group membership changes.
```

### Install franky (no source checkout required)

Install a prebuilt package matching the robot firmware in the environment above. The original FR3 has **robot server version 9**:

```bash
python -m pip install --pre franky-control \
  --extra-index-url https://timschneider42.github.io/franky/whl/by-robot-server-version/9/

# Check required interfaces without connecting to the robot.
python -c "from franky import Robot, Gripper, JointMotion, JointImpedanceTracker; print('franky imports OK')"
python teleop.py --help
python record.py --help
```

The package is named `franky-control` and is used as `import franky`. A matching Linux wheel includes the required compiled modules; cloning franky, building it manually, or installing ROS2 is unnecessary. For other firmware, replace `9` in the index URL with the robot's server version. Check this match if you encounter `IncompatibleVersionException`. See the [official franky instructions](https://github.com/TimSchneider42/franky#-python-quickstart-guide).

The original tested version was `1.1.5.dev33+g67ef3acb.libfranka.0.17.0`. This is a reference, **not a requirement to recover that old development build**. The command above selects a candidate currently available from the package sources. That candidate has not been tested in a clean environment or on the real robot as part of this handoff. If imports fail, especially for `JointImpedanceTracker`, check the installed version and package source before using the robot.

### Optional tactile SDK

Teleoperation and visual-only recording do not require Xense SDK. The original tactile setup uses the vendor-provided **xensesdk 2.1.2 cp311 Linux x86_64 wheel**. Obtain this installer from Xense or the hardware owner; no SDK source checkout is needed.

```bash
python -m pip install /path/to/xensesdk-2.1.2-cp311-cp311-linux_x86_64.whl
bash setup_xense_udev.sh
# Reconnect the sensors or reboot, then check discovery.
python -c "from xensesdk import Sensor; print(Sensor.scanSerialNumber())"
```

Recording uses `disable_infer=True` and captures only Rectify/Difference images. GPU inference extras are unnecessary; do not install `xensesdk[full]` just to capture images. Python dependency pins reflect the original environment. The tactile wheel is a separate vendor package.

## 3. Before each robot session

Connect the laptop by Ethernet to the Control unit port supporting FCI. Configure an address on the robot's subnet (the original laptop used `192.168.3.100/24`). Ensure the robot route is not redirected through a VPN.

```bash
conda activate franka_tele
cd /path/to/franka_tele
ip route get 192.168.3.1
# Repeat after reboot to avoid real-time control packet loss.
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Status only; use your site's Desk credentials.
python enable_fci.py --host 192.168.3.1 \
  --user <username> --password <password> --status-only
# Remove --status-only to take control, unlock the brakes, and enable FCI.
```

Alternatively, clear errors, unlock the brakes, and enable FCI in the Desk web interface. Control scripts retain the Desk end-effector load settings; these must match the actual gripper and attached hardware.

For initial setup, a replacement leader, or lost calibration:

```bash
python uarm_calibrate.py --host 192.168.3.1 --port /dev/ttyUSB0
python gripper_init.py --host 192.168.3.1
# Read status only.
python gripper_init.py --status
```

Calibration moves the FR3 as prompted. Check workspace clearance and emergency stop access before homing or following. This version retains the original control gains and relatively high collision thresholds (`COLLISION_*` in `teleop.py`); teleoperation also retains automatic reflex recovery. These settings came from contact tasks on the original setup and should be reviewed by the person responsible for the robot.

### Home the FR3 to the ready pose

Unlock the brakes and enable FCI first. Run this separately after exiting any active teleoperation, recording, or deployment program:

```bash
# Inspect current joint state and target pose without moving.
python home_robot.py --dry-run
# Move to the default ready pose.
python home_robot.py
# Home more slowly.
python home_robot.py --factor 0.03
```

The default robot address is `192.168.3.1`; add `--host <robot_IP>` to change it. Actual homing requires typing `yes` at the confirmation prompt and uses a default speed factor of 0.05. The same command can be used after a teleoperation session.

## 4. Teleoperation only

```bash
# Read the leader and print mapped targets without connecting to FR3 control.
python teleop.py --dry-run --port /dev/ttyUSB0
# Use binary gripper control with grasping force for grasping tasks.
python teleop.py --host 192.168.3.1 --gripper-binary --gripper-force 40
# Disable gripper control.
python teleop.py --no-gripper
```

Press Enter as prompted to home the robot and align the leader zero. During operation, Space toggles the clutch and `q` exits. `python teleop.py` defaults to continuous gripper opening control; `--gripper-binary` enables binary grasping. Leader servo torque is released by default; use `--keep-torque` if the leader relies on torque for support. For more conservative following, try `--speed 0.6 --damping-scale 1.5`.

## 5. Teleoperation and recording

The commands below use the built-in camera defaults: wrist `233622077251`, third-person `135122077750`. Serial number arguments are unnecessary with those devices. Only add `--wrist-serial <wrist_serial> --base-serial <base_serial>` when replacing cameras.

### Visual-only recording (wrist and third-person)

```bash
python record.py --task "put the cube on the plate" \
  --out data/cube_on_plate_vision --no-tactile
```

Add `--no-base` to use only the wrist camera. Tactile recording is enabled by default in `record.py`, so visual-only recording requires `--no-tactile`.

### Visual recording with tactile sensors on both fingers

```bash
python record.py --task "put the cube on the plate" \
  --out data/cube_on_plate_tactile
```

Defaults are left `OG001273` and right `OG001275`; no extra arguments are needed. For replacement sensors, pass `--tactile left=<left_serial> right=<right_serial>`. For a single sensor, use `--tactile left=OG001273`. At startup, press Enter as prompted to calibrate the Difference baseline while **all tactile surfaces are free of contact**. Do not start `teleop.py` separately for recording.

| Key | Action |
|---|---|
| First Enter | Home the FR3 and sample the leader zero |
| Second Enter | Start following and recording |
| Space | Pause / resume following |
| `s` | Save the current episode |
| `d` | Discard the current episode / cancel the armed state |
| `q` | Exit; save or discard an active recording first |

Repeat the same sequence for the next episode. Reuse `--out` to append episodes, but the set of image streams must match. Use a new directory when adding or removing cameras or tactile sensors. If a source has no new frame for over 0.2 seconds, the current episode is discarded. Ctrl+C cleanup attempts to save an active recording; use `d` to discard it deliberately.

Both cameras default to `640×480 @15 fps`, fixed exposure 220, gain 16, white balance 3800, and no base-camera rotation. Override these with `--exposure`, `--gain`, `--wb`, and `--base-rotate`; keep training and deployment settings consistent. Each tactile sensor records `700×400` Rectify and Difference images, giving six video streams with both cameras and both sensors.

List devices or check individual sensor images:

```bash
python -c "from camera_d435i import list_devices; print(list_devices())"
python snap_base.py --serial 135122077750 --out base_view.png
python snap_base.py --serial 233622077251 --out wrist_view.png
python tactile_xense.py --serial OG001273 --calibrate --save
```

## 6. Dataset format

```text
data/<task>/
├── data/chunk-000/episode_000000.parquet
├── videos/chunk-000/observation.images.<stream>/episode_000000.mp4
└── meta/
    ├── info.json
    ├── tasks.jsonl
    ├── episodes.jsonl
    ├── episodes_stats.jsonl
    ├── stats.json
    └── tactile_ref/             # Tactile baseline PNGs and index.jsonl
```

Datasets declare `codebase_version=v2.1`, a nominal recording rate of 15 Hz, and H.264/yuv420p video encoding. Image streams are `wrist` and `base`, with `tactile_left`, `tactile_left_diff`, `tactile_right`, and `tactile_right_diff` added for tactile recording.

| Field | Meaning |
|---|---|
| `observation.state` | float32[8]: 7 measured joint angles in rad + measured gripper width in m |
| `action` | float32[8]: 7 absolute joint targets in rad + binary gripper command (1 close, 0 open) |
| `timestamp` | `frame_index / 15`, not hardware-synchronized sensor timestamps |
| `task_index` | Language instruction reference in `meta/tasks.jsonl` |

Sensor threads run asynchronously; the recording loop samples their latest frames without hardware synchronization. Capture buffers use BGR; encoded videos are read by standard RGB video decoders. `tactile_ref` is additional, nonstandard metadata and must be preserved separately during migration, upload, or conversion to v3.

Saving directly produces a LeRobot dataset. This project implements the original v2.1 interface; downstream tools requiring v3 or additional tactile model inputs still need conversion or model configuration. OXT/Zarr exports and the old scripts replacing gripper actions with measured widths are excluded.

Run the writer self-check without hardware (creates random data under `/tmp`):

```bash
python lerobot_writer.py
```

## 7. Real robot deployment example: remote pi0.5

```text
Local dual-camera RGB + measured state + prompt
                 ↓ WebSocket/msgpack
Remote GPU server: pi0.5 + input/output transforms
                 ↓ actions (50, 8)
Local deploy_pi05.py: execute at 15 Hz while impedance control runs in the background
```

The included client uses **two cameras, without tactile input**. Its lightweight `openpi_client` communication modules are included. The control computer does not need an openpi source checkout, model installation, or GPU; install the extra dependencies:

```bash
python -m pip install -r requirements-deploy.txt
python -c "from openpi_client.websocket_client_policy import WebsocketClientPolicy; print('client imports OK')"

# Keep SSH forwarding running in another terminal; add -p <port> as needed.
ssh -N -L 8000:localhost:8000 <user>@<server>

# Once the server and model are ready, test the network without controlling the robot.
python test_infer.py --host localhost --port 8000
# Connect the robot and confirm the prompts to start autonomous execution.
python deploy_pi05.py --host localhost --port 8000 \
  --fr3-host 192.168.3.1 --prompt "put the cube on the plate" \
  --exec-steps 10
```

The server is a separate model environment. The model deployment owner must provide a policy service with trained weights matching the contract below. This directory does not include the pi0.5 model server, training code, or weights. Teleoperation and recording do not require a server.

For a GPU server already configured with the original Franka pi0.5 service, start it in its existing openpi environment:

```bash
cd /path/to/server/openpi
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 \
  .venv/bin/python examples/franka/serve_franka.py \
  --checkpoint /path/to/checkpoint --self-test
# Remove --self-test after the check to serve on port 8000 by default.
```

This example uses the original server's `franka_single_real_vision` configuration and `serve_franka.py`; it explains starting an existing service, not downloading another codebase on the control computer. The deployment owner configures model caches, normalization statistics, checkpoints, and input/output transforms. Any service implementing the same protocol can be used. For a direct connection, set the client's `--host` to the server address.

Request contract: `observation/state` is float32[8] (joint rad + gripper m); `observation/image` is base RGB uint8[H,W,3]; `observation/wrist_image` is wrist RGB; `prompt` is the task string.

Response contract: `actions` has shape [50,8]. The server must already have restored the first seven dimensions to **absolute joint angles**; the last dimension is a closing command in [0,1]. The original configuration trains arm actions as deltas, and the server uses `AbsoluteActions` to add the current state back. The client must not add state again.

`--exec-steps 10` executes the first 10 steps of each batch before observing and inferring again; the default is 50. During inference the robot holds its previous target. Network inference blocks, and `q` is checked in the action execution loop, so it may not respond immediately while waiting on the network. The model must match the task, camera placement, exposure, and state/action definitions. The old cube-on-plate model used a different camera arrangement; a successful connection alone does not establish suitability for the current setup.

## 8. Handoff scope

This directory contains teleoperation, LeRobot recording, essential setup tools, and a lightweight deployment client. Datasets, model weights, ROS2 projects, complete third-party repositories, vendor binaries, and the original Git configuration are excluded. `openpi_client/` comes from the original openpi fork; see `THIRD_PARTY_NOTICES.md` for attribution and licensing. No Desk passwords or SSH accounts are stored. The original source directory is unchanged.

On the control computer, install the Python and system dependencies in Section 2, configure the real-time kernel and permissions, and connect the hardware to use teleoperation and recording. Tactile recording additionally requires the vendor wheel. Remote deployment additionally requires `requirements-deploy.txt` and an existing policy service. No other source repositories are needed on the control computer.

Python syntax, CLI entry points, communication module loading, and offline dataset writing have been checked in the original environment. Clean-environment installation, real robot control, and remote model inference have not been revalidated as part of this handoff.
