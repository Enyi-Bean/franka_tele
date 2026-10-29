# U-Arm → Franka FR3 遥操与视觉触觉数据采集

使用 **franky/libfranka 直接控制单臂 FR3，无需 ROS2**。支持只遥操、视觉采集、视觉＋双指触觉采集。录制时直接生成 **LeRobot v2.1** 数据，不需要额外转换，也不需要安装 lerobot 或 PyTorch。末尾提供 pi0.5 服务器推理、本地执行的示例。

## 1. 硬件与代码入口

| 项目 | 原设备配置 / 入口 |
|---|---|
| 从臂 | Franka Research 3 + Franka Hand，FCI IP `192.168.3.1` |
| 主臂 | U-Arm，8 舵机（关节 ID 0–6，扳机 ID 7），中励串口协议，115200 baud |
| 主臂串口 | `/dev/ttyUSB0`，可用 `--port` 修改 |
| 腕部相机 | D435i，`233622077251` |
| 第三人称相机 | D435i，`135122077750` |
| 触觉（可选） | Xense 千觉，左 `OG001273`，右 `OG001275` |
| 只遥操 | `teleop.py` |
| 遥操＋录制 | `record.py` |
| 取流 / 写盘 | `camera_d435i.py`、`tactile_xense.py`、`lerobot_writer.py` |
| 初始化与诊断 | `enable_fci.py`、`home_robot.py`、`gripper_init.py`、`uarm_calibrate.py`、`snap_base.py` |
| 远端策略客户端 | `deploy_pi05.py`、`test_infer.py` |

序列号是原设备的默认值，换设备后要显式传参。`uarm_calib.json` 保留原主臂方向、扳机行程和 ready 姿态，换主臂需重新标定；每次开始跟随时仍会重新采主臂零点。

## 2. 环境安装

原环境：Ubuntu 22.04、Python 3.11、Linux x86_64、PREEMPT_RT 内核。真机控制需实时内核与权限，普通内核未在本交接版本验证。CPU 控制端无需 GPU。

```bash
conda create -n franka_tele python=3.11
conda activate franka_tele
cd /path/to/franka_tele
sudo apt-get install ffmpeg
python -m pip install -r requirements.txt
sudo bash setup_realtime_perms.sh
sudo usermod -aG dialout "$USER"
# 重新登录或重启后，组权限生效
```

### 安装 franky 控制包（无需拉取源码）

在上面的环境中直接安装适配机器人固件的预编译包。本项目原 FR3 的 **robot server version = 9**：

```bash
python -m pip install --pre franky-control \
  --extra-index-url https://timschneider42.github.io/franky/whl/by-robot-server-version/9/

# 不连接机械臂，验证运行本项目需要的接口
python -c "from franky import Robot, Gripper, JointMotion, JointImpedanceTracker; print('franky imports OK')"
python teleop.py --help
python record.py --help
```

`franky-control` 是安装包名，代码中通过 `import franky` 使用。安装匹配的 Linux wheel 后会带上所需的编译模块，不需要另行克隆 franky、手动编译源码或安装 ROS2。若使用其他机器人的固件，请将安装源末尾的 `9` 换成该机器人的 server version；出现 `IncompatibleVersionException` 时先检查这个对应关系。安装方式见 [franky 官方说明](https://github.com/TimSchneider42/franky#-python-quickstart-guide)。

原机已验证的版本是 `1.1.5.dev33+g67ef3acb.libfranka.0.17.0`，仅作为版本参考，**不要求同事找回这个旧开发版**。上面的安装命令会选择安装源当前提供的兼容候选包；本次未在全新环境和真机上验证该候选版本。若导入检查失败（尤其缺少 `JointImpedanceTracker`），说明所装包的接口不适用，应检查实际安装版本和上述安装源，不能跳过检查直接上机。

### 可选：触觉 SDK

纯视觉与只遥操不需要 Xense SDK。触觉采集原机使用厂商提供的 **xensesdk 2.1.2、cp311 Linux x86_64 wheel**；请向千觉厂商或设备负责人取得该安装包（无需拉取 SDK 仓库）：

```bash
python -m pip install /path/to/xensesdk-2.1.2-cp311-cp311-linux_x86_64.whl
bash setup_xense_udev.sh
# 重插传感器或重启后验证
python -c "from xensesdk import Sensor; print(Sensor.scanSerialNumber())"
```

采集设置 `disable_infer=True`，只取 Rectify/Difference，无需安装 GPU 推理 extra。不要为了取图安装 `xensesdk[full]`。Python 依赖文件记录的是原环境版本；触觉 wheel 是额外的厂商安装包，不需要下载 SDK 源码。

## 3. 每次上机准备

笔记本有线直连 Control 单元支持 FCI 的网口，按现场网络设置同网段地址（原笔记本 `192.168.3.100/24`）。确认到机械臂的路由没有被 VPN 接管。

```bash
conda activate franka_tele
cd /path/to/franka_tele
ip route get 192.168.3.1
# 重启后需要重新设置，避免实时控制丢包
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# 只查看状态；账号密码使用现场 Desk 凭据
python enable_fci.py --host 192.168.3.1 \
  --user <用户名> --password <密码> --status-only
# 去掉 --status-only 会取得控制权、解刹车并激活 FCI
```

也可以直接在 Desk 网页中解刹车、清错、激活 FCI。控制程序沿用 Desk 的末端负载配置，须与实际夹爪和安装件匹配。

首次安装、换主臂或丢失标定时：

```bash
python uarm_calibrate.py --host 192.168.3.1 --port /dev/ttyUSB0
python gripper_init.py --host 192.168.3.1
# 仅读状态
python gripper_init.py --status
python home_robot.py --dry-run
```

标定程序会按提示移动 FR3。真机归位、开始跟随前确认运动空间与急停可用。本版本保留原控制增益和较高碰撞阈值（`teleop.py` 的 `COLLISION_*`），只遥操还保留 reflex 自动恢复逻辑；这些参数来自原设备的接触任务，交接后应由现场负责人核对。

## 4. 只遥操

```bash
# 只读主臂并打印映射，不连接 FR3 控制
python teleop.py --dry-run --port /dev/ttyUSB0
# 推荐抓取任务使用带夹持力的二值夹爪
python teleop.py --host 192.168.3.1 --gripper-binary --gripper-force 40
# 不控夹爪
python teleop.py --no-gripper
```

按终端提示回车完成归位和主臂零点对齐；运行中空格离合，`q` 退出。`python teleop.py` 默认使用连续夹爪开度模式，`--gripper-binary` 改为二值抓取。默认释放主臂舵机力矩，若主臂依赖力矩支撑可加 `--keep-torque`。更保守的跟随可使用 `--speed 0.6 --damping-scale 1.5`。

## 5. 遥操并采集

### 只采视觉（腕部＋第三人称）

```bash
python record.py --task "put the cube on the plate" \
  --out data/cube_on_plate_vision --no-tactile \
  --wrist-serial 233622077251 --base-serial 135122077750
```

临时只接腕部相机可再加 `--no-base`。`record.py` 默认启用触觉，因此纯视觉必须加 `--no-tactile`。

### 视觉＋两指触觉

```bash
python record.py --task "put the cube on the plate" \
  --out data/cube_on_plate_tactile \
  --wrist-serial 233622077251 --base-serial 135122077750 \
  --tactile left=OG001273 right=OG001275
```

只接一只触觉时用 `--tactile left=OG001273`。传感器接好后，按启动提示在**所有触觉表面无接触**时回车校准 Difference 基线。采集不需要单独启动 `teleop.py`。

| 按键 | 行为 |
|---|---|
| 第一次 Enter | FR3 归位、重新采主臂零点 |
| 第二次 Enter | 开始遥操并录制 |
| 空格 | 暂停 / 恢复跟随 |
| `s` | 保存当前 episode |
| `d` | 丢弃当前 episode / 取消待录制状态 |
| `q` | 退出；录制中需先保存或丢弃 |

保存后按相同流程开始下一条。相同 `--out` 可续采，但图像路集合必须一致；增加或移除相机、触觉时使用新目录。数据源超过 0.2 秒没有新帧会丢弃当前 episode。Ctrl+C 的清理路径会尝试保存当前录制，主动丢弃请用 `d`。

相机默认都是 `640×480 @15 fps`、固定曝光 220、gain 16、白平衡 3800、base 不旋转。参数可用 `--exposure`、`--gain`、`--wb`、`--base-rotate` 修改，训练与部署要保持一致。两指触觉各存 `700×400` Rectify 和 Difference，共计 6 路视频。

查看设备或单独验图：

```bash
python -c "from camera_d435i import list_devices; print(list_devices())"
python snap_base.py --serial 135122077750 --out base_view.png
python snap_base.py --serial 233622077251 --out wrist_view.png
python tactile_xense.py --serial OG001273 --calibrate --save
```

## 6. 输出的数据格式

```text
data/<任务>/
├── data/chunk-000/episode_000000.parquet
├── videos/chunk-000/observation.images.<路名>/episode_000000.mp4
└── meta/
    ├── info.json
    ├── tasks.jsonl
    ├── episodes.jsonl
    ├── episodes_stats.jsonl
    ├── stats.json
    └── tactile_ref/             # 触觉基线 PNG 和 index.jsonl
```

数据集标记 `codebase_version=v2.1`，名义采集频率 15 Hz，视频编码 H.264/yuv420p。图像路为 `wrist`、`base`，加触觉后增加 `tactile_left`、`tactile_left_diff`、`tactile_right`、`tactile_right_diff`。

| 字段 | 定义 |
|---|---|
| `observation.state` | float32[8]：7 个实测关节角 rad ＋ 实测夹爪宽度 m |
| `action` | float32[8]：7 个绝对关节目标 rad ＋ 二值夹爪命令（1 闭合，0 张开） |
| `timestamp` | `frame_index / 15`，不是各设备硬件同步时间戳 |
| `task_index` | 关联 `meta/tasks.jsonl` 的语言指令 |

各取流线程异步运行，采集循环取最新帧，并非硬件同步。采集端使用 BGR 缓冲，编码视频后供常规 RGB 视频解码器读取。`tactile_ref` 是非标准附加元数据，迁移、上传或转 v3 时需单独保留。

保存时直接生成 LeRobot 数据。这里只保证原代码实现的 v2.1 接口；下游若要求 v3 或额外触觉输入适配，仍需相应转换或模型配置。本项目不包括 OXT/Zarr 导出及修改夹爪 action 为实测宽度的旧转换脚本。

无硬件写盘自测（会在 `/tmp` 生成随机数据）：

```bash
python lerobot_writer.py
```

## 7. 简单真机部署示例：远端 pi0.5

```text
本地双相机 RGB + 实测 state + prompt
                 ↓ WebSocket/msgpack
远端 GPU 服务器：pi0.5 + 输入输出变换
                 ↓ actions (50, 8)
本地 deploy_pi05.py：15 Hz 执行动作，后台阻抗控制持续运行
```

本目录保留的是**双相机纯视觉**客户端，不发送触觉；所需的轻量 `openpi_client` 通信模块已随本目录提供。控制电脑不需要拉取 openpi 仓库、安装模型或配置 GPU，只需安装额外依赖：

```bash
python -m pip install -r requirements-deploy.txt
python -c "from openpi_client.websocket_client_policy import WebsocketClientPolicy; print('client imports OK')"

# 若使用 SSH 转发，另开终端保持连接；按现场配置增加 -p <端口>
ssh -N -L 8000:localhost:8000 <用户>@<服务器>

# 服务器已启动且模型加载完毕后，先测试网络，不连接或驱动机械臂
python test_infer.py --host localhost --port 8000
# 接真机，按提示确认后自主执行
python deploy_pi05.py --host localhost --port 8000 \
  --fr3-host 192.168.3.1 --prompt "put the cube on the plate" \
  --wrist-serial 233622077251 --base-serial 135122077750 --exec-steps 10
```

服务器是独立的模型运行环境，需要由模型部署方预先提供符合下述接口的策略服务和训练权重。本目录不包含 pi0.5 模型服务端、训练代码和权重；遥操与数据采集不依赖服务器。

以已经配置好原 Franka pi0.5 服务的 GPU 服务器为例，在其已有 openpi 环境中启动：

```bash
cd /path/to/server/openpi
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 \
  .venv/bin/python examples/franka/serve_franka.py \
  --checkpoint /path/to/checkpoint --self-test
# 自测后，去掉 --self-test 启动服务，默认端口 8000
```

该命令对应原服务器的 `franka_single_real_vision` 配置及 `serve_franka.py`，用于说明已部署服务如何启动，并非要求控制端另下载一份代码。服务器的模型缓存、归一化参数、checkpoint 和输入输出变换需由部署方配置；也可连接任何满足相同协议的服务，直接连接时将客户端 `--host` 改为服务器地址。

发送契约：`observation/state` 为 float32[8]（rad＋夹爪 m），`observation/image` 为 base RGB uint8[H,W,3]，`observation/wrist_image` 为 wrist RGB，`prompt` 为任务字符串。

返回契约：`actions` 为 [50,8]，前 7 维必须已由服务器还原为**绝对关节角**，最后一维为 [0,1] 的闭合命令。原配置训练时将手臂 action 转为 delta，服务器通过 `AbsoluteActions` 加回当前 state；客户端不能再次加 state。

`--exec-steps 10` 每批只执行前 10 步，再取新观测推理；默认 50 步。推理期间机械臂保持上一目标，网络推理是阻塞调用，`q` 在动作执行循环内检查，不能保证网络等待时立即响应。模型必须与任务、相机布置、曝光、state/action 定义匹配；旧 cube-on-plate 模型使用过不同相机布置，不能仅凭能连通就视为适用于当前设备。

## 8. 交接范围

本目录只保留遥操、LeRobot 采集、必要初始化工具及轻量部署客户端。未复制数据集、模型权重、ROS2 工程、完整第三方仓库、厂商二进制或原仓库 Git 配置。轻量通信代码 `openpi_client/` 来自原 openpi fork，来源和许可证见 `THIRD_PARTY_NOTICES.md`。没有保存 Desk 密码或 SSH 账号。原目录保持不变。

拿到本目录后，控制端按第 2 节安装 Python 包与系统依赖，配置实时内核、权限和设备连接即可使用遥操及采集命令；触觉模式再安装厂商 wheel，远端部署模式再安装 requirements-deploy.txt 并连接已有策略服务。无需获取其他源码仓库。

交接整理验证了原环境下的 Python 语法、命令行入口、通信模块加载与离线数据写盘，尚未验证全新环境安装、真机控制或远端模型推理。
