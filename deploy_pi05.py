#!/usr/bin/env python3
"""阶段三：pi05 真机部署（笔记本瘦客户端）。

服务器(A100)跑 pi05 推理，本机只做 I/O + 驱动 FR3：
    每步读 [两相机 + 8 维 state] → 组 obs 发给服务器 → 收 actions(50,8) → 驱动机械臂。

模型输出的臂 delta 已在**服务器端**用 AbsoluteActions 还原成绝对关节角，
所以这里收到的 actions[:, :7] 直接是绝对关节角(rad)，actions[:, 7] 是夹爪指令[0,1]。

一致性（必须和采集 record.py 完全对齐，否则模型没见过）：
    - base 相机(D435i 135122077750)：固定曝光 220 / 增益 16 / 白平衡 3800 / rotate=0 / 15fps
    - wrist 相机(D435i 233622077251)：固定曝光 220 / 增益 16 / 白平衡 3800 / 15fps
      （2026-09-19 起。此前 base 是 D405 固定曝光，cube_on_plate_2cam 那 54 条是那套参数）
    - 图像 BGR→RGB（相机给 BGR，训练存 RGB）
    - state = [robot.current_joint_state.position(7 rad), gripper.width(m)]
    - 夹爪 a[7]>0.5 → grasp(0.0, 0.2, 40N)，否则 open(0.2)，仅二值状态变化时发
    - 起点：先归位 ready 再开跑

节拍：数据 fps=15，actions 按 15Hz 逐步回放（每步 66.7ms），50 步 ≈ 3.3 秒/批。
推理期间由后台 TrackerDriver 线程持续 tick，保证 1kHz 控制不断供。

先在另一个终端起 SSH 端口转发（把本机 8000 转到服务器 8000）：
    ssh -N -L 8000:localhost:8000 <user>@<server>

用法：
    python deploy_pi05.py                                  # 连 localhost:8000
    python deploy_pi05.py --prompt "put the cube on the plate"
    python deploy_pi05.py --exec-steps 25                  # 每批只执行前 25 步就重推理
"""

from __future__ import annotations

from argparse import ArgumentParser
import threading
import time

import numpy as np

import franky
from franky import Gripper, JointImpedanceTracker, JointMotion, Robot

from openpi_client import websocket_client_policy

# 复用采集/遥操已验证的组件与常量
from teleop import (
    KeyPoller,
    STIFFNESS, DAMPING, FR3_LOWER, FR3_UPPER,
    COLLISION_TORQUE, COLLISION_FORCE,
)
from camera_d435i import CameraD435i, list_devices

# ---- 采集节拍与相机参数：必须和 record.py 一致 ----
FPS = 15
CAM_W, CAM_H = 640, 480
WRIST_SERIAL = "233622077251"   # 两台同型号，必须按序列号锁定（与 record.py 一致）
BASE_SERIAL = "135122077750"
WRIST_FPS = 15                  # 与 record.py 一致（USB2 带宽）
BASE_FPS = 15
CAM_EXPOSURE = 220              # 固定曝光，与 record.py 一致
CAM_GAIN = 16
CAM_WB = 3800
BASE_ROTATE = 0

# ready 起点：= home_robot.py 的 "ready" = uarm_calib.json 的 franka_init_rad
READY = np.array([0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, np.pi / 4])

# 夹爪：和 record.py 采集默认一致
GRIPPER_THRESH = 0.5
GRIPPER_FORCE = 40.0
GRIPPER_SPEED = 0.2

ACTION_HORIZON = 50           # 模型一次返回的步数


class TrackerDriver(threading.Thread):
    """后台线程：持续 tick JointImpedanceTracker 并把机械臂驱向共享 target。

    推理(client.infer)会阻塞主线程几百毫秒，期间必须有人继续 tick，否则 1kHz
    控制命令断供会报 communication error。主线程只调 set_target 更新目标。
    """

    def __init__(self, robot, target_init: np.ndarray):
        super().__init__(daemon=True)
        self.robot = robot
        self._target = np.asarray(target_init, dtype=float).copy()
        self._lock = threading.Lock()
        self._running = True
        self.alive = True         # tracker 是否仍在正常运行
        self.error: Exception | None = None

    def set_target(self, target: np.ndarray) -> None:
        with self._lock:
            self._target = np.asarray(target, dtype=float).copy()

    def run(self) -> None:
        try:
            with JointImpedanceTracker(
                self.robot, stiffness=STIFFNESS, damping=DAMPING,
                lower_joint_limits=FR3_LOWER, upper_joint_limits=FR3_UPPER,
                period=0.002,
            ) as tracker:
                tracker.set_target(self._target)
                while self._running:
                    if not tracker.tick():
                        break
                    with self._lock:
                        t = self._target
                    tracker.set_target(t)
        except Exception as exc:                 # noqa: BLE001 碰撞/控制异常都在此兜住
            self.error = exc
        finally:
            self.alive = False

    def stop(self) -> None:
        self._running = False


class Deployer:
    def __init__(self, robot, gripper, cams, client, args):
        self.robot = robot
        self.gripper = gripper
        self.cams = cams              # {"base": CameraD435i, "wrist": CameraD435i}
        self.client = client
        self.args = args

        self.driver: TrackerDriver | None = None
        self.grip_future = None
        self.last_want_close: bool | None = None

    # ---------- 归位 ----------

    def home(self) -> None:
        """归位到 ready（tracker 未启动时用 JointMotion，之后再交给 tracker）。"""
        q = np.asarray(self.robot.current_joint_state.position)
        if np.rad2deg(np.max(np.abs(READY - q))) > 1.0:
            print("  FR3 归位到 ready ...")
            self.robot.move(JointMotion(READY.tolist(), relative_dynamics_factor=0.05))
        print("  ✅ 已在 ready")

    # ---------- 组观测 ----------

    def build_obs(self) -> dict:
        q = np.asarray(self.robot.current_joint_state.position, dtype=np.float32)  # 7 rad
        width = np.float32(self.gripper.width if self.gripper else 0.0)            # 米制
        state = np.concatenate([q, [width]]).astype(np.float32)                   # (8,)

        base_bgr, _ = self.cams["base"].latest()
        wrist_bgr, _ = self.cams["wrist"].latest()
        # 相机给 BGR，训练用 RGB → 翻通道，并转成连续数组（msgpack 需要）
        base_rgb = np.ascontiguousarray(base_bgr[..., ::-1])
        wrist_rgb = np.ascontiguousarray(wrist_bgr[..., ::-1])
        return {
            "observation/state": state,
            "observation/image": base_rgb,
            "observation/wrist_image": wrist_rgb,
            "prompt": self.args.prompt,
        }

    # ---------- 夹爪（二值，和 record.py 一致）----------

    def _update_gripper(self, cmd: float) -> None:
        if self.gripper is None:
            return
        want_close = cmd > GRIPPER_THRESH            # a[7]=1 闭合抓取，0 张开
        ready = self.grip_future is None or self.grip_future.wait(0)
        if ready and want_close != self.last_want_close:
            if want_close:
                self.grip_future = self.gripper.grasp_async(
                    0.0, GRIPPER_SPEED, GRIPPER_FORCE, epsilon_outer=1.0)
            else:
                self.grip_future = self.gripper.open_async(GRIPPER_SPEED)
            self.last_want_close = want_close

    # ---------- 主循环 ----------

    def run(self) -> None:
        args = self.args
        print("\n" + "=" * 62)
        print("pi05 部署就绪")
        print(f"  服务器: {args.host}:{args.port}   任务: {args.prompt}")
        print(f"  每批执行 {args.exec_steps}/{ACTION_HORIZON} 步 @ {FPS}Hz "
              f"(~{args.exec_steps / FPS:.1f}s/批)")
        print("=" * 62)

        self.home()
        input("\n按回车开始自主执行（人手放急停旁；运行中按 q 退出）> ")

        # 启动后台控制线程（从 ready 接管）
        self.driver = TrackerDriver(self.robot, READY)
        self.driver.start()
        time.sleep(0.3)                              # 等 tracker 起来
        if not self.driver.alive:
            raise RuntimeError(f"tracker 启动失败: {self.driver.error}")

        cap_dt = 1.0 / FPS
        n_batch = 0
        with KeyPoller() as keys:
            while True:
                # --- 推理一批 ---
                t_infer = time.perf_counter()
                obs = self.build_obs()
                actions = np.asarray(self.client.infer(obs)["actions"])   # (50, 8)
                infer_ms = (time.perf_counter() - t_infer) * 1000
                n_batch += 1

                # --- 按 15Hz 回放前 exec_steps 步 ---
                batch_t0 = time.perf_counter()
                for step in range(args.exec_steps):
                    a = actions[step]
                    self.driver.set_target(a[:7])
                    self._update_gripper(float(a[7]))

                    # 卡节拍到该步结束
                    step_end = batch_t0 + (step + 1) * cap_dt
                    while time.perf_counter() < step_end:
                        if keys.poll() == "q":
                            print("\n收到退出。")
                            return
                        if not self.driver.alive:
                            print(f"\n❌ tracker 停止（可能碰撞）: {self.driver.error}")
                            return
                        time.sleep(0.002)

                print(f"\r批 {n_batch}  infer {infer_ms:.0f}ms  "
                      f"已执行 {args.exec_steps} 步   ", end="", flush=True)


def main():
    ap = ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="localhost", help="策略服务器地址（SSH 转发后为 localhost）")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--fr3-host", default="192.168.3.1")
    ap.add_argument("--prompt", default="put the cube on the plate")
    ap.add_argument("--exec-steps", type=int, default=ACTION_HORIZON,
                    help=f"每批执行前几步就重推理（≤{ACTION_HORIZON}，默认全部 50 = 纯开环）")
    ap.add_argument("--wrist-serial", default=WRIST_SERIAL)
    ap.add_argument("--base-serial", default=BASE_SERIAL)
    ap.add_argument("--no-gripper", action="store_true")
    args = ap.parse_args()
    args.exec_steps = max(1, min(args.exec_steps, ACTION_HORIZON))

    # 相机（严格照抄采集参数 record.py）
    online = {sn for _, sn in list_devices()}
    missing = [sn for sn in (args.wrist_serial, args.base_serial) if sn not in online]
    if missing:
        raise SystemExit(f"❌ 相机不在线: {missing}，当前 RealSense: {sorted(online)}")
    cam_opts = dict(exposure=CAM_EXPOSURE, gain=CAM_GAIN, white_balance=CAM_WB)
    cams = {
        "wrist": CameraD435i(CAM_W, CAM_H, WRIST_FPS, auto_exposure=True,
                             serial=args.wrist_serial, **cam_opts),
        "base": CameraD435i(CAM_W, CAM_H, BASE_FPS, auto_exposure=True,
                            serial=args.base_serial, rotate=BASE_ROTATE, **cam_opts),
    }
    started = []
    for name, cam in cams.items():
        try:
            cam.start()
            cam.wait_first_frame()
        except RuntimeError as exc:
            # 一台起不来时先停掉已启动的，否则后台线程还在跑 pipeline，
            # 进程会 "terminate called without an active exception" 核心转储。
            for c in started:
                c.stop()
            raise SystemExit(
                f"❌ 相机 {name}（serial={cam.serial}）启动失败: {exc}\n"
                f"   D435i 走 USB2 时会间歇性不出帧，重插这台的 USB 线后重试；"
                f"   单独验证: python snap_base.py --serial {cam.serial}")
        started.append(cam)
    print(f"✅ 相机已连接 wrist={args.wrist_serial} base={args.base_serial}  exposure={CAM_EXPOSURE} wb={CAM_WB}")

    # 机械臂 + 夹爪
    robot = Robot(args.fr3_host)
    robot.recover_from_errors()
    robot.set_collision_behavior(
        torque_thresholds=COLLISION_TORQUE, force_thresholds=COLLISION_FORCE)
    robot.joint_acceleration_limit.set([5.0] * 7)
    gripper = None
    if not args.no_gripper:
        gripper = Gripper(args.fr3_host)
        if gripper.max_width <= 0.0:
            print("⚠️ 夹爪未标定，先跑 gripper_init.py。禁用夹爪。")
            gripper = None
    print("✅ FR3 + 夹爪已连接")

    # 策略客户端（连不上会阻塞重试）
    print(f"连接策略服务器 {args.host}:{args.port} ...")
    client = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    print(f"✅ 已连接，服务器 metadata: {client.get_server_metadata()}")

    dep = Deployer(robot, gripper, cams, client, args)
    try:
        dep.run()
    except KeyboardInterrupt:
        print("\n\n收到 Ctrl+C")
    except franky.ControlException as exc:
        print(f"\n❌ 控制错误: {exc}")
    finally:
        if dep.driver is not None:
            dep.driver.stop()
            dep.driver.join(timeout=1.0)
        for cam in cams.values():
            cam.stop()
        print("\n已停止。FR3 归位请跑 home_robot.py")


if __name__ == "__main__":
    try:
        main()
    except franky.NetworkException as exc:
        print(f"❌ FR3 网络错误: {exc}（检查 FCI/网口/调频）")
