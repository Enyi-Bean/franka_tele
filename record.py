#!/usr/bin/env python3
"""阶段二：遥操作 + 数据采集，存成 LeRobot v2.1 格式。

在 teleop.py 的遥操作基础上加录制：以 15Hz 为节拍，每拍打包一帧
    state   = [FR3 当前 7 关节角(绝对), 夹爪实测宽度]
    action  = [遥操作目标 7 关节角(绝对), 夹爪 0/1]
    images  = wrist(D435i) + base(D405) + 触觉两指各 Rectify/Difference，共 6 路
    task    = 固定语言指令
存进 LeRobotV21Writer。

复用 teleop.py 的 UArmReader / Mapper / KeyPoller / 控制参数，控制逻辑保持一致。

按键：
    Enter  第一次：FR3 归位到 ready；第二次：开始遥操作并录制
    s      保存当前 episode
    d      丢弃当前 episode（重录）
    空格   离合（暂停/恢复跟随，录制时也可用）
    q      退出（会先 finalize 数据集）

用法：
    python record.py --task "put the cube on the plate" --out data/cube_on_plate_tactile
    python record.py ... --no-tactile                       # 纯视觉（wrist + base）
    python record.py ... --tactile left=OG001273            # 只用一个触觉
    python record.py ... --tactile left=OG001273 right=OG001275   # 默认就是这两个
"""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import time

import numpy as np

import franky
from franky import Gripper, JointImpedanceTracker, JointMotion, Robot

# 复用 teleop.py 里已验证的组件与常量
from teleop import (
    UArmReader, Mapper, KeyPoller, wait_for_leader,
    FR3_LOWER, FR3_UPPER, STIFFNESS, DAMPING, MAX_JOINT_VEL,
    LEADER_TIMEOUT, COLLISION_TORQUE, COLLISION_FORCE,
    N_JOINTS, GRIPPER_ID, PORT, BAUD,
)
from camera_d435i import CameraD435i
from lerobot_writer import LeRobotV21Writer
from tactile_xense import TactileXense, TACTILE_H, TACTILE_W

FPS = 15                       # 采集节拍 = 主臂控制频率
STALE_SEC = 0.2                # 数据源超过这个秒数没出新帧，判为掉线并中止 episode
CAM_W, CAM_H = 640, 480
# 两台都是 D435i（2026-09-19 起；此前第三人称是 D405）。两台同型号时不能按型号找，
# 必须按序列号锁定，否则 wrist/base 会静默对调。
WRIST_SERIAL = "233622077251"  # 腕部，2026-10-02 按实际画面确认
BASE_SERIAL = "135122077750"   # 第三人称，2026-10-02 按实际画面确认
WRIST_FPS = 15                 # 2026-09-19 从 30 降到 15：采集节拍就是 15Hz，30 只在 USB3 上有余量；
                               # 现在两台 D435i 挤在一个 USB2 拓展坞上，30fps 白占一半带宽
BASE_FPS = 15                  # 拓展坞 USB2，保守 15fps（采集节拍就是 15Hz，无损失）
BASE_ROTATE = 0                # 相机已物理摆正，无需旋转（若倒装可设 180）
# 曝光锁 220、白平衡锁 3800（2026-09-19 实测：自动白平衡在本实验室光源下偏粉，
# 3800 中性；两台 D435i 同一光源，用同一组值）。部署端必须照抄。
CAM_EXPOSURE = 220             # 固定曝光（D435i 单位≈0.1ms）。2026-09-19 实测：自动曝光会把
                               # 黑桌面上的白色插线板推过曝、插孔看不清；220 插孔清楚且整体亮，240 开始泛白
CAM_GAIN = 16                  # 硬件下限
CAM_WB = 3800
# 触觉传感器：夹爪两指各一个。名字进数据集（observation.images.tactile_left 等），
# left 插原拓展坞，right 插 Type-C 拓展坞。
TACTILE_DEFAULT = ["left=OG001273", "right=OG001275"]   # 2026-09-21 换机：原 OG001270/OG001268 损坏
TACTILE_PREFIX = "tactile_"
STATE_NAMES = [f"joint_{i}" for i in range(7)] + ["gripper"]
ACTION_NAMES = [f"joint_{i}" for i in range(7)] + ["gripper"]


class Recorder:
    """把遥操作循环 + 采集节拍 + episode 管理封装起来。"""

    def __init__(self, robot, gripper, reader, sources, mapper, writer, args,
                 pending_refs: dict | None = None):
        self.robot = robot
        # 触觉基线图（{图像路名: Rectify}）。⭐ 不在启动时落盘，而是等本次第一条
        # episode 保存成功再写 —— 否则每次启动（哪怕连机械臂就失败）都会往
        # meta/tactile_ref/ 堆一对 PNG + 一行 index（2026-09-19 一晚堆了 9 对）。
        self.pending_refs = pending_refs or {}
        self.gripper = gripper
        self.reader = reader
        # 「源」和「图像路」是两个概念，不能合并成一个字典：
        #   源   = 设备，start/stop/帧数/掉线判断按它来（触觉是 1 个源）
        #   图像路 = 数据集里的 observation.images.*（触觉出 2 路）
        self.sources = sources                      # [CameraD435i×2, TactileXense×N]
        self.image_shapes = {}                      # {图像路名: (H,W,3)}
        for s in sources:
            self.image_shapes.update(s.image_shapes)
        self.mapper = mapper
        self.writer = writer
        self.args = args

        self.recording = False
        self.armed = False          # 已归位就位、等回车开始录制的中间态
        # 启动后待机（不跟随）：零点要等按 r 归位时才采。
        # 未对齐前跟随会拿 None 零点做映射而崩溃，也不安全。
        self.engaged = False
        self.last_target = None
        self.last_grip_ratio = -1.0
        self.grip_future = None
        self.n_saved = 0

    # ---------- 归位 ----------

    def home(self) -> None:
        """FR3 归位到 ready（每条 episode 起点一致），并重新采主臂零点。

        注意：此时阻抗跟踪器(tracker)正在运行，不能再调 robot.move(JointMotion)
        —— franky 不允许运行中切换运动类型，会抛 InvalidMotionTypeException。
        改用正在跑的 tracker，把目标从当前位姿平滑插值到 ready。
        """
        ready = self.mapper.franka_ready_ref
        tracker = self.tracker
        start = np.asarray(self.robot.current_joint_state.position)
        if np.rad2deg(np.max(np.abs(ready - start))) > 1.0:
            print("  FR3 归位到 ready ...")
            T = 3.0                               # 归位耗时（秒），越大越慢越稳
            t0 = time.perf_counter()
            while tracker.tick():
                elapsed = time.perf_counter() - t0
                a = min(1.0, elapsed / T)
                tracker.set_target(start + a * (ready - start))
                if a >= 1.0 and elapsed > T + 0.5:  # 到点后再稳定 0.5s
                    break
        # 锚点 = 实际到位姿态；零点 = 此刻主臂读数
        self.mapper.franka_ready = np.asarray(self.robot.current_joint_state.position)
        vals = wait_for_leader(self.reader)
        self.mapper.set_zero(vals)
        self.last_target = None

    # ---------- 采集一帧 ----------

    def _build_frame(self, target: np.ndarray, grip_binary: float):
        """打包当前时刻的一帧。返回 (state, action, images, stale)。

        stale = [(源名, age), ...] 里超时的那些。**必须用 age 判掉线，不能用
        `is not None`** —— latest() 首帧之后永远返回上一帧而非 None，
        用 None 判断的话，设备掉线会把一串重复死帧静静录进数据集。
        """
        q = np.asarray(self.robot.current_joint_state.position, dtype=np.float32)
        width = np.float32(self.gripper.width if self.gripper else 0.0)
        state = np.concatenate([q, [width]]).astype(np.float32)          # 观测：实测
        action = np.concatenate([target, [grip_binary]]).astype(np.float32)  # 命令

        images, stale = {}, []
        for s in self.sources:
            imgs, age = s.latest_images()
            if imgs is None or age > STALE_SEC:
                stale.append((s.name, age))
            if imgs is not None:
                images.update(imgs)
        return state, action, images, stale

    # ---------- 主循环 ----------

    def run(self) -> None:
        args = self.args
        damping = (np.array(DAMPING) * args.damping_scale).tolist()

        print("\n" + "=" * 62)
        print("采集已就绪")
        print(f"  任务: {args.task}")
        print("  [Enter]=归位到 home  →  再按 [Enter]=开始遥操作并录制")
        print("  s=保存  d=丢弃重录  空格=离合  q=退出")
        print("=" * 62 + "\n")
        print(">>> 按 [Enter] 让机械臂归位到 home ...\n")

        with KeyPoller() as keys, JointImpedanceTracker(
            self.robot, stiffness=STIFFNESS, damping=damping,
            lower_joint_limits=FR3_LOWER, upper_joint_limits=FR3_UPPER,
            period=0.002,
        ) as tracker:
            tracker.set_target(np.asarray(self.robot.current_joint_state.position))
            self.tracker = tracker

            t_prev = time.perf_counter()
            t_capture = 0.0
            t_report = 0.0
            n_timeout = 0
            cap_dt = 1.0 / FPS

            while tracker.tick():
                # --- 按键 ---
                k = keys.poll()
                if k == "q":
                    if self.recording:
                        print("\n先按 s 保存或 d 丢弃当前录制，再退出。")
                    else:
                        print("\n收到退出。")
                        return
                elif k in ("\r", "\n") and not self.recording and not self.armed:
                    self._arm()                     # 第一次回车：仅归位，暂不跟随
                elif k in ("\r", "\n") and self.armed and not self.recording:
                    self._begin_recording()         # 第二次回车：开始遥操作并录制
                elif k == "s" and self.recording:
                    self._save_recording()
                elif k == "d" and (self.recording or self.armed):
                    self._discard_recording()       # armed 或录制中都可取消
                elif k == " ":
                    self._toggle_clutch()

                # --- 读主臂 ---
                vals, age = self.reader.latest()
                if vals is None or age > LEADER_TIMEOUT:
                    n_timeout += 1
                    if n_timeout > 500:
                        print("\n❌ 主臂持续掉线，退出。")
                        return
                    continue
                n_timeout = 0

                if not self.engaged:
                    continue

                # --- 映射 + 限速（与 teleop 一致）---
                vals = self.mapper.apply_deadband(vals, args.deadband)
                raw_target = self.mapper.to_franka(vals)
                now = time.perf_counter()
                dt = max(now - t_prev, 1e-4)
                t_prev = now
                if self.last_target is None:
                    self.last_target = np.asarray(self.robot.current_joint_state.position)
                if args.smooth > 0:
                    a = min(1.0, dt / (args.smooth / 1000.0))
                    target = self.last_target + a * (raw_target - self.last_target)
                else:
                    target = raw_target
                step = MAX_JOINT_VEL * dt * args.speed
                target = self.last_target + np.clip(target - self.last_target, -step, step)
                self.last_target = target
                tracker.set_target(target)

                # --- 夹爪（二值）---
                grip_binary = self._update_gripper(vals)

                # --- 采集节拍：每 1/FPS 秒抓一帧 ---
                if self.recording and now - t_capture >= cap_dt:
                    t_capture = now
                    state, action, images, stale = self._build_frame(target, grip_binary)
                    if stale:
                        # 掉线：中止并丢弃整条。缺帧的 episode 对 VTLA 没用，
                        # 与其录一串重复死帧事后难以发现，不如当场重录。
                        detail = "  ".join(f"{n}({a * 1000:.0f}ms)" for n, a in stale)
                        print(f"\n\n❌ 数据源超时: {detail}")
                        print("   本条已丢弃。检查 USB 连接后按 [Enter] 重录。\n")
                        self.writer.discard_episode()
                        self.recording = False
                        self.engaged = False
                        continue
                    if len(images) != len(self.image_shapes):
                        print(f"\n⚠️ 图像路数不对（得到 {len(images)}，"
                              f"期望 {len(self.image_shapes)}），跳过这一帧")
                        continue
                    self.writer.add_frame(
                        state=state, action=action,
                        images=images, task=args.task,
                    )

                # --- 状态行 ---
                if now - t_report >= 0.5:
                    t_report = now
                    self._print_status()

    # ---------- 子操作 ----------

    def _arm(self):
        """第一次回车：只归位到 home，机械臂停住不跟随（等第二次回车）。"""
        print("\n\n>>> 归位到 home ...")
        self.home()                 # 归位 ready + 采主臂零点
        self.engaged = False        # 归位后不跟随，防止手一动机械臂就跟着走
        self.armed = True
        print(">>> ✅ 已归位。摆好物体后按 [Enter] 开始遥操作并录制（d=取消）\n")

    def _begin_recording(self):
        """第二次回车：以当前主臂读数重新对零，开始跟随并写数据。"""
        # 归位到此刻主臂可能已被挪动，重新对零避免起录瞬间跳变
        vals, _ = self.reader.latest()
        if vals is not None:
            self.mapper.set_zero(vals)
            self.mapper.franka_ready = np.asarray(self.robot.current_joint_state.position)
            self.last_target = None
        self.armed = False
        self.engaged = True         # 现在才开始跟随
        self.writer.start_episode()
        self.recording = True
        print(">>> ● 录制中（s=保存  d=丢弃）\n")

    def _save_recording(self):
        n = self.writer.n_buffered
        if n < 3:
            print(f"\n⚠️ 只有 {n} 帧，太短，自动丢弃。")
            self.writer.discard_episode()
        else:
            ep = self.writer.save_episode()
            self.n_saved += 1
            if self.pending_refs:
                out = Path(self.args.out)
                names = {k: save_tactile_ref(out, k, img) for k, img in self.pending_refs.items()}
                append_ref_index(out, ep, names)
                self.pending_refs = {}
                print(f"\n>>> 触觉基线已存档（从 episode {ep} 起生效）: "
                      + "  ".join(names.values()))
            print(f"\n>>> ✅ 已保存 episode {ep}（{n} 帧）。本次已采 {self.n_saved} 条。")
            print(">>> 按 [Enter] 归位录下一条，q 退出\n")
        self.recording = False
        self.engaged = False        # 回到待机（FR3 停在原地），按 Enter 再录下一条

    def _discard_recording(self):
        if self.recording:
            self.writer.discard_episode()   # armed 阶段还没 start_episode，不能调
        self.recording = False
        self.armed = False
        self.engaged = False        # 回到待机，按 Enter 重来
        print("\n>>> 🗑️ 已取消。按 [Enter] 归位重来\n")

    def _toggle_clutch(self):
        self.engaged = not self.engaged
        if self.engaged:
            vals, _ = self.reader.latest()
            if vals is not None:
                self.mapper.set_zero(vals)
                self.mapper.franka_ready = np.asarray(self.robot.current_joint_state.position)
                self.last_target = None
        print(f"\n>>> {'恢复跟随' if self.engaged else '已离合（暂停）'}\n")

    def _update_gripper(self, vals) -> float:
        """返回夹爪的二值 action（0/1），同时下发控制指令。"""
        if self.gripper is None:
            return 0.0
        ratio = self.mapper.to_gripper_ratio(vals[GRIPPER_ID])
        want_close = ratio < self.args.gripper_thresh
        ready = self.grip_future is None or self.grip_future.wait(0)
        if ready and want_close != (self.last_grip_ratio < self.args.gripper_thresh):
            if want_close:
                self.grip_future = self.gripper.grasp_async(
                    0.0, self.args.gripper_speed, self.args.gripper_force,
                    epsilon_outer=1.0)
            else:
                self.grip_future = self.gripper.open_async(self.args.gripper_speed)
            self.last_grip_ratio = ratio
        return 1.0 if want_close else 0.0

    def _print_status(self):
        n_read, n_fail = self.reader.stats
        state = "● 录制" if self.recording else ("○ 离合" if not self.engaged else "○ 待机")
        buf = self.writer.n_buffered if self.recording else 0
        # 按「源」打印帧数（每个触觉是 1 个源出 2 路图，不重复打印）
        srcs = "  ".join(f"{s.name}:{s.n_frames}" for s in self.sources)
        print(f"\r{state}   本条 {buf} 帧   已采 {self.n_saved} 条   "
              f"{srcs}   ", end="", flush=True)


def save_tactile_ref(out: Path, name: str, ref_img: np.ndarray) -> str:
    """把某个触觉传感器的基线帧存进 meta/tactile_ref/，返回文件名。

    为什么必须存：Difference 的参考帧只活在 SDK 内存里，calibrateSensor()
    一调就没了 —— **这是整个采集方案里唯一不可逆的东西**。存下来才能在跨会话
    基线漂移时，用统一基线从 Rectify 批量重算 Difference。

    ⚠️ meta/tactile_ref/ 是非标准文件，convert_dataset_v21_to_v30.py 转换或
    push 到 HuggingFace hub 时不一定会被搬运，转换/上传后手动确认它还在。
    """
    import cv2

    d = out / "meta" / "tactile_ref"
    d.mkdir(parents=True, exist_ok=True)
    fname = f"ref_{name}_{time.strftime('%Y%m%d_%H%M%S')}.png"
    cv2.imwrite(str(d / fname), ref_img)
    return fname


def append_ref_index(out: Path, from_episode: int, refs: dict[str, str]) -> None:
    """记录「从第几条 episode 起，每个触觉传感器用的是哪张基线」。

    refs = {"tactile_left": "ref_tactile_left_xxx.png", "tactile_right": ...}

    跨会话基线会漂（温度、弹性体蠕变），事后想用统一基线重算 Difference 时，
    必须知道每条 episode 当时用的是哪张参考帧。
    """
    d = out / "meta" / "tactile_ref"
    d.mkdir(parents=True, exist_ok=True)
    import json

    with (d / "index.jsonl").open("a") as f:
        f.write(json.dumps({"from_episode": from_episode, "refs": refs}) + "\n")


def check_image_keys(out: Path, image_shapes: dict) -> None:
    """续采前核对图像路集合，不一致就拒绝。

    旧 guard 只数 observation.images.base 的 mp4 个数，拦不住「带触觉往
    纯视觉数据集里续采」。而那种混采失败得非常安静：
      - finalize() 用当前 cameras 重写 info.json，旧 episode 被声明成有触觉
      - total_videos 算错
      - lerobot_writer 的 stats 聚合取 per_ep[0].keys()，新增的触觉 stats 被静默丢弃
    已训出模型的数据集是资产，不能靠"记得换目录"的纪律来保护。
    """
    info_path = out / "meta" / "info.json"
    if not info_path.exists():
        return                      # 新数据集，随便写
    import json

    feats = json.loads(info_path.read_text()).get("features", {})
    prev = {k[len("observation.images."):] for k in feats
            if k.startswith("observation.images.")}
    now = set(image_shapes)
    if prev != now:
        raise SystemExit(
            f"❌ {out} 已有数据的图像路和本次不一致，拒绝混采：\n"
            f"   已有: {sorted(prev)}\n"
            f"   本次: {sorted(now)}\n"
            f"   缺少: {sorted(prev - now) or '无'}   多出: {sorted(now - prev) or '无'}\n"
            f"   → 请换一个新的 --out 目录。混采会覆盖 info.json、算错 total_videos，"
            f"并静默丢掉新增路的 stats。"
        )


def main():
    ap = ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True, help="语言指令，如 'put the cube on the plate'")
    ap.add_argument("--out", required=True, help="数据集输出目录")
    ap.add_argument("--host", default="192.168.3.1")
    ap.add_argument("--port", default=PORT)
    ap.add_argument("--wait", type=float, default=0.008)
    ap.add_argument("--speed", type=float, default=0.8)
    ap.add_argument("--smooth", type=float, default=0.0)
    ap.add_argument("--deadband", type=float, default=0.0)
    ap.add_argument("--damping-scale", type=float, default=1.0)
    ap.add_argument("--gripper-thresh", type=float, default=0.5)
    ap.add_argument("--gripper-force", type=float, default=40.0)
    ap.add_argument("--gripper-speed", type=float, default=0.2)
    ap.add_argument("--keep-torque", action="store_true")
    ap.add_argument("--wrist-serial", default=WRIST_SERIAL, help="腕部相机序列号")
    ap.add_argument("--base-serial", default=BASE_SERIAL, help="第三人称相机序列号")
    ap.add_argument("--no-base", action="store_true", help="只用腕部相机（调试/临时缺 D405）")
    ap.add_argument("--no-tactile", action="store_true",
                    help="不接触觉传感器（退回纯视觉采集）")
    ap.add_argument("--tactile", nargs="+", default=TACTILE_DEFAULT, metavar="NAME=SERIAL",
                    help="触觉传感器列表，如 left=OG001273 right=OG001275。"
                         f"图像路名为 {TACTILE_PREFIX}<NAME> 和 {TACTILE_PREFIX}<NAME>_diff。"
                         f"默认 {' '.join(TACTILE_DEFAULT)}")
    ap.add_argument("--no-calib", action="store_true",
                    help="跳过触觉基线校准。⚠️ 只在明确知道基线还准时用 —— "
                         "Difference 完全建立在基线上")
    ap.add_argument("--base-rotate", type=int, default=BASE_ROTATE,
                    choices=[0, 90, 180, 270], help="第三人称相机旋正角度（已摆正默认 0）")
    ap.add_argument("--exposure", type=float, default=CAM_EXPOSURE,
                    help="两台相机固定曝光（D435i 单位≈0.1ms），默认 220；0=自动曝光")
    ap.add_argument("--gain", type=float, default=CAM_GAIN, help="两台相机增益（仅固定曝光时生效），默认 16")
    ap.add_argument("--wb", type=float, default=CAM_WB, help="两台相机白平衡色温(2800~6500)，默认锁 3800")
    args = ap.parse_args()

    calib = Path(__file__).parent / "uarm_calib.json"
    mapper = Mapper.from_calib(calib)
    # 记录 ready 参考（Mapper.franka_ready 会在运行中被改写作锚点，这里留一份原始的）
    mapper.franka_ready_ref = mapper.franka_ready.copy()
    print(f"✅ 标定载入 scale={mapper.scale.astype(int).tolist()}")

    # 主臂
    reader = UArmReader(args.port, BAUD, wait=args.wait)
    if reader.read_once() is None:
        raise SystemExit("❌ 主臂无响应")
    if not args.keep_torque:
        reader.release_torque()
    reader.start()
    print("✅ 主臂已连接")

    # 相机（两台同型号，必须按序列号锁定）
    cam_opts = dict(exposure=(args.exposure or None), gain=args.gain, white_balance=args.wb)
    sources = [
        CameraD435i(CAM_W, CAM_H, WRIST_FPS, auto_exposure=True, serial=args.wrist_serial,
                    name="wrist", **cam_opts),
    ]
    if not args.no_base:
        sources.append(CameraD435i(CAM_W, CAM_H, BASE_FPS, auto_exposure=True,
                                   serial=args.base_serial, name="base",
                                   rotate=args.base_rotate, **cam_opts))
    # 触觉源先构造出来（构造函数不碰硬件），好在启动任何设备之前算出完整图像路集合
    tactiles: list[TactileXense] = []
    if not args.no_tactile:
        for spec in args.tactile:
            if "=" not in spec:
                raise SystemExit(f"❌ --tactile 格式应为 NAME=SERIAL，收到 {spec!r}")
            name, serial = spec.split("=", 1)
            tactiles.append(TactileXense(serial.strip(), name=TACTILE_PREFIX + name.strip()))
        if len({t.name for t in tactiles}) != len(tactiles):
            raise SystemExit("❌ --tactile 里的 NAME 重复了")
        if len({t.serial for t in tactiles}) != len(tactiles):
            raise SystemExit("❌ --tactile 里的序列号重复了")
    sources.extend(tactiles)

    # ⭐ guard 必须在**任何设备启动、任何文件写盘之前**跑：
    #    image_shapes 只读对象属性、不碰硬件，所以能提到最前。
    #    否则往路数不符的旧数据集续采时，基线 PNG 会先写进那个目录再被拒绝，
    #    在已训练好的数据集里留下垃圾文件。
    image_shapes = {}
    for s in sources:
        image_shapes.update(s.image_shapes)
    check_image_keys(Path(args.out), image_shapes)

    started = []
    for cam in sources:
        if isinstance(cam, TactileXense):
            continue                      # 触觉要先校准再 start，下面单独处理
        try:
            cam.start()
            cam.wait_first_frame()
        except RuntimeError as exc:
            # 某台相机起不来：先把已启动的停掉再退出。否则异常抛出时后台线程
            # 还在跑 pipeline，进程会 "terminate called without an active exception"
            # 直接核心转储（2026-09-19 遇到过）。
            for c in started:
                c.stop()
            raise SystemExit(
                f"❌ 相机 {cam.name}（serial={cam.serial}）启动失败: {exc}\n"
                f"   D435i 走 USB2 时会间歇性不出帧，重插线 / 换 USB3 口后重试；"
                f"   单独验证: python snap_base.py --serial {cam.serial}")
        started.append(cam)
        fps = WRIST_FPS if cam.name == "wrist" else BASE_FPS
        mode = f"exposure={args.exposure:.0f}" if args.exposure else "自动曝光"
        print(f"✅ 相机 {cam.name} 已连接 {CAM_W}x{CAM_H}@{fps}  serial={cam.serial}  "
              f"{mode}  wb={args.wb}")

    # ---- 触觉传感器（每个源 → <name> / <name>_diff 两路图）----
    ref_imgs: dict[str, np.ndarray] = {}      # 校准图先留内存，首条保存成功再落盘
    if tactiles:
        for t in tactiles:
            t.open()
        # ⚠️ calibrateSensor() 必须在起后台线程之前调（它重设参考帧，
        #    与线程里的 selectSensorInfo 并发时行为未知）。所有传感器先一起校准，
        #    再一起起线程。
        if not args.no_calib:
            print("\n" + "=" * 62)
            print("触觉基线校准：请确认**所有传感器表面完全无接触**（夹爪张开、别碰）")
            print("   " + "  ".join(f"{t.name}={t.serial}" for t in tactiles))
            print("=" * 62)
            input(">>> 确认无接触后按 [Enter] 校准 ... ")
            for t in tactiles:
                ref_imgs[t.name] = t.calibrate()
                print(f"✅ {t.name} 基线已校准（保存第一条 episode 时存档）")
        else:
            print("⚠️ 跳过触觉校准（--no-calib）：Difference 用的是上次的基线")
        for t in tactiles:
            t.start()
            t.wait_first_frame()
            print(f"✅ 触觉 {t.name} ({t.serial}) 已连接 "
                  f"{TACTILE_W}x{TACTILE_H}（Rectify + Difference 两路）")

    # 机械臂
    robot = Robot(args.host)
    robot.recover_from_errors()
    robot.set_collision_behavior(
        torque_thresholds=COLLISION_TORQUE, force_thresholds=COLLISION_FORCE)
    robot.joint_acceleration_limit.set([5.0] * 7)
    gripper = Gripper(args.host)
    if gripper.max_width <= 0.0:
        print("⚠️ 夹爪未标定，先跑 gripper_init.py。禁用夹爪。")
        gripper = None
    print("✅ FR3 + 夹爪已连接")

    # writer
    writer = LeRobotV21Writer(
        args.out, fps=FPS, state_dim=8, action_dim=8,
        cameras=image_shapes, robot_type="franka",
        state_names=STATE_NAMES, action_names=ACTION_NAMES)
    print(f"✅ 数据集输出: {args.out}（已有 {writer._episode_index} 条，续采）"
          f"  图像路: {list(image_shapes)}")
    rec = Recorder(robot, gripper, reader, sources, mapper, writer, args,
                   pending_refs=ref_imgs)
    try:
        rec.run()
    except KeyboardInterrupt:
        print("\n\n收到 Ctrl+C")
    except franky.ControlException as exc:
        print(f"\n❌ 控制错误: {exc}")
    finally:
        if rec.recording:
            print("退出前保存当前录制 ...")
            rec._save_recording()
        if writer._episode_index > 0:
            writer.finalize()           # 空数据集不写 info.json/stats.json
        reader.stop()
        for cam in sources:
            cam.stop()
        print(f"\n本次新采 {rec.n_saved} 条，数据集 {args.out} 共 {writer._episode_index} 条 episode。"
              + ("（本次没有写入任何轨迹）" if rec.n_saved == 0 else ""))
        print("FR3 归位请跑 home_robot.py")


if __name__ == "__main__":
    try:
        main()
    except franky.NetworkException as exc:
        print(f"❌ FR3 网络错误: {exc}（检查 FCI/网口/调频）")
