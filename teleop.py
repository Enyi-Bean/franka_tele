#!/usr/bin/env python3
"""阶段一：U-Arm 主臂遥操作 FR3。只做控制，不采集数据。

流程：
  ① FR3 归位到 ready（固定姿态，保证后续每条 episode 起点一致）
  ② 按回车 → 采集主臂当前读数为零点（主臂摆哪都行，FR3 不会跳）
  ③ 循环：读主臂 → 映射 → 限位/限速 → JointImpedanceTracker
  ④ 扳机控制夹爪

映射：
    franka_q[i] = (leader_deg[i] - leader_zero[i]) * scale[i] * π/180 + ready[i]

用法：
    python teleop.py --dry-run     # FR3 完全不动，只打印映射结果 ← 先跑这个
    python teleop.py               # 真正遥操作
    python teleop.py --no-gripper  # 不控制夹爪

运行中按键：
    空格  离合（暂停/恢复跟随）
    q     退出
"""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
import json
import re
import select
import sys
import termios
import threading
import time
import tty

import numpy as np
import serial

import franky
from franky import Gripper, JointImpedanceTracker, JointMotion, Robot

# ---------- 常量 ----------

PORT, BAUD = "/dev/ttyUSB0", 115200
PWM_MIN, PWM_MAX, ANGLE_RANGE = 500, 2500, 270.0
N_JOINTS, GRIPPER_ID = 7, 7

FR3_LOWER = np.array([-2.9007, -1.8361, -2.9007, -3.0770, -2.8763, 0.4398, -3.0508])
FR3_UPPER = np.array([2.9007, 1.8361, 2.9007, -0.1169, 2.8763, 4.6216, 3.0508])
LIMIT_MARGIN = np.deg2rad(3.0)  # 离限位留 3° 余量，别贴着边跑

# GELLO 用于「被动主臂 + FR3 从臂」场景的增益（仅为 Franka 官方遥操作的 40%）。
# 操作者感知不到接触力，从臂必须更柔顺才安全。
STIFFNESS = [240.0, 240.0, 240.0, 240.0, 100.0, 60.0, 20.0]
DAMPING = [20.0, 20.0, 20.0, 10.0, 10.0, 10.0, 5.0]

# 逐周期速度钳位：防止猛甩主臂导致从臂暴冲（思路来自 U-Arm 的 ARX 示例）。
MAX_JOINT_VEL = np.array([1.0, 1.0, 1.0, 1.0, 1.5, 1.5, 1.5])  # rad/s

LEADER_TIMEOUT = 0.5  # 主臂读数超过这个秒数没更新就视为掉线

# 碰撞阈值。取自 franka_ros2_teleop/config/fr3_teleop_config.yaml，
# 几乎顶到 FR3 的关节力矩上限 [87,87,87,87,12,12,12] Nm ——
# 遥操作和接触密集任务（插孔、擦拭）本来就会持续受力，
# 阈值设低了会因为机械臂自身的动态力矩就误触发 reflex（红灯）。
COLLISION_TORQUE = [85.0, 85.0, 85.0, 85.0, 11.0, 11.0, 11.0]
COLLISION_FORCE = [85.0, 85.0, 85.0, 85.0, 11.0, 11.0]


# ---------- 主臂读取（独立线程） ----------

class UArmReader:
    """后台线程持续读主臂，主控制循环只取最新值。

    串口读 8 个舵机要约 80ms（约 12Hz），而阻抗控制循环跑 500Hz。
    放在同一个循环里会把控制循环拖慢，所以必须解耦。
    """

    def __init__(self, port: str, baud: int, wait: float = 0.01):
        self._ser = serial.Serial(port, baud, timeout=0.05)
        self._wait = wait
        self._lock = threading.Lock()
        self._latest: np.ndarray | None = None
        self._stamp = 0.0
        self._running = False
        self._thread: threading.Thread | None = None
        self._n_read = 0
        self._n_fail = 0
        time.sleep(0.2)

    def _read_servo(self, sid: int) -> float | None:
        self._ser.reset_input_buffer()
        self._ser.write(f"#{sid:03d}PRAD!".encode("ascii"))
        time.sleep(self._wait)
        m = re.search(
            rf"#{sid:03d}P(\d{{4}})",
            self._ser.read_all().decode("ascii", errors="ignore"),
        )
        if not m:
            return None
        return (int(m.group(1)) - PWM_MIN) / (PWM_MAX - PWM_MIN) * ANGLE_RANGE

    def release_torque(self) -> None:
        """给所有舵机发 PULK，释放力矩。

        舵机上电默认是位置保持模式：电位器反馈有噪声 → 舵机以为自己偏了 →
        施力修正 → 过冲 → 再修正，形成持续的微幅振荡（伺服打摆）。
        这是【物理性】抖动，靠读数侧滤波只能掩盖，治本要把力矩放掉。
        释放后主臂变成纯被动的编码器支架，也更好操作。
        """
        for i in range(8):
            self._ser.write(f"#{i:03d}PULK!".encode("ascii"))
            time.sleep(0.02)
        self._ser.reset_input_buffer()

    def read_once(self) -> np.ndarray | None:
        """同步读一次全部 8 个舵机，任一读失败则返回 None。"""
        vals = [self._read_servo(i) for i in range(8)]
        if any(v is None for v in vals):
            return None
        return np.array(vals)

    def _loop(self) -> None:
        while self._running:
            vals = self.read_once()
            if vals is None:
                self._n_fail += 1
                continue
            with self._lock:
                self._latest = vals
                self._stamp = time.perf_counter()
                self._n_read += 1

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        self._ser.close()

    def latest(self) -> tuple[np.ndarray | None, float]:
        """返回 (最新读数, 距今秒数)。"""
        with self._lock:
            if self._latest is None:
                return None, float("inf")
            return self._latest.copy(), time.perf_counter() - self._stamp

    @property
    def stats(self) -> tuple[int, int]:
        return self._n_read, self._n_fail


# ---------- 映射 ----------

@dataclass
class Mapper:
    scale: np.ndarray          # ±1，每个关节的方向
    franka_ready: np.ndarray   # FR3 锚点姿态（rad）
    trigger_open: float        # 扳机松开时的读数（度）
    trigger_close: float       # 扳机扣紧时的读数（度）
    leader_zero: np.ndarray | None = None
    _held: np.ndarray | None = None   # 死区保持的上一组读数

    @classmethod
    def from_calib(cls, path: Path) -> Mapper:
        c = json.loads(path.read_text())
        scale = np.array(c["scale"], dtype=float)
        if np.any(scale == 0):
            missing = [i + 1 for i, s in enumerate(scale) if s == 0]
            raise SystemExit(f"❌ 关节 J{missing} 尚未标定，先跑 uarm_calibrate.py")
        g = c.get("gripper") or {}
        if "open_deg" not in g:
            raise SystemExit("❌ 夹爪未标定，先跑 uarm_calibrate.py --gripper-only")
        return cls(
            scale=scale,
            franka_ready=np.array(c["franka_init_rad"], dtype=float),
            trigger_open=g["open_deg"],
            trigger_close=g["close_deg"],
        )

    def set_zero(self, leader_deg: np.ndarray) -> None:
        self.leader_zero = leader_deg[:N_JOINTS].copy()

    def to_franka(self, leader_deg: np.ndarray) -> np.ndarray:
        """主臂读数（度）→ FR3 关节角（rad），已钳到限位内。"""
        delta = (leader_deg[:N_JOINTS] - self.leader_zero) * self.scale
        q = np.deg2rad(delta) + self.franka_ready
        return np.clip(q, FR3_LOWER + LIMIT_MARGIN, FR3_UPPER - LIMIT_MARGIN)

    def apply_deadband(self, leader_deg: np.ndarray, deadband: float) -> np.ndarray:
        """死区：变化小于阈值就沿用上一次的值，滤掉读数噪声。

        思路取自 U-Arm 的 servo_reader.py（那里用 step_size = 1°）。
        相比低通滤波，死区对「有意的大幅动作」零延迟代价 —— 只掐掉静止时的抖动。
        """
        if deadband <= 0:
            return leader_deg
        if self._held is None:
            self._held = leader_deg.copy()
            return self._held
        moved = np.abs(leader_deg - self._held) > deadband
        self._held[moved] = leader_deg[moved]
        return self._held

    def to_gripper_ratio(self, trigger_deg: float) -> float:
        """扳机读数 → 开合比例 0（全闭）~ 1（全开）。"""
        span = self.trigger_open - self.trigger_close
        if abs(span) < 1e-6:
            return 1.0
        return float(np.clip((trigger_deg - self.trigger_close) / span, 0.0, 1.0))


# ---------- 非阻塞键盘 ----------

class KeyPoller:
    def __enter__(self):
        self._fd = sys.stdin.fileno()
        self._old = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        return self

    def __exit__(self, *exc):
        termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)

    def poll(self) -> str | None:
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None


# ---------- 主流程 ----------

def wait_for_leader(reader: UArmReader, timeout: float = 5.0) -> np.ndarray:
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        vals, age = reader.latest()
        if vals is not None and age < 0.5:
            return vals
        time.sleep(0.05)
    raise SystemExit("❌ 读不到主臂数据，检查连线（检查串口路径、供电和舵机 ID）")


def run_dry(reader: UArmReader, mapper: Mapper) -> None:
    print("\n[dry-run] FR3 不会动。掰动主臂，观察映射出的目标角度是否合理。")
    print("按 q 退出。\n")
    hdr = "  " + "".join(f"{'J' + str(i + 1):>9}" for i in range(N_JOINTS))
    with KeyPoller() as keys:
        last = 0.0
        while True:
            if keys.poll() == "q":
                break
            vals, age = reader.latest()
            if vals is None or age > LEADER_TIMEOUT:
                time.sleep(0.05)
                continue
            now = time.perf_counter()
            if now - last < 0.3:
                time.sleep(0.02)
                continue
            last = now

            q = mapper.to_franka(vals)
            at_limit = (q <= FR3_LOWER + LIMIT_MARGIN + 1e-6) | (
                q >= FR3_UPPER - LIMIT_MARGIN - 1e-6
            )
            print(f"\n主臂(deg): {np.round(vals[:N_JOINTS], 1)}   "
                  f"扳机 {vals[GRIPPER_ID]:.1f}° → 开合 {mapper.to_gripper_ratio(vals[GRIPPER_ID]):.0%}")
            print(hdr)
            print("  " + "".join(f"{np.rad2deg(x):8.1f}°" for x in q))
            if at_limit.any():
                print("  ⚠️ 触及限位: " + ", ".join(
                    f"J{i + 1}" for i in np.where(at_limit)[0]))


def run_teleop(
    reader: UArmReader, mapper: Mapper, robot: Robot,
    gripper: Gripper | None, args
) -> None:
    """外层负责 reflex 自动恢复，内层是实际的跟随循环。

    接触密集任务（插孔、擦拭）难免触发 reflex。每次都退出程序没法用，
    所以这里捕获 ControlException → 清错 → 重新锚定 → 继续，
    只有短时间内反复触发才判定为异常并退出。
    """
    n_recover = 0
    recover_times: list[float] = []

    print("\n" + "=" * 60)
    print("遥操作已启动")
    print("  空格 = 离合（暂停/恢复跟随）    q = 退出")
    print(f"  碰撞阈值 {COLLISION_TORQUE[0]:.0f}/{COLLISION_TORQUE[4]:.0f} Nm"
          f"（触发后自动恢复，最多 {args.max_recover} 次/分钟）")
    print("=" * 60 + "\n")

    while True:
        try:
            quit_requested = _teleop_session(reader, mapper, robot, gripper, args)
            if quit_requested:
                return
        except franky.ControlException as exc:
            now = time.perf_counter()
            recover_times = [t for t in recover_times if now - t < 60.0]
            recover_times.append(now)
            n_recover += 1

            msg = str(exc).split("\n")[0][:110]
            print(f"\n\n⚠️ 触发 reflex（第 {n_recover} 次）: {msg}")

            if len(recover_times) > args.max_recover:
                print(f"❌ 一分钟内触发超过 {args.max_recover} 次，可能有持续的异常受力。")
                print("   请检查机械臂是否卡住/顶住障碍物，然后重新启动。")
                return

            print("   正在清除错误并恢复 ...")
            try:
                robot.recover_from_errors()
            except Exception as rec_exc:  # noqa: BLE001
                print(f"   ❌ 恢复失败: {rec_exc}")
                print("   请到 Desk 网页查看错误，可能需要手动确认。")
                return

            # 重新锚定：FR3 停在哪就以哪为新起点，主臂当前姿态为新零点。
            # 不这样做的话，恢复瞬间会按旧映射猛冲。
            vals, _ = reader.latest()
            if vals is not None:
                mapper.set_zero(vals)
            mapper.franka_ready = np.asarray(robot.current_joint_state.position)
            print("   ✅ 已恢复并重新对齐，继续遥操作\n")
            time.sleep(0.3)


def _teleop_session(
    reader: UArmReader, mapper: Mapper, robot: Robot,
    gripper: Gripper | None, args
) -> bool:
    """一次跟随会话。返回 True 表示用户主动退出，抛异常表示触发 reflex。"""
    n_timeout = 0
    engaged = True
    last_target: np.ndarray | None = None
    last_grip_ratio = -1.0
    grip_future = None
    t_prev = time.perf_counter()
    t_report = 0.0

    damping = (np.array(DAMPING) * args.damping_scale).tolist()
    if args.damping_scale != 1.0:
        print(f"阻尼 ×{args.damping_scale}: {np.round(damping, 1).tolist()}")

    with KeyPoller() as keys, JointImpedanceTracker(
        robot,
        stiffness=STIFFNESS,
        damping=damping,
        lower_joint_limits=FR3_LOWER,
        upper_joint_limits=FR3_UPPER,
        period=0.002,
    ) as tracker:
        # 先把目标设成当前位置：万一首个循环因超时或离合而跳过 set_target，
        # 控制器也不会拿着未初始化的参考值去算力矩。
        tracker.set_target(np.asarray(robot.current_joint_state.position))

        while tracker.tick():
            k = keys.poll()
            if k == "q":
                print("\n收到退出指令 ...")
                return True
            if k == " ":
                engaged = not engaged
                if engaged:
                    # 重新对齐：把当前主臂姿态重新设为零点，避免恢复瞬间跳变。
                    vals, _ = reader.latest()
                    if vals is not None:
                        mapper.set_zero(vals)
                        mapper.franka_ready = np.asarray(
                            robot.current_joint_state.position
                        )
                        last_target = None
                print(f"\n>>> {'恢复跟随' if engaged else '已离合（暂停）'}\n")

            vals, age = reader.latest()
            if vals is None or age > LEADER_TIMEOUT:
                n_timeout += 1
                if n_timeout == 1:
                    print(f"\n⚠️ 主臂数据超时（{age:.1f}s），保持当前目标 ...")
                if n_timeout > 500:  # 约 1 秒
                    print("\n❌ 主臂持续掉线，退出。")
                    return True
                continue
            n_timeout = 0

            if not engaged:
                continue

            # 死区：滤掉舵机读数噪声（主臂静止时从臂不该抖）
            vals = mapper.apply_deadband(vals, args.deadband)
            raw_target = mapper.to_franka(vals)

            now = time.perf_counter()
            dt = max(now - t_prev, 1e-4)
            t_prev = now
            if last_target is None:
                last_target = np.asarray(robot.current_joint_state.position)

            # 指数插值：主臂只有 15Hz，目标每 66ms 才更新一次。直接下发会让
            # FR3「追一下停一下」，这里以 500Hz 平滑逼近，填平台阶。
            # （U-Arm 的 servo_reader.py 用 num_interp=5 × delta*0.2 做同一件事）
            if args.smooth > 0:
                alpha = min(1.0, dt / (args.smooth / 1000.0))
                target = last_target + alpha * (raw_target - last_target)
            else:
                target = raw_target

            # 逐周期速度钳位（防止猛甩主臂导致暴冲）
            step = MAX_JOINT_VEL * dt * args.speed
            target = last_target + np.clip(target - last_target, -step, step)
            last_target = target

            tracker.set_target(target)

            # 夹爪。Franka Hand 的 move 是「走到指定宽度」的离散动作，不是
            # 连续流式接口，所以只能在变化超过阈值时发新指令。
            # 阈值太小 → 指令太密，每条的通信开销叠起来反而更顿；
            # 太大 → 分段明显。配合速度一起调。
            if gripper is not None:
                ratio = mapper.to_gripper_ratio(vals[GRIPPER_ID])
                ready = grip_future is None or grip_future.wait(0)
                if args.gripper_binary:
                    # 二值模式：过阈值就 grasp（带夹持力，能真正抓住东西），
                    # 否则张开。比连续调宽度可靠，代价是失去中间开度。
                    want_close = ratio < args.gripper_thresh
                    if ready and want_close != (last_grip_ratio < args.gripper_thresh):
                        if want_close:
                            grip_future = gripper.grasp_async(
                                0.0, args.gripper_speed, args.gripper_force,
                                epsilon_outer=1.0,
                            )
                        else:
                            grip_future = gripper.open_async(args.gripper_speed)
                        last_grip_ratio = ratio
                elif abs(ratio - last_grip_ratio) > args.gripper_step and ready:
                    width = ratio * gripper.max_width
                    grip_future = gripper.move_async(
                        float(width), args.gripper_speed
                    )
                    last_grip_ratio = ratio

            if now - t_report >= 1.0:
                t_report = now
                q_now = np.asarray(robot.current_joint_state.position)
                err = np.rad2deg(np.max(np.abs(target - q_now)))
                n_read, n_fail = reader.stats
                print(f"\r跟随误差 {err:5.1f}°   主臂读取 {n_read} 次"
                      f"（失败 {n_fail}）   {'●跟随' if engaged else '○离合'}   ",
                      end="", flush=True)

    # tick() 返回 False：控制会话被 franky 结束，交给外层判断要不要恢复。
    return False


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="192.168.3.1")
    parser.add_argument("--port", default=PORT)
    parser.add_argument("--dry-run", action="store_true", help="FR3 不动，只打印映射")
    parser.add_argument("--no-gripper", action="store_true")
    parser.add_argument("--no-home", action="store_true", help="跳过归位（假定已在 ready）")
    parser.add_argument(
        "--speed", type=float, default=1.0, help="速度钳位倍率，<1 更保守"
    )
    parser.add_argument(
        "--gripper-speed", type=float, default=0.2,
        help="夹爪速度 m/s。实测 Franka Hand 的实际速度在 0.2 设定处饱和"
        "（约 0.05 m/s，全行程 1.4s），设更高无效",
    )
    parser.add_argument(
        "--gripper-step", type=float, default=0.05,
        help="连续模式下开合比例变化超过多少才发新指令（0~1）。"
        "太小则指令过密、通信开销叠加反而更顿",
    )
    parser.add_argument(
        "--gripper-binary", action="store_true",
        help="二值模式：过阈值就 grasp（带夹持力，能真抓住东西），否则张开。"
        "比连续调宽度可靠，适合抓取任务，代价是没有中间开度",
    )
    parser.add_argument(
        "--gripper-thresh", type=float, default=0.5,
        help="二值模式的闭合阈值（0~1，扳机开合比例低于此值判为要闭合）",
    )
    parser.add_argument(
        "--gripper-force", type=float, default=30.0,
        help="二值模式的夹持力 N",
    )
    parser.add_argument(
        "--collision-scale", type=float, default=1.0,
        help="碰撞阈值倍率。默认 1.0 = Franka 官方遥操作配置（85/11 Nm，接近上限）。"
        "调小会更早触发保护但更容易误报；接触密集任务不建议低于 1.0",
    )
    parser.add_argument(
        "--max-recover", type=int, default=10,
        help="一分钟内允许自动恢复的次数，超过则判定异常并退出",
    )
    parser.add_argument(
        "--keep-torque", action="store_true",
        help="不释放主臂舵机力矩。默认会释放 —— 舵机位置保持模式下会因反馈噪声"
        "持续打摆（物理抖动）。仅当主臂靠舵机力矩撑着、放掉会垮时才用这个",
    )
    parser.add_argument(
        "--damping-scale", type=float, default=1.0,
        help="从臂阻尼倍率。阻尼专门抑制振荡，加大能压住抖动，"
        "且不像插值那样牺牲整体延迟。抖得厉害可以试 1.5~2.0",
    )
    # 死区和插值默认关闭：抖动的根因是主臂伺服打摆，已由启动时释放力矩解决。
    # 这两层滤波原本是治那个症状的，症状消失后它们只剩延迟代价。
    # 留作可选项，万一换了硬件或环境又出现抖动可以随时开启。
    parser.add_argument(
        "--deadband", type=float, default=0.0,
        help="主臂读数死区（度）。变化小于此值视为噪声、忽略。"
        "对有意动作零延迟代价。U-Arm 原代码用 1.0。默认 0 = 关闭",
    )
    parser.add_argument(
        "--smooth", type=float, default=0.0,
        help="目标插值时间常数（毫秒）。填平 15Hz 采样的台阶，"
        "但直接换延迟。默认 0 = 关闭",
    )
    parser.add_argument(
        "--wait", type=float, default=0.008,
        help="每条串口指令后的等待秒数。默认 0.008（15Hz），"
        "是 uarm_bench.py 实测的可用下限（847 次读取 0 失败）。"
        "再小会断崖式丢包 —— 中励舵机固件响应约需 6-7ms，是硬件特性。",
    )
    args = parser.parse_args()

    calib_path = Path(__file__).parent / "uarm_calib.json"
    if not calib_path.exists():
        raise SystemExit(f"❌ 找不到标定文件 {calib_path}，先跑 uarm_calibrate.py")
    mapper = Mapper.from_calib(calib_path)
    print(f"✅ 标定已载入  scale={mapper.scale.astype(int).tolist()}  "
          f"扳机 {mapper.trigger_open:.1f}°→{mapper.trigger_close:.1f}°")

    reader = UArmReader(args.port, BAUD, wait=args.wait)
    probe = reader.read_once()
    if probe is None:
        raise SystemExit("❌ 主臂无响应，检查 ZLink 连线和舵机供电")
    print(f"✅ 主臂已连接，8/8 舵机响应")

    if not args.keep_torque:
        print("   释放舵机力矩（消除伺服打摆造成的物理抖动）...")
        reader.release_torque()
        print("   ✅ 已释放，主臂现在应该可以自由掰动")

    reader.start()

    robot = gripper = None
    try:
        if not args.dry_run:
            robot = Robot(args.host)
            robot.recover_from_errors()
            robot.set_collision_behavior(
                torque_thresholds=[t * args.collision_scale for t in COLLISION_TORQUE],
                force_thresholds=[f * args.collision_scale for f in COLLISION_FORCE],
            )
            robot.joint_acceleration_limit.set([5.0] * 7)
            print(f"✅ FR3 已连接")

            if not args.no_home:
                q = np.asarray(robot.current_joint_state.position)
                delta = np.rad2deg(np.max(np.abs(mapper.franka_ready - q)))
                print(f"\nFR3 将归位到 ready（最大转动 {delta:.0f}°）")
                input("按回车执行 ...")
                robot.move(JointMotion(
                    mapper.franka_ready.tolist(), relative_dynamics_factor=0.05
                ))
                print("✅ 已归位")

            # 锚点 = FR3 实际所在位置（归位后即 ready）
            mapper.franka_ready = np.asarray(robot.current_joint_state.position)

            if not args.no_gripper:
                gripper = Gripper(args.host)
                if gripper.max_width <= 0.0:
                    print("⚠️ 夹爪未标定过，先跑 gripper_init.py。本次禁用夹爪。")
                    gripper = None
                else:
                    print(f"✅ 夹爪已连接（最大 {gripper.max_width * 1000:.0f}mm）")

        # 采零点：此刻主臂姿态 ←→ FR3 当前姿态
        leader_now = wait_for_leader(reader)
        print(f"\n主臂当前读数: {np.round(leader_now[:N_JOINTS], 1)}")

        if args.dry_run:
            mapper.set_zero(leader_now)
            run_dry(reader, mapper)
            return

        print("\n⚠️ 按回车后 FR3 将开始跟随主臂。急停放手边，周围清空。")
        input("按回车开始遥操作 ...")
        leader_now = wait_for_leader(reader)
        mapper.set_zero(leader_now)

        run_teleop(reader, mapper, robot, gripper, args)

    except KeyboardInterrupt:
        print("\n\n收到 Ctrl+C")
    except franky.ControlException as exc:
        print(f"\n❌ 控制错误: {exc}")
    finally:
        reader.stop()
        n_read, n_fail = reader.stats
        if n_read:
            print(f"主臂共读取 {n_read} 次，失败 {n_fail} 次")
        print("已退出。FR3 停在当前位置，归位请跑 python home_robot.py")


if __name__ == "__main__":
    try:
        main()
    except franky.NetworkException as exc:
        print(f"❌ FR3 网络错误: {exc}")
        print("   检查 FCI 是否激活、网口速率、CPU 调频（见 franka开发坑记录.md）")
    except serial.SerialException as exc:
        print(f"❌ 串口错误: {exc}")
