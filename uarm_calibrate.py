#!/usr/bin/env python3
"""U-Arm 主臂 ↔ FR3 从臂的方向标定。

默认用「双姿态法」：因为 U-Arm Config 3 是 FR3 的等比复制品，
"把主臂摆成和 FR3 一样的姿势" 是个很自然的动作。只做两次，
就能一次解出全部 7 个关节的方向符号：

    scale[i] = sign( Δleader[i] / Δfranka[i] )

比逐关节测试快得多（两次摆姿势 vs 七轮点头+掰动）。

顺带得到一个质量指标：两边角度变化量的比值。U-Arm 是等比复制品，
关节角应该 1:1 对应，所以 |ratio| 应该接近 1。明显偏离 1 说明
关节对应关系有问题（比如像 teleop_sim.py 里那样存在关节顺序对调）。

用法：
    python uarm_calibrate.py                  # 双姿态法，标定全部
    python uarm_calibrate.py --per-joint 2 5  # 逐关节复测 J3 和 J6（FR3 点头示范）
    python uarm_calibrate.py --gripper-only   # 只重标夹爪行程
"""

from argparse import ArgumentParser
import json
import re
import time
from pathlib import Path

import numpy as np
import serial

import franky
from franky import JointMotion, Robot

PORT, BAUD = "/dev/ttyUSB0", 115200
PWM_MIN, PWM_MAX, ANGLE_RANGE = 500, 2500, 270.0

N_JOINTS = 7
GRIPPER_ID = 7

FR3_LOWER = np.array([-2.9007, -1.8361, -2.9007, -3.0770, -2.8763, 0.4398, -3.0508])
FR3_UPPER = np.array([2.9007, 1.8361, 2.9007, -0.1169, 2.8763, 4.6216, 3.0508])

# 姿态 A = 标准 ready；姿态 B 在每个关节上都和 A 差 30° 以上，
# 这样每个符号都能被明确解出。两者都在 FR3 限位内且余量充足。
POSE_A_DEG = np.array([0.0, -45.0, 0.0, -135.0, 0.0, 90.0, 45.0])
POSE_B_DEG = np.array([35.0, -15.0, 35.0, -100.0, 35.0, 125.0, 0.0])

OUT = Path(__file__).parent / "uarm_calib.json"


def read_servo(ser: serial.Serial, sid: int, wait: float = 0.01) -> float | None:
    ser.reset_input_buffer()
    ser.write(f"#{sid:03d}PRAD!".encode("ascii"))
    time.sleep(wait)
    m = re.search(
        rf"#{sid:03d}P(\d{{4}})", ser.read_all().decode("ascii", errors="ignore")
    )
    return None if not m else (int(m.group(1)) - PWM_MIN) / (PWM_MAX - PWM_MIN) * ANGLE_RANGE


def read_stable(ser: serial.Serial, sid: int, samples: int = 7) -> float:
    """多采几次取中位数，压掉手抖和偶发丢包。"""
    vals = [v for v in (read_servo(ser, sid) for _ in range(samples)) if v is not None]
    if not vals:
        raise RuntimeError(f"舵机 {sid} 无响应")
    return float(np.median(vals))


def read_arm(ser: serial.Serial) -> np.ndarray:
    return np.array([read_stable(ser, i) for i in range(N_JOINTS)])


def move_fr3(robot: Robot, target_deg: np.ndarray, factor: float, label: str) -> None:
    q_now = np.asarray(robot.current_joint_state.position)
    target = np.deg2rad(target_deg)
    print(f"\n  FR3 将移动到{label}，最大转动 "
          f"{np.rad2deg(np.max(np.abs(target - q_now))):.0f}°")
    input("  按回车执行（Ctrl+C 中止）...")
    robot.move(JointMotion(target.tolist(), relative_dynamics_factor=factor))
    print("  ✅ 已到位")


def two_pose_calibration(ser, robot, args, calib) -> None:
    print("\n" + "=" * 64)
    print("双姿态标定")
    print("=" * 64)
    print("流程：FR3 摆姿态 A → 你把主臂摆成同样的样子 → 记录")
    print("      FR3 摆姿态 B → 你再摆一次 → 记录 → 自动解出全部符号")
    print("\n提示：只需大致相似即可，符号判断对精度不敏感。")

    move_fr3(robot, POSE_A_DEG, args.factor, "姿态 A（ready）")
    print("\n  👉 请把主臂摆成和 FR3 现在【同样的姿势】，保持不动")
    input("     按回车记录...")
    leader_a = read_arm(ser)
    print(f"     主臂读数: {np.round(leader_a, 1)}")

    move_fr3(robot, POSE_B_DEG, args.factor, "姿态 B")
    print("\n  👉 请把主臂再次摆成和 FR3 现在【同样的姿势】，保持不动")
    input("     按回车记录...")
    leader_b = read_arm(ser)
    print(f"     主臂读数: {np.round(leader_b, 1)}")

    d_leader = leader_b - leader_a
    d_franka = POSE_B_DEG - POSE_A_DEG

    print("\n" + "=" * 64)
    print(f"{'关节':>4} {'Δ主臂':>10} {'ΔFR3':>10} {'比值':>8} {'符号':>6}  质量")
    print("-" * 64)
    for i in range(N_JOINTS):
        ratio = d_leader[i] / d_franka[i]
        s = 1 if ratio > 0 else -1
        # U-Arm 是等比复制品，关节角应 1:1 对应，|ratio| 该接近 1。
        if abs(d_leader[i]) < 10.0:
            note, ok = "⚠️ 主臂变化太小，不可靠", False
        elif not 0.5 <= abs(ratio) <= 2.0:
            note, ok = f"⚠️ 比值偏离 1 较多，关节对应可能有误", False
        else:
            note, ok = "✅", True
        if ok:
            calib["scale"][i] = s
        print(f"J{i + 1:<3} {d_leader[i]:9.1f}° {d_franka[i]:9.1f}° "
              f"{ratio:8.2f} {s:+6d}  {note}")

    bad = [i + 1 for i in range(N_JOINTS) if calib["scale"][i] == 0]
    if bad:
        print(f"\n  以下关节未能可靠标定: {bad}")
        print(f"  用逐关节模式复测: python {Path(__file__).name} "
              f"--per-joint {' '.join(str(b - 1) for b in bad)}")


def nod(robot: Robot, joint: int, amp_deg: float, factor: float) -> bool:
    """让 FR3 的某个关节往正方向转一下再转回来。"""
    q0 = np.asarray(robot.current_joint_state.position)
    q1 = q0.copy()
    q1[joint] += np.deg2rad(amp_deg)
    if not (FR3_LOWER[joint] <= q1[joint] <= FR3_UPPER[joint]):
        q1[joint] = q0[joint] - np.deg2rad(amp_deg)
        if not (FR3_LOWER[joint] <= q1[joint] <= FR3_UPPER[joint]):
            print(f"  ⚠️ J{joint + 1} 两个方向都接近限位，先归位再试。")
            return False
        print("  （正方向受限，改为往负方向示范 —— 判断时请反过来看）")
    try:
        robot.move(JointMotion(q1.tolist(), relative_dynamics_factor=factor))
        time.sleep(0.3)
        robot.move(JointMotion(q0.tolist(), relative_dynamics_factor=factor))
    except franky.ControlException as exc:
        print(f"  ❌ FR3 运动失败: {exc}")
        return False
    return True


def per_joint_calibration(ser, robot, joints, args, calib) -> None:
    for i in joints:
        print(f"\n{'=' * 64}\n复测 J{i + 1}（主臂 ID {i}）\n{'=' * 64}")
        input(f"  按回车让 FR3 的 J{i + 1} 演示正方向（转 {args.amp:.0f}° 再转回）...")
        if not nod(robot, i, args.amp, args.factor):
            continue
        print(f"\n  请把主臂第 {i + 1} 个关节往【FR3 刚才转的方向】掰动约 20°")
        before = read_stable(ser, i)
        input("  掰好后按回车...")
        after = read_stable(ser, i)
        delta = after - before
        print(f"  主臂 ID {i}: {before:.1f}° → {after:.1f}°  变化 {delta:+.1f}°")
        if abs(delta) < 5.0:
            print("  ⚠️ 变化太小，跳过。请掰动更大幅度后重测。")
            continue
        calib["scale"][i] = 1 if delta > 0 else -1
        print(f"  → scale[{i}] = {calib['scale'][i]:+d}")


def gripper_calibration(ser, calib) -> None:
    print(f"\n{'=' * 64}\n标定夹爪 trigger（主臂 ID {GRIPPER_ID}）\n{'=' * 64}")
    input("  完全【松开】扳机，按回车...")
    g_open = read_stable(ser, GRIPPER_ID)
    input("  完全【扣紧】扳机，按回车...")
    g_close = read_stable(ser, GRIPPER_ID)
    span = abs(g_close - g_open)
    print(f"  松开 {g_open:.1f}°  →  扣紧 {g_close:.1f}°   行程 {span:.1f}°")
    if span < 10.0:
        print("  ⚠️ 行程太小，可能没扣到底或读错了 ID")
    calib["gripper"] = {"open_deg": g_open, "close_deg": g_close}


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="192.168.3.1")
    parser.add_argument("--port", default=PORT)
    parser.add_argument("--factor", type=float, default=0.05, help="FR3 速度系数")
    parser.add_argument("--amp", type=float, default=8.0, help="逐关节模式的演示幅度（度）")
    parser.add_argument(
        "--per-joint", type=int, nargs="+", default=None,
        help="逐关节复测指定关节（0-6），用 FR3 点头示范"
    )
    parser.add_argument("--gripper-only", action="store_true")
    args = parser.parse_args()

    calib = {"scale": [0] * N_JOINTS, "gripper": {}}
    if OUT.exists():
        calib.update(json.loads(OUT.read_text()))
        print(f"已载入现有配置 {OUT}")

    with serial.Serial(args.port, BAUD, timeout=0.05) as ser:
        time.sleep(0.2)
        n_ok = sum(read_servo(ser, i) is not None for i in range(8))
        print(f"串口已打开，{n_ok}/8 个舵机响应")
        if n_ok == 0:
            print(
                "\n❌ 没有任何舵机响应，无法标定。检查：\n"
                "   1. ZLink 到主臂的排线接好了吗？\n"
                "   2. 主臂舵机供电正常吗？\n"
                "   3. 确认串口路径、舵机 ID 0–7 和供电"
            )
            return
        if n_ok < 8:
            print(
                f"\n⚠️ 只有 {n_ok}/8 个舵机响应，标定结果会不完整。\n"
                "   建议先检查每个舵机的接线和 ID。"
            )
            if input("   仍要继续？(y/N) ").strip().lower() != "y":
                return

        if args.gripper_only:
            gripper_calibration(ser, calib)
        else:
            robot = Robot(args.host)
            robot.recover_from_errors()
            robot.set_collision_behavior(
                torque_thresholds=[20.0] * 7, force_thresholds=[20.0] * 6
            )
            robot.joint_acceleration_limit.set([5.0] * 7)
            q = np.asarray(robot.current_joint_state.position)
            print(f"✅ FR3 已连接，当前姿态(deg): {np.round(np.rad2deg(q), 1)}")

            try:
                if args.per_joint is not None:
                    per_joint_calibration(ser, robot, args.per_joint, args, calib)
                else:
                    two_pose_calibration(ser, robot, args, calib)
                    gripper_calibration(ser, calib)
            except KeyboardInterrupt:
                print("\n\n已中止，保存已完成的部分。")

    calib["leader_ids"] = {"arm": list(range(N_JOINTS)), "gripper": GRIPPER_ID}
    calib["franka_init_rad"] = np.deg2rad(POSE_A_DEG).tolist()
    OUT.write_text(json.dumps(calib, indent=2, ensure_ascii=False))

    print(f"\n{'=' * 64}\n=== 标定结果 ===")

    # 只详列本次测过的关节；其余用一行汇总，避免刷屏。
    tested = args.per_joint if args.per_joint is not None else range(N_JOINTS)
    for i in tested:
        s = calib["scale"][i]
        print(f"  J{i + 1}: scale = {s:+d}" if s else f"  J{i + 1}: ❌ 未标定")

    if args.per_joint is not None:
        summary = "  ".join(
            f"J{i + 1}:{s:+d}" if s else f"J{i + 1}:--"
            for i, s in enumerate(calib["scale"])
        )
        print(f"\n  全部: {summary}")

    if calib.get("gripper"):
        g = calib["gripper"]
        print(f"  夹爪: {g['open_deg']:.1f}° (开) → {g['close_deg']:.1f}° (闭)")

    missing = [i + 1 for i, s in enumerate(calib["scale"]) if s == 0]
    if missing:
        print(f"  ⚠️ 未标定: J{', J'.join(map(str, missing))}")
    print(f"\n已保存到 {OUT}")


if __name__ == "__main__":
    try:
        main()
    except franky.NetworkException as exc:
        print(f"❌ FR3 网络错误: {exc}\n   检查网线速率和 CPU 调频")
    except serial.SerialException as exc:
        print(f"❌ 串口错误: {exc}")
