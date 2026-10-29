#!/usr/bin/env python3
"""把 FR3 用关节空间运动归位到一个安全姿态。

关节空间运动是脱离奇异点的正确方式：它只保证每个关节平滑转到目标角度，
不做逆运动学，因此不受雅可比病态影响。（笛卡尔运动在奇异点附近会让
关节速度爆炸，绝不要用来脱困。）

用法：
    python home_robot.py --dry-run          # 只看状态，不动
    python home_robot.py                    # 归位到 ready 姿态
    python home_robot.py --pose zero        # 归位到全零姿态
    python home_robot.py --factor 0.03      # 更慢

前提：Desk 里已清除错误、解开刹车、激活 FCI。
"""

from argparse import ArgumentParser

import numpy as np

import franky
from franky import JointMotion, Robot

# FR3 的关节限位（来自 franky examples/cartesian_impedance_tracker.py）。
# 注意 J4 上限恒为负、J6 下限恒为正，这两个最容易在手动拖动时拖过头。
FR3_LOWER = np.array([-2.9007, -1.8361, -2.9007, -3.0770, -2.8763, 0.4398, -3.0508])
FR3_UPPER = np.array([2.9007, 1.8361, 2.9007, -0.1169, 2.8763, 4.6216, 3.0508])

POSES = {
    # libfranka 官方示例的标准姿态，各关节余量都很充足。
    "ready": [0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, np.pi / 4],
    # franky 教程里用的姿态，比 ready 更舒展一些。
    "tutorial": [0.0, -0.6, 0.0, -2.2, 0.0, 1.7, 0.7],
    # 尽量接近全零（J4/J6 取各自限位内最接近 0 的合法值）。
    "zero": [0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.0],
}


def report(q: np.ndarray, title: str) -> bool:
    """打印关节角与限位余量，返回是否全部在限位内。"""
    print(f"\n=== {title} ===")
    print("关节   角度(rad)   角度(deg)     下限      上限    余量    状态")
    ok = True
    for i, (v, lo, hi) in enumerate(zip(q, FR3_LOWER, FR3_UPPER)):
        margin = min(v - lo, hi - v)
        if margin < 0:
            flag, ok = "❌ 越界", False
        elif margin < 0.1:
            flag = "⚠️ 接近"
        else:
            flag = "ok"
        print(
            f"J{i + 1}   {v:9.4f}  {np.rad2deg(v):9.2f}  "
            f"{lo:8.4f}  {hi:8.4f}  {margin:6.3f}  {flag}"
        )
    return ok


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    # FCI 在 Control 单元的 LAN 口（Desk 里叫 "C2 - Shop Floor network"）上。
    # Arm 的口（"X5 - Robot network"，192.168.0.1）只有 Desk 网页，没有 FCI。
    parser.add_argument("--host", default="192.168.3.1", help="机械臂 FCI IP")
    parser.add_argument(
        "--pose", default="ready", choices=sorted(POSES), help="目标姿态"
    )
    parser.add_argument(
        "--factor", type=float, default=0.05, help="速度系数（占上限的比例）"
    )
    parser.add_argument(
        "--accel",
        type=float,
        default=5.0,
        help="关节加速度上限 rad/s²。默认 5.0 对齐 Franka ROS2 官方示例；"
        "franky 自己的默认值是 7.5~20，会更容易触发 acceleration_discontinuity。",
    )
    parser.add_argument("--dry-run", action="store_true", help="只读状态，不运动")
    parser.add_argument(
        "--zero-load",
        action="store_true",
        help="把负载清零。仅在确实没装任何末端执行器时使用；装了夹爪不要加这个。",
    )
    parser.add_argument(
        "--yes", action="store_true", help="跳过确认提示（谨慎使用）"
    )
    args = parser.parse_args()

    print(f"连接 {args.host} ...")
    robot = Robot(args.host)
    print("✅ 已连接")

    # 默认沿用 Desk 里配置的末端负载 —— 装了夹爪时这才是对的。
    # 只有在实际没装任何末端、而 Desk 里还配着质量时，才用 --zero-load 覆盖。
    if args.zero_load:
        robot.set_load(0.0, [0.0, 0.0, 0.0], [0.0] * 9)
        print("已将负载设为 0（--zero-load：确认未装末端执行器）")
    else:
        print("负载沿用 Desk 配置（装了夹爪时应如此）")

    state = robot.state
    errors = [
        n
        for n in dir(state.current_errors)
        if not n.startswith("_") and getattr(state.current_errors, n) is True
    ]
    print(f"\n当前错误: {errors if errors else '（无）'}")
    if errors:
        print("尝试清除 ...")
        robot.recover_from_errors()
        print("已调用 recover_from_errors()")

    q_now = np.asarray(robot.current_joint_state.position)
    in_limits = report(q_now, "当前姿态")
    target = np.asarray(POSES[args.pose], dtype=float)
    report(target, f"目标姿态 ({args.pose})")

    max_delta = np.max(np.abs(target - q_now))
    print(f"\n最大单关节转动量: {np.rad2deg(max_delta):.1f}°")

    if args.dry_run:
        print("\n[dry-run] 未发送任何运动指令。")
        return

    if not in_limits:
        print(
            "\n❌ 当前姿态仍有关节越界，拒绝运动。\n"
            "   请先手动把越界关节掰回范围内，并在 Desk 中确认恢复。"
        )
        return

    if not args.yes:
        print(f"\n机械臂将以 {args.factor:.0%} 速度移动到 '{args.pose}' 姿态。")
        print("确认周围无障碍物、急停在手边。")
        if input("输入 yes 继续: ").strip().lower() != "yes":
            print("已取消。")
            return

    # 保守的碰撞阈值：撞到东西会立刻停，而不是硬顶过去。
    robot.set_collision_behavior(
        torque_thresholds=[20.0] * 7,
        force_thresholds=[20.0] * 6,
    )
    # 加速度单独限制，不靠 relative_dynamics_factor 压——后者会把速度一起拖慢。
    robot.joint_acceleration_limit.set([args.accel] * 7)
    robot.relative_dynamics_factor = args.factor
    print(f"\n加速度上限设为 {args.accel} rad/s²，速度系数 {args.factor}")

    print("\n运动中 ...")
    robot.move(JointMotion(target.tolist()))
    print("✅ 完成")

    report(np.asarray(robot.current_joint_state.position), "归位后")


if __name__ == "__main__":
    try:
        main()
    except franky.NetworkException as exc:
        print(f"\n❌ 网络错误: {exc}")
        print("   检查：Desk 里 FCI 是否已激活？VPN 是否关闭？")
    except franky.ControlException as exc:
        print(f"\n❌ 控制错误: {exc}")
        print("   机械臂可能触发了碰撞检测或限位。检查 Desk 上的错误提示。")
