#!/usr/bin/env python3
"""Franka Hand 回零标定与基本功能测试。

夹爪装上后第一次使用必须做 homing：手指全开再全闭，自己测出行程极限。
没做之前 max_width 会是 0，move/grasp 都无法正常工作。

用法：
    python gripper_init.py --status        # 只看状态，不动
    python gripper_init.py                 # 回零标定
    python gripper_init.py --test          # 标定后再做开合测试
"""

from argparse import ArgumentParser

import franky


def show(gripper: franky.Gripper, label: str) -> None:
    print(f"  [{label}] 宽度={gripper.width:.5f}m  最大={gripper.max_width:.5f}m  "
          f"夹持中={gripper.is_grasped}")


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="192.168.3.1")
    parser.add_argument("--status", action="store_true", help="只读状态")
    parser.add_argument("--test", action="store_true", help="标定后做开合测试")
    parser.add_argument("--yes", action="store_true", help="跳过确认")
    args = parser.parse_args()

    gripper = franky.Gripper(args.host)
    print(f"✅ 已连接夹爪（server version {gripper.server_version}）")
    show(gripper, "当前")

    if args.status:
        if gripper.max_width == 0.0:
            print("\n⚠️ max_width 为 0 —— 尚未标定，需要运行 homing。")
        return

    if gripper.max_width > 0.0:
        print("\n夹爪似乎已标定过（max_width > 0）。重新标定也无妨。")

    if not args.yes:
        print("\n将执行回零标定：手指会先全开再全闭。")
        print("⚠️ 确保手指之间没有任何物体，手不要放在夹爪附近。")
        if input("输入 yes 继续: ").strip().lower() != "yes":
            print("已取消。")
            return

    print("\n标定中（约 10-20 秒）...")
    if not gripper.homing():
        print("❌ 标定失败。检查夹爪供电和连接。")
        return
    print("✅ 标定完成")
    show(gripper, "标定后")

    if gripper.max_width == 0.0:
        print("\n⚠️ max_width 仍为 0，标定可能没有真正生效。")
        return

    if not args.test:
        return

    print("\n=== 开合测试 ===")
    speed = 0.05  # m/s

    print(f"\n>>> 张开到最大 {gripper.max_width:.4f}m")
    gripper.move(gripper.max_width, speed)
    show(gripper, "张开后")

    half = gripper.max_width / 2
    print(f"\n>>> 合到一半 {half:.4f}m")
    gripper.move(half, speed)
    show(gripper, "半开后")

    print(f"\n>>> 再张开")
    gripper.open(speed)
    show(gripper, "完成")


if __name__ == "__main__":
    try:
        main()
    except franky.NetworkException as exc:
        print(f"❌ 网络错误: {exc}")
