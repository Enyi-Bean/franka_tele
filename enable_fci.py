#!/usr/bin/env python3
"""通过 Desk API 取得控制权、解开刹车并激活 FCI。

这个脚本不会让机械臂运动，只改变控制权和 FCI 开关状态。

用法：
    python enable_fci.py --user <用户名> --password <密码>
    python enable_fci.py --user U --password P --status-only   # 只看状态
    python enable_fci.py --user U --password P --force         # 强制夺取控制权
"""

from argparse import ArgumentParser
import socket
import time

import franky


def fci_port_open(host: str, timeout: float = 2.0) -> bool:
    """FCI 的 libfranka 命令端口是否在监听。"""
    try:
        with socket.create_connection((host, 1337), timeout=timeout):
            return True
    except OSError:
        return False


def show_status(desk, host: str) -> None:
    print("  刹车     :", desk.brake_state)
    print("  运行模式 :", desk.operating_mode)
    if hasattr(desk, "get_fci_status"):
        try:
            print("  FCI      :", desk.get_fci_status())
        except Exception as exc:  # noqa: BLE001
            print("  FCI      : (读取失败)", exc)
    if hasattr(desk, "get_recovery_status"):
        try:
            print("  待恢复项 :", desk.get_recovery_status() or "无")
        except Exception as exc:  # noqa: BLE001
            print("  待恢复项 : (读取失败)", exc)
    print("  1337端口 :", "开放 ✅" if fci_port_open(host) else "关闭 ❌")


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="192.168.3.1")
    parser.add_argument("--user", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--status-only", action="store_true", help="只读状态")
    parser.add_argument(
        "--force", action="store_true", help="强制夺取控制权（需按机械臂上的按钮）"
    )
    parser.add_argument(
        "--keep-brakes", action="store_true", help="不解刹车，仅激活 FCI"
    )
    parser.add_argument(
        "--release",
        action="store_true",
        help="结束时释放控制权。注意：这会同时关闭 FCI（用于收工，不是日常启动）",
    )
    args = parser.parse_args()

    last_error = None
    for cls in (franky.Desk, franky.DeskWebSession):
        try:
            print(f"用 {cls.__name__} 连接 {args.host} ...")
            with cls(args.host, args.user, args.password) as desk:
                print(f"✅ 连接成功\n")

                print("=== 当前状态 ===")
                show_status(desk, args.host)

                if args.status_only:
                    return

                print("\n=== 取得控制权 ===")
                try:
                    desk.take_control(wait_timeout=10.0)
                    print("  ✅ 已取得控制权")
                except franky.TakeControlTimeoutError:
                    if not args.force:
                        print(
                            "  ❌ 10 秒内未取得控制权。\n"
                            "     其他会话（浏览器标签）可能占着控制权。\n"
                            "     关掉其他 Desk 页面后重试，或加 --force 强制夺取。"
                        )
                        return
                    print("  超时，改为强制夺取 —— 请按机械臂手腕上方的按钮！")
                    desk.take_control(wait_timeout=30.0, force=True)
                    print("  ✅ 已强制取得控制权")

                if not args.keep_brakes:
                    print("\n=== 解开刹车 ===")
                    if desk.brake_state == franky.BrakeState.UNLOCKED:
                        print("  已经是解开状态，跳过")
                    else:
                        desk.unlock_brakes()
                        print("  ✅ 已解开（会听到七声咔哒）")

                print("\n=== 切到 Execution 模式 ===")
                desk.set_mode_execution()
                print("  ✅ 已切换")

                print("\n=== 激活 FCI ===")
                desk.enable_fci()
                print("  ✅ 已发送激活指令")

                for i in range(10):
                    if fci_port_open(args.host):
                        print(f"\n🎉 FCI 已激活（1337 端口开放，耗时 {i}s）")
                        break
                    time.sleep(1.0)
                else:
                    print("\n⚠️ 指令已发送，但 1337 端口仍未开放。")
                    print("   最终状态：")
                    show_status(desk, args.host)
                    return

                if args.release:
                    print("\n--release：释放控制权（FCI 会随之关闭）")
                    return

                # ⚠️ FCI 只在持有控制权期间有效 —— 释放控制权会立刻把它关掉。
                # 而 BaseDesk.close() 会自动 release_control()，所以这里必须
                # 阻止它释放：清掉内部 token，让 close() 跳过释放逻辑。
                # 效果等同于「浏览器标签一直开着」——服务端保留 token，FCI 保持激活。
                desk._BaseDesk__control_token = None
                desk._BaseDesk__control_token_id = None
                print("\n   控制权保留（未释放），FCI 将持续有效。")
                print("   机械臂指示灯应变成绿色。现在可以跑 home_robot.py 了。")
                print("   ⚠️ 下次再运行本脚本可能需要 --force + 按机械臂顶部按钮夺回控制权。")
            return
        except Exception as exc:  # noqa: BLE001 - 逐个尝试两种 API
            last_error = exc
            print(f"❌ {cls.__name__} 失败: {type(exc).__name__}: {exc}\n")

    print(f"两种 Desk API 都失败了。最后的错误: {last_error}")


if __name__ == "__main__":
    main()
