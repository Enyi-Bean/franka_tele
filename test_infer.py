#!/usr/bin/env python3
"""最小连接自测：只连 pi05 策略服务器、发随机 obs、打印返回，不接机械臂。

用来把「网络/客户端/推理」和「机械臂控制」分开调试 —— 这一步过了，
再跑 deploy_pi05.py 接真机就只剩控制逻辑要调。

前置：另一个终端起 SSH 端口转发（把本机 8000 转到服务器 8000）
    ssh -N -L 8000:localhost:8000 <user>@<server>
且服务器已起 serve_franka.py。

用法：
    python test_infer.py                 # 连 localhost:8000，发 3 帧
    python test_infer.py --n 10          # 发 10 帧看推理耗时
"""

from argparse import ArgumentParser
import time

import numpy as np

from openpi_client import websocket_client_policy

ap = ArgumentParser()
ap.add_argument("--host", default="localhost")
ap.add_argument("--port", type=int, default=8000)
ap.add_argument("--prompt", default="put the cube on the plate")
ap.add_argument("--n", type=int, default=3)
ap.add_argument("--zero-state", action="store_true",
                help="用零 state（默认用 ready 姿态，能验证绝对角还原落在 ready 附近）")
args = ap.parse_args()

# ready 起点（= 部署真实起点），用它当 state 能直接看出绝对角还原是否正确
READY = np.array([0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, np.pi / 4])
state0 = np.zeros(8, dtype=np.float32)
if not args.zero_state:
    state0[:7] = READY
    state0[7] = 0.04            # 夹爪半开（米制）

print(f"连接策略服务器 {args.host}:{args.port} ...")
client = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
print("✅ 已连接。服务器 metadata:", client.get_server_metadata())

for i in range(args.n):
    obs = {
        "observation/state": state0.copy(),
        "observation/image": np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8),
        "prompt": args.prompt,
    }
    t = time.perf_counter()
    out = client.infer(obs)
    dt = (time.perf_counter() - t) * 1000
    a = np.asarray(out["actions"])
    print(f"[{i}] actions shape={a.shape} dtype={a.dtype} "
          f"infer={dt:.0f}ms  含NaN={np.isnan(a).any()}")
    if i == 0:
        print("  action[0] 臂 =", np.array2string(a[0, :7], precision=4))
        if not args.zero_state:
            print("  ready 姿态 =", np.array2string(READY, precision=4))
            print("  action[0]臂 - ready =", np.array2string(a[0, :7] - READY, precision=4),
                  "(应是小 delta，说明绝对角落在 ready 附近)")
        print("  夹爪[7] 范围 =", f"{a[:, 7].min():.3f} ~ {a[:, 7].max():.3f}")

print("✅ 自测完成。期望：shape=(50, 8)、无 NaN、夹爪∈[0,1]")
if not args.zero_state:
    print("   且 action 臂[:7] 应接近 ready 姿态（绝对角还原验证）")
