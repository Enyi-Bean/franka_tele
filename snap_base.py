#!/usr/bin/env python3
"""D435i 抓一张图，调机位/曝光用。默认第三人称那台，采集同款参数（曝光 220 + 增益 16 + 白平衡 3800）。

用法：
    python snap_base.py                              # 第三人称 → base_view.png
    python snap_base.py --serial 233622077251 --out wrist_view.png   # 腕部那台
    python snap_base.py --exposure 0                 # 自动曝光对比
    python snap_base.py --wb 0                       # 白平衡自动对比
"""
import argparse
import time

import cv2

from camera_d435i import CameraD435i

BASE_SERIAL = "135122077750"     # 第三人称 D435i；腕部那台是 233622077251

ap = argparse.ArgumentParser()
ap.add_argument("--serial", default=BASE_SERIAL)
ap.add_argument("--exposure", type=float, default=220, help="固定曝光，采集用 220；0=自动曝光")
ap.add_argument("--gain", type=float, default=16)
ap.add_argument("--wb", type=float, default=3800, help="白平衡色温，采集用 3800；0=自动")
ap.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270])
ap.add_argument("--out", default="base_view.png")
args = ap.parse_args()

cam = CameraD435i(640, 480, 15, auto_exposure=True, serial=args.serial, name="base",
                  rotate=args.rotate, exposure=(args.exposure or None), gain=args.gain,
                  white_balance=(args.wb or None))
cam.start()
cam.wait_first_frame()
time.sleep(1.5)                  # 等自动曝光稳定
img, _ = cam.latest()
cam.stop()
cv2.imwrite(args.out, img)
mode = f"exposure={args.exposure:.0f}" if args.exposure else "自动曝光"
print(f"✅ 已保存 {args.out}  shape={img.shape}  serial={args.serial}  {mode}")
