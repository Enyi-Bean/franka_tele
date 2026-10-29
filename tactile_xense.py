#!/usr/bin/env python3
"""千觉光子触觉传感器取流封装（Xense SDK 2.1.2）。

和 camera_d435i.CameraD435i 同样的思路：后台线程持续取帧，主循环取最新帧。
区别是**一个源产出两路图**，所以 latest_images() 返回 dict 而不是单张图：

    {"<name>": Rectify(700,400,3), "<name>_diff": Difference(700,400,3)}

name 默认 "tactile"；夹爪两指各装一个时用 "tactile_left" / "tactile_right"，
两个实例的图像路名字才不会撞（writer 按名字分路）。

⭐ 两路必须在同一次 selectSensorInfo 里取 —— SDK 保证同一次调用的多个输出来自
同一帧，且计算最优（实测全量 38ms ≈ 只要一路的 39ms）。拆成两次调用会丢掉这个保证。

采集期 disable_infer=True 只取图、不跑推理：
  - Rectify / Difference 都是纯 CV，不需要神经网络
  - Depth / Force / Marker 事后可用 Sensor.createSolver() 从存下的 Rectify 补算
  - 省下的 CPU 留给 1kHz 控制环（开发记录第 0/9 节：曾被相机+编码抢到
    control_command_success_rate = 0.14）

用法：
    from tactile_xense import TactileXense
    with TactileXense("OG001273", name="tactile_left") as tac:
        tac.calibrate()              # 必须在无接触时、起线程前调
        tac.start()
        imgs, age = tac.latest_images()   # imgs["tactile_left"], imgs["tactile_left_diff"]
"""

from __future__ import annotations

import threading
import time

import numpy as np

try:
    from xensesdk import Sensor
except ImportError:
    Sensor = None  # 纯视觉采集不需要安装触觉 SDK。

# 数据集里的图像路名字（= LeRobot 的 observation.images.<key>）：
#   Rectify → <name>，Difference → <name>_diff
DEFAULT_NAME = "tactile"
DIFF_SUFFIX = "_diff"

# Rectify/Difference 的固定尺寸（实测 OG001276）
TACTILE_H, TACTILE_W = 700, 400

# 后台轮询限速。采集节拍只要 15Hz，取图能跑 58Hz —— 全速跑会占满一个核，
# 限到 25Hz 既有余量又不抢 CPU。
POLL_HZ = 25.0


class TactileXense:
    """千觉触觉传感器后台取流。一个源 → 两路图。"""

    def __init__(self, serial: str = "OG001273", name: str = DEFAULT_NAME,
                 poll_hz: float = POLL_HZ):
        # cam_id 的注解是 Optional[Union[int, str]]，传序列号字符串合法。
        # 不写死 cam_id=0 —— 多设备时会抓错设备（D405 那次已踩过）。
        self.serial = serial
        self._name = name
        self.key_rectify = name
        self.key_diff = name + DIFF_SUFFIX
        self._poll_dt = 1.0 / poll_hz
        self._sensor: Sensor | None = None
        self._lock = threading.Lock()
        self._latest: dict[str, np.ndarray] | None = None
        self._stamp = 0.0
        self._n_frames = 0
        self._thread: threading.Thread | None = None
        self._running = False
        self._error: str | None = None

    # ---------- 连接 ----------

    def open(self) -> None:
        """创建传感器实例。失败时抛异常并带上可查的排障提示。"""
        if Sensor is None:
            raise RuntimeError("触觉采集需要 xensesdk 2.1.2；见 README.md。纯视觉请加 --no-tactile。")
        found = Sensor.scanSerialNumber()
        if self.serial not in found:
            raise RuntimeError(
                f"❌ 未找到触觉传感器 {self.serial}，扫描到的是 {found or '（空）'}。\n"
                f"   查：USB 线是否插好 / udev 规则是否装了（bash setup_xense_udev.sh）"
            )
        # 2.1.2 没有 use_gpu 参数（README 的 API 文档是 1.x 的）；
        # 没有 CUDA 时 SDK 自己回退 CPU，无需手动指定。
        self._sensor = Sensor.create(self.serial, disable_infer=True)

    def calibrate(self) -> np.ndarray:
        """重设参考帧（Difference 的基线），返回校准时刻的 Rectify 图。

        ⚠️ 必须在**传感器完全无接触**时调用，且必须在 start() 之前 ——
        它会重设参考帧，而后台线程正在轮询 selectSensorInfo，
        SDK 的线程安全性未知，并发调用可能拿到半新半旧的基线。

        返回的图要存进 meta/tactile_ref/：参考帧只活在 SDK 内存里，
        下次 calibrate 就没了，是整个方案里唯一不可逆的东西。存下来才能
        在跨会话基线漂移时，用统一基线从 Rectify 批量重算 Difference。
        """
        if self._sensor is None:
            raise RuntimeError("请先 open()")
        if self._running:
            raise RuntimeError("calibrate() 必须在 start() 之前调用（见 docstring）")
        self._sensor.calibrateSensor()
        # 校准后取一帧作为基线存档
        ref = self._sensor.selectSensorInfo(Sensor.OutputType.Rectify)
        return np.array(ref, dtype=np.uint8, copy=True)   # 同 _loop：防复用缓冲

    # ---------- 后台取流 ----------

    def start(self) -> None:
        if self._sensor is None:
            self.open()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        assert self._sensor is not None
        while self._running:
            t0 = time.perf_counter()
            try:
                rect, diff = self._sensor.selectSensorInfo(
                    Sensor.OutputType.Rectify,
                    Sensor.OutputType.Difference,
                )
            except Exception as e:          # 掉线/USB 抖动
                self._error = str(e)
                time.sleep(0.05)
                continue
            if rect is None or diff is None:
                time.sleep(0.01)
                continue
            # ⚠️ 必须真拷贝：SDK 返回的可能是内部复用缓冲的**视图**，
            # 下一帧覆写同一块内存时，已经存进 _latest 的"上一帧"会被静默改掉
            # （表现为串帧 / 图像撕裂，事后极难定位）。
            rect = np.array(rect, dtype=np.uint8, copy=True)
            diff = np.array(diff, dtype=np.uint8, copy=True)
            with self._lock:
                self._latest = {self.key_rectify: rect, self.key_diff: diff}
                self._stamp = time.perf_counter()
                self._n_frames += 1
            # 限速：留 CPU 给 1kHz 控制环
            rest = self._poll_dt - (time.perf_counter() - t0)
            if rest > 0:
                time.sleep(rest)

    def latest_images(self) -> tuple[dict[str, np.ndarray] | None, float]:
        """返回 ({图像路名: BGR}, 距今秒数)。

        名字和 CameraD435i.latest_images() 一致 —— 两者都是「源」，
        record.py 的主循环对它们一视同仁。

        ⚠️ 和 CameraD435i 一样：首帧之后**永远不返回 None**，只返回上一帧 + age。
        调用方必须用 age 判掉线，不能用 `is not None`。
        """
        with self._lock:
            if self._latest is None:
                return None, float("inf")
            return (
                {k: v.copy() for k, v in self._latest.items()},
                time.perf_counter() - self._stamp,
            )

    def wait_first_frame(self, timeout: float = 5.0) -> dict[str, np.ndarray]:
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout:
            imgs, _ = self.latest_images()
            if imgs is not None:
                return imgs
            time.sleep(0.02)
        raise TimeoutError(
            f"触觉传感器 {timeout}s 内没有出图"
            + (f"（后台线程报错: {self._error}）" if self._error else "")
        )

    # ---------- 供 record.py / writer 用 ----------

    @property
    def image_shapes(self) -> dict[str, tuple[int, int, int]]:
        """这个源产出的所有图像路及其 shape（一个源 → 两路）。"""
        return {
            self.key_rectify: (TACTILE_H, TACTILE_W, 3),
            self.key_diff: (TACTILE_H, TACTILE_W, 3),
        }

    @property
    def name(self) -> str:
        """源名（状态行按源打印帧数；也是 Rectify 那路的图像路名）。"""
        return self._name

    @property
    def n_frames(self) -> int:
        return self._n_frames

    @property
    def error(self) -> str | None:
        return self._error

    # ---------- 生命周期 ----------

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._sensor is not None:
            self._sensor.release()
            self._sensor = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc):
        self.stop()


def main() -> None:
    """单独跑一遍验证：出图、帧率、掉线检测。"""
    from argparse import ArgumentParser

    import cv2

    ap = ArgumentParser(description="触觉传感器自检")
    ap.add_argument("--serial", default="OG001273")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--calibrate", action="store_true",
                    help="先校准基线（确保传感器无接触！）")
    ap.add_argument("--save", action="store_true", help="存一帧到 tactile_selftest_*.png")
    args = ap.parse_args()

    with TactileXense(args.serial) as tac:
        if args.calibrate:
            input("确认传感器表面无接触，回车开始校准 ...")
            ref = tac.calibrate()
            print(f"✅ 已校准，基线图 shape={ref.shape}")

        tac.start()
        tac.wait_first_frame()
        print(f"✅ 出图正常，跑 {args.seconds}s ...")

        t0 = time.perf_counter()
        n0 = tac.n_frames
        max_age = 0.0
        while time.perf_counter() - t0 < args.seconds:
            imgs, age = tac.latest_images()
            max_age = max(max_age, age)
            time.sleep(1.0 / 15)          # 模拟 15Hz 采集节拍

        dt = time.perf_counter() - t0
        print(f"   后台帧率 {(tac.n_frames - n0) / dt:.1f} Hz"
              f"（限速目标 {POLL_HZ:.0f} Hz）")
        print(f"   最大 age {max_age * 1000:.0f} ms（>200ms 判为掉线）")
        imgs, _ = tac.latest_images()
        for k, v in imgs.items():
            print(f"   {k}: shape={v.shape} dtype={v.dtype} "
                  f"均值={v.mean():.1f} 标准差={v.std():.1f}")
            if args.save:
                cv2.imwrite(f"tactile_selftest_{k}.png", v)
        if args.save:
            print("   已存 tactile_selftest_*.png")


if __name__ == "__main__":
    main()
