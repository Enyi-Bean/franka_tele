#!/usr/bin/env python3
"""RealSense 相机取流封装（D435i / D405 通用）。后台线程持续取帧，主循环取最新帧。

和 UArmReader 同样的思路：相机 30fps，采集循环 15Hz，两者解耦。
主循环每拍调 latest() 拿"当前最新的那一帧"，中间的帧丢弃。

多相机时必须按序列号指定设备（否则 pipeline 随机抓一台），并且不能写死
sensor 下标——D435i 有 3 个 sensor（[1]=RGB），D405 只有 1 个 sensor
（Stereo Module 自带 color），所以自动找"拥有 color 流"的 sensor 设曝光。

用法：
    from camera_d435i import CameraD435i, find_serial
    sn = find_serial("D405")
    with CameraD435i(640, 480, 30, serial=sn) as cam:
        img, age = cam.latest()      # img: BGR uint8 (H,W,3)
"""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np
import pyrealsense2 as rs

# 相机物理装歪时，取流后按度数旋正（只允许 90 的整倍数）
_ROTATE_CODE = {
    0: None,
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def list_devices() -> list[tuple[str, str]]:
    """返回 [(name, serial), ...]。"""
    return [
        (d.get_info(rs.camera_info.name),
         d.get_info(rs.camera_info.serial_number))
        for d in rs.context().query_devices()
    ]


def find_serial(model_substr: str) -> str | None:
    """按型号子串（如 'D405' / 'D435'）查序列号，找不到返回 None。"""
    for name, serial in list_devices():
        if model_substr.lower() in name.lower():
            return serial
    return None


def _find_color_sensor(profile):
    """在设备的所有 sensor 里找拥有 color 流的那个（D435i=RGB, D405=Stereo）。"""
    for s in profile.get_device().query_sensors():
        for p in s.get_stream_profiles():
            if p.stream_type() == rs.stream.color:
                return s
    return None


class CameraD435i:
    def __init__(self, width: int = 640, height: int = 480, fps: int = 30,
                 auto_exposure: bool = True, serial: str | None = None,
                 rotate: int = 0, exposure: float | None = None,
                 gain: float | None = None, white_balance: float | None = None,
                 name: str = "cam"):
        self.name = name                            # 图像路名字（= 数据集里的 key）
        self.width, self.height, self.fps = width, height, fps
        self.auto_exposure = auto_exposure
        self.serial = serial
        # 给了 exposure / white_balance 就锁定固定值（关自动），保证成像一致
        self.exposure = exposure
        self.gain = gain
        self.white_balance = white_balance
        if rotate not in _ROTATE_CODE:
            raise ValueError(f"rotate 只能是 {list(_ROTATE_CODE)}，收到 {rotate}")
        self._rotate_code = _ROTATE_CODE[rotate]     # 相机倒装/侧装时旋正
        self._pipeline = rs.pipeline()
        self._config = rs.config()
        if serial is not None:
            self._config.enable_device(serial)     # 多相机时锁定具体设备
        self._config.enable_stream(
            rs.stream.color, width, height, rs.format.bgr8, fps
        )
        self._lock = threading.Lock()
        self._latest: np.ndarray | None = None
        self._stamp = 0.0
        self._running = False
        self._thread: threading.Thread | None = None
        self._n_frames = 0

    def _apply_options(self, sensor) -> None:
        """设置曝光/白平衡。给了固定值就关自动锁死，否则用自动。"""
        if sensor is None:
            return
        # 曝光
        if self.exposure is not None:
            if sensor.supports(rs.option.enable_auto_exposure):
                sensor.set_option(rs.option.enable_auto_exposure, 0)
            if sensor.supports(rs.option.exposure):
                sensor.set_option(rs.option.exposure, self.exposure)
            # gain 也必须一并锁定（不给则用默认 16），否则会残留上次的值 → 不可复现
            if sensor.supports(rs.option.gain):
                sensor.set_option(rs.option.gain,
                                  self.gain if self.gain is not None else 16.0)
        elif sensor.supports(rs.option.enable_auto_exposure):
            sensor.set_option(rs.option.enable_auto_exposure,
                              1 if self.auto_exposure else 0)
        # 白平衡
        if self.white_balance is not None:
            if sensor.supports(rs.option.enable_auto_white_balance):
                sensor.set_option(rs.option.enable_auto_white_balance, 0)
            if sensor.supports(rs.option.white_balance):
                sensor.set_option(rs.option.white_balance, self.white_balance)

    def start(self) -> None:
        profile = self._pipeline.start(self._config)
        sensor = _find_color_sensor(profile)        # 不写死下标，D405/D435i 通用
        self._apply_options(sensor)
        # 丢掉前几帧（曝光稳定期）
        for _ in range(15):
            self._pipeline.wait_for_frames()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            try:
                frames = self._pipeline.wait_for_frames(timeout_ms=1000)
            except RuntimeError:
                continue
            color = frames.get_color_frame()
            if not color:
                continue
            # ⚠️ asanyarray 得到的是 RealSense 帧缓冲的**视图**，不是拷贝。
            # frames 对象每轮循环就出作用域，缓冲随时可能被回收复用 ——
            # 那样已存进 _latest 的"上一帧"会被静默改掉（串帧/撕裂）。
            # cv2.rotate 本身会产生新数组，但 rotate=0 时不走那条路，必须显式拷。
            img = np.array(color.get_data(), dtype=np.uint8, copy=True)   # BGR
            if self._rotate_code is not None:
                img = cv2.rotate(img, self._rotate_code)
            with self._lock:
                self._latest = img
                self._stamp = time.perf_counter()
                self._n_frames += 1

    def latest(self) -> tuple[np.ndarray | None, float]:
        """返回 (最新帧 BGR, 距今秒数)。"""
        with self._lock:
            if self._latest is None:
                return None, float("inf")
            return self._latest.copy(), time.perf_counter() - self._stamp

    # ---------- 与 TactileXense 统一的「源」接口 ----------
    # 触觉是「一个源 → 两路图」，相机是「一个源 → 一路图」。
    # 统一成 dict 返回，主循环就不需要为触觉写特例分支。

    def latest_images(self) -> tuple[dict[str, np.ndarray] | None, float]:
        """返回 ({图像路名: BGR}, 距今秒数)。相机只有一路。"""
        img, age = self.latest()
        return (None if img is None else {self.name: img}), age

    @property
    def image_shapes(self) -> dict[str, tuple[int, int, int]]:
        """这个源产出的所有图像路及其 shape。

        注意用旋转**之后**的尺寸：rotate 90/270 时宽高互换，
        写进 info.json 的必须和实际 mp4 一致。
        """
        h, w = (self.width, self.height) if self._rotate_code in (
            cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE
        ) else (self.height, self.width)
        return {self.name: (h, w, 3)}

    def wait_first_frame(self, timeout: float = 5.0) -> np.ndarray:
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout:
            img, age = self.latest()
            if img is not None and age < 0.5:
                return img
            time.sleep(0.02)
        tag = f"（serial={self.serial}）" if self.serial else ""
        raise RuntimeError(f"相机取不到帧{tag}，检查连接/USB")

    @property
    def n_frames(self) -> int:
        return self._n_frames

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        try:
            self._pipeline.stop()
        except RuntimeError:
            pass

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()


if __name__ == "__main__":
    # 单测：列出设备并逐台取 30 帧，报告实际帧率
    devs = list_devices()
    print(f"检测到 {len(devs)} 台 RealSense：")
    for name, serial in devs:
        print(f"  - {name}  serial={serial}")
    for name, serial in devs:
        print(f"\n== 测试 {name} (serial={serial}) ==")
        with CameraD435i(serial=serial) as cam:
            cam.wait_first_frame()
            t0 = time.perf_counter()
            n0 = cam.n_frames
            time.sleep(2.0)
            img, age = cam.latest()
            dt = time.perf_counter() - t0
            print(f"   ✅ 实际帧率: {(cam.n_frames - n0) / dt:.1f} fps")
            print(f"   最新帧: shape={img.shape} dtype={img.dtype} 距今 {age*1000:.0f}ms")
