#!/usr/bin/env python3
"""手写 LeRobot v2.1 数据集写入器（不依赖 lerobot 库）。

只用 pandas + pyarrow + opencv，产出 pi0/openpi 直接可读的 v2.1 格式。
数据格式与操作说明见 README.md。

磁盘结构：
    <root>/
    ├── data/chunk-000/episode_{i:06d}.parquet
    ├── videos/chunk-000/observation.images.<cam>/episode_{i:06d}.mp4
    └── meta/{info.json, episodes.jsonl, tasks.jsonl, episodes_stats.jsonl, stats.json}

用法（也可脱离机械臂用假数据单测，见文件末尾 __main__）：
    w = LeRobotV21Writer(root, fps=15,
                         state_dim=8, action_dim=8,
                         cameras={"wrist": (480, 640, 3)},
                         robot_type="franka")
    w.start_episode()
    for ...:
        w.add_frame(state=s, action=a, images={"wrist": bgr}, task="put the cube on the plate")
    w.save_episode()      # 写这条 episode 的 parquet + mp4 + 累积 stats
    ...
    w.finalize()          # 写/更新 info.json、stats.json、各 meta
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd

CODEBASE_VERSION = "v2.1"
CHUNKS_SIZE = 1000
VIDEO_CODEC = "h264"       # libx264，避免 AV1 解码依赖
VIDEO_PIX_FMT = "yuv420p"

# openpi 约定：proprio 存 observation.state，动作存 action
STATE_KEY = "observation.state"
ACTION_KEY = "action"


def _img_key(cam: str) -> str:
    return f"observation.images.{cam}"


class LeRobotV21Writer:
    def __init__(
        self,
        root: str | Path,
        *,
        fps: int = 15,
        state_dim: int = 8,
        action_dim: int = 8,
        cameras: dict[str, tuple[int, int, int]],   # {name: (H, W, C)}
        robot_type: str = "franka",
        state_names: Optional[list[str]] = None,
        action_names: Optional[list[str]] = None,
    ):
        self.root = Path(root)
        self.fps = int(fps)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.cameras = cameras
        self.robot_type = robot_type
        self.state_names = state_names or [f"s{i}" for i in range(state_dim)]
        self.action_names = action_names or [f"a{i}" for i in range(action_dim)]

        (self.root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
        for cam in cameras:
            (self.root / "videos" / "chunk-000" / _img_key(cam)).mkdir(
                parents=True, exist_ok=True
            )
        (self.root / "meta").mkdir(parents=True, exist_ok=True)

        # 载入已有进度（支持续采）
        self._episode_index = self._scan_existing_episodes()
        self._global_index = self._scan_global_index()
        self._tasks: dict[str, int] = self._load_tasks()

        # 当前 episode 的缓冲
        self._buf_state: list[np.ndarray] = []
        self._buf_action: list[np.ndarray] = []
        self._buf_images: dict[str, list[np.ndarray]] = {c: [] for c in cameras}
        self._buf_task: Optional[str] = None
        self._recording = False

    # ---------- 续采扫描 ----------

    def _scan_existing_episodes(self) -> int:
        files = list((self.root / "data" / "chunk-000").glob("episode_*.parquet"))
        return len(files)

    def _scan_global_index(self) -> int:
        total = 0
        for f in sorted((self.root / "data" / "chunk-000").glob("episode_*.parquet")):
            total += len(pd.read_parquet(f, columns=["frame_index"]))
        return total

    def _load_tasks(self) -> dict[str, int]:
        p = self.root / "meta" / "tasks.jsonl"
        tasks: dict[str, int] = {}
        if p.exists():
            for line in p.read_text().splitlines():
                if line.strip():
                    d = json.loads(line)
                    tasks[d["task"]] = d["task_index"]
        return tasks

    # ---------- episode 录制 ----------

    def start_episode(self) -> None:
        self._buf_state.clear()
        self._buf_action.clear()
        for c in self.cameras:
            self._buf_images[c].clear()
        self._buf_task = None
        self._recording = True

    def add_frame(
        self,
        *,
        state: np.ndarray,
        action: np.ndarray,
        images: dict[str, np.ndarray],   # {cam: BGR uint8 (H,W,3)}
        task: str,
    ) -> None:
        if not self._recording:
            raise RuntimeError("先调用 start_episode()")
        state = np.asarray(state, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)
        assert state.shape == (self.state_dim,), f"state 应为 ({self.state_dim},)"
        assert action.shape == (self.action_dim,), f"action 应为 ({self.action_dim},)"
        assert set(images) == set(self.cameras), "images 的相机与初始化不一致"

        self._buf_state.append(state)
        self._buf_action.append(action)
        for c in self.cameras:
            self._buf_images[c].append(np.ascontiguousarray(images[c]))
        if self._buf_task is None:
            self._buf_task = task
        elif self._buf_task != task:
            raise ValueError("同一 episode 内 task 不应改变")

    def discard_episode(self) -> None:
        """丢弃当前缓冲，不写盘。"""
        self._recording = False
        self._buf_state.clear()
        self._buf_action.clear()
        for c in self.cameras:
            self._buf_images[c].clear()

    @property
    def n_buffered(self) -> int:
        return len(self._buf_state)

    def save_episode(self) -> int:
        """把当前缓冲写成一条 episode，返回它的 episode_index。"""
        if not self._recording or not self._buf_state:
            raise RuntimeError("没有可保存的帧")
        ep = self._episode_index
        n = len(self._buf_state)

        # 1. parquet
        self._write_parquet(ep, n)
        # 2. 每个相机一个 mp4
        for cam in self.cameras:
            self._write_video(ep, cam)
        # 3. meta 增量更新
        task_index = self._register_task(self._buf_task)
        self._append_episodes_jsonl(ep, self._buf_task, n)
        self._append_episodes_stats(ep)

        self._episode_index += 1
        self._global_index += n
        self._recording = False
        return ep

    # ---------- 落盘细节 ----------

    def _write_parquet(self, ep: int, n: int) -> None:
        states = np.stack(self._buf_state)     # (n, state_dim)
        actions = np.stack(self._buf_action)   # (n, action_dim)
        # LeRobot 每帧向量存成 list（parquet 里是变长 list<float>）
        df = pd.DataFrame({
            STATE_KEY: list(states.astype(np.float32)),
            ACTION_KEY: list(actions.astype(np.float32)),
            "timestamp": (np.arange(n) / self.fps).astype(np.float32),
            "frame_index": np.arange(n, dtype=np.int64),
            "episode_index": np.full(n, ep, dtype=np.int64),
            "index": np.arange(self._global_index, self._global_index + n, dtype=np.int64),
            "task_index": np.full(n, self._peek_task_index(self._buf_task), dtype=np.int64),
        })
        path = self.root / "data" / "chunk-000" / f"episode_{ep:06d}.parquet"
        df.to_parquet(path, index=False, engine="pyarrow")

    def _write_video(self, ep: int, cam: str) -> None:
        """用 ffmpeg libx264 软件编码，产出 h264 + yuv420p。

        不用 opencv 的 VideoWriter —— 它在 Linux 上会走硬件编码器
        (h264_v4l2m2m)，无专用芯片时失败并静默回退到 mp4v，
        导致 info.json 里声明的 codec 与实际不符。
        ffmpeg 管道稳定产出真正的 h264。
        """
        frames = self._buf_images[cam]
        h, w = frames[0].shape[:2]
        path = (self.root / "videos" / "chunk-000" / _img_key(cam)
                / f"episode_{ep:06d}.mp4")
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{w}x{h}", "-r", str(self.fps),
            "-i", "-",
            "-an",
            "-vcodec", "libx264", "-pix_fmt", VIDEO_PIX_FMT,
            "-crf", "23",              # 视觉近无损，体积合理
            # 限制编码线程：保存 4 路视频是串行的，不限的话每路都会吃满所有核，
            # 和 1kHz 控制环抢 CPU（开发记录第 0/9 节：曾把
            # control_command_success_rate 打到 0.14）。
            "-threads", "2",
            str(path),
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for f in frames:
            proc.stdin.write(np.ascontiguousarray(f, dtype=np.uint8).tobytes())
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError(f"ffmpeg 编码失败: {path}")

    def _peek_task_index(self, task: str) -> int:
        if task in self._tasks:
            return self._tasks[task]
        return len(self._tasks)   # 预测将分配的 index（_register_task 会真正加入）

    def _register_task(self, task: str) -> int:
        if task not in self._tasks:
            self._tasks[task] = len(self._tasks)
            self._rewrite_tasks_jsonl()
        return self._tasks[task]

    def _rewrite_tasks_jsonl(self) -> None:
        p = self.root / "meta" / "tasks.jsonl"
        with open(p, "w") as f:
            for task, idx in sorted(self._tasks.items(), key=lambda kv: kv[1]):
                f.write(json.dumps({"task_index": idx, "task": task}) + "\n")

    def _append_episodes_jsonl(self, ep: int, task: str, n: int) -> None:
        p = self.root / "meta" / "episodes.jsonl"
        with open(p, "a") as f:
            f.write(json.dumps({"episode_index": ep, "tasks": [task], "length": n}) + "\n")

    def _append_episodes_stats(self, ep: int) -> None:
        stats = self._compute_episode_stats()
        p = self.root / "meta" / "episodes_stats.jsonl"
        with open(p, "a") as f:
            f.write(json.dumps({"episode_index": ep, "stats": stats}) + "\n")

    def _compute_episode_stats(self) -> dict:
        """计算当前 episode 各 feature 的 mean/std/min/max/count。

        向量特征 shape (dim,1)，图像 per-channel shape (3,1,1)。
        openpi 训练会重算 norm stats，这里只需 schema 合法、shape 对。
        """
        out = {}
        for key, arr in [
            (STATE_KEY, np.stack(self._buf_state)),
            (ACTION_KEY, np.stack(self._buf_action)),
        ]:
            out[key] = self._vec_stats(arr)   # (n, dim)
        for cam in self.cameras:
            out[_img_key(cam)] = self._img_stats(self._buf_images[cam])
        return out

    @staticmethod
    def _vec_stats(arr: np.ndarray) -> dict:
        # arr: (n, dim) -> 每个 stat shape (dim, 1)
        a = arr.astype(np.float64)
        return {
            "mean": a.mean(0, keepdims=False).reshape(-1, 1).tolist(),
            "std": a.std(0, keepdims=False).reshape(-1, 1).tolist(),
            "min": a.min(0).reshape(-1, 1).tolist(),
            "max": a.max(0).reshape(-1, 1).tolist(),
            "count": [len(arr)],
        }

    @staticmethod
    def _img_stats(frames: list[np.ndarray]) -> dict:
        # per-channel，归一化到 [0,1]，shape (3,1,1)；采样至多 100 帧省时
        idx = np.linspace(0, len(frames) - 1, min(len(frames), 100)).astype(int)
        sample = np.stack([frames[i] for i in idx]).astype(np.float64) / 255.0
        # sample: (k, H, W, 3) -> per channel
        ch = sample.reshape(-1, sample.shape[-1])   # (k*H*W, 3)
        def r(x):  # -> (3,1,1)
            return x.reshape(3, 1, 1).tolist()
        return {
            "mean": r(ch.mean(0)),
            "std": r(ch.std(0)),
            "min": r(ch.min(0)),
            "max": r(ch.max(0)),
            "count": [len(frames)],
        }

    # ---------- 收尾：info.json + 全局 stats.json ----------

    def finalize(self) -> None:
        self._write_info_json()
        self._write_global_stats()

    def _write_info_json(self) -> None:
        n_ep = self._episode_index
        total_frames = self._global_index
        features = {
            _img_key(cam): {
                "dtype": "video",
                "shape": list(shape),
                "names": ["height", "width", "channel"],
                "info": {
                    "video.height": shape[0], "video.width": shape[1],
                    "video.codec": VIDEO_CODEC, "video.pix_fmt": VIDEO_PIX_FMT,
                    "video.is_depth_map": False, "video.fps": self.fps,
                    "video.channels": shape[2], "has_audio": False,
                },
            }
            for cam, shape in self.cameras.items()
        }
        features[STATE_KEY] = {"dtype": "float32", "shape": [self.state_dim],
                               "names": self.state_names}
        features[ACTION_KEY] = {"dtype": "float32", "shape": [self.action_dim],
                                "names": self.action_names}
        features["timestamp"] = {"dtype": "float32", "shape": [1], "names": None}
        features["frame_index"] = {"dtype": "int64", "shape": [1], "names": None}
        features["episode_index"] = {"dtype": "int64", "shape": [1], "names": None}
        features["index"] = {"dtype": "int64", "shape": [1], "names": None}
        features["task_index"] = {"dtype": "int64", "shape": [1], "names": None}

        info = {
            "codebase_version": CODEBASE_VERSION,
            "robot_type": self.robot_type,
            "total_episodes": n_ep,
            "total_frames": total_frames,
            "total_tasks": len(self._tasks),
            "total_chunks": 1,
            "total_videos": n_ep * len(self.cameras),
            "chunks_size": CHUNKS_SIZE,
            "fps": self.fps,
            "splits": {"train": f"0:{n_ep}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": features,
        }
        (self.root / "meta" / "info.json").write_text(json.dumps(info, indent=2))

    def _write_global_stats(self) -> None:
        """把各 episode 的 stats 加权聚合成全局 stats.json。"""
        p = self.root / "meta" / "episodes_stats.jsonl"
        if not p.exists():
            return
        per_ep = [json.loads(l)["stats"] for l in p.read_text().splitlines() if l.strip()]
        if not per_ep:
            return
        keys = per_ep[0].keys()
        agg = {}
        for k in keys:
            counts = np.array([np.asarray(e[k]["count"]).ravel()[0] for e in per_ep], dtype=np.float64)
            total = counts.sum()
            means = np.array([e[k]["mean"] for e in per_ep], dtype=np.float64)   # (E, ...)
            stds = np.array([e[k]["std"] for e in per_ep], dtype=np.float64)
            mins = np.array([e[k]["min"] for e in per_ep], dtype=np.float64)
            maxs = np.array([e[k]["max"] for e in per_ep], dtype=np.float64)
            w = (counts / total).reshape(-1, *([1] * (means.ndim - 1)))
            gmean = (means * w).sum(0)
            # 合并方差：E[var] + var(mean)
            gvar = ((stds ** 2 + (means - gmean) ** 2) * w).sum(0)
            agg[k] = {
                "mean": gmean.tolist(),
                "std": np.sqrt(gvar).tolist(),
                "min": mins.min(0).tolist(),
                "max": maxs.max(0).tolist(),
                "count": [int(total)],
            }
        (self.root / "meta" / "stats.json").write_text(json.dumps(agg, indent=2))


# ---------- 自测：用假数据生成一个数据集，验证格式 ----------

if __name__ == "__main__":
    import shutil
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="lerobot_test_"))
    print(f"测试输出目录: {tmp}\n")

    w = LeRobotV21Writer(
        tmp, fps=15, state_dim=8, action_dim=8,
        cameras={"wrist": (480, 640, 3)}, robot_type="franka",
        state_names=[f"joint_{i}" for i in range(7)] + ["gripper"],
        action_names=[f"joint_{i}" for i in range(7)] + ["gripper"],
    )

    rng = np.random.RandomState(0)
    for ep in range(2):
        w.start_episode()
        n = 20 + ep * 5
        for t in range(n):
            w.add_frame(
                state=rng.rand(8).astype(np.float32),
                action=rng.rand(8).astype(np.float32),
                images={"wrist": (rng.rand(480, 640, 3) * 255).astype(np.uint8)},
                task="put the cube on the plate",
            )
        idx = w.save_episode()
        print(f"episode {idx}: {n} 帧已保存")
    w.finalize()

    print("\n=== 生成的文件 ===")
    for f in sorted(tmp.rglob("*")):
        if f.is_file():
            print(f"  {f.relative_to(tmp)}  ({f.stat().st_size} B)")

    print("\n=== 校验 ===")
    info = json.loads((tmp / "meta" / "info.json").read_text())
    print("  codebase_version:", info["codebase_version"])
    print("  total_episodes:", info["total_episodes"], " total_frames:", info["total_frames"])
    df = pd.read_parquet(tmp / "data" / "chunk-000" / "episode_000000.parquet")
    print("  parquet 列:", list(df.columns))
    print("  state[0] shape:", np.asarray(df[STATE_KEY].iloc[0]).shape)
    print("  index 范围:", df["index"].min(), "-", df["index"].max())
    cap = cv2.VideoCapture(str(tmp / "videos" / "chunk-000"
                               / "observation.images.wrist" / "episode_000000.mp4"))
    print("  视频帧数:", int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
          " 分辨率:", int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "x",
          int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    print(f"\n清理: rm -rf {tmp}")
