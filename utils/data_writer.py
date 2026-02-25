"""
data_writer.py

Utilities for writing collected data to disk in various formats.

Supported formats:
    - hdf5:  原始 HDF5 格式（旧格式，用于调试）
    - pickle: Pickle 格式（每个 episode 一个文件）
    - rlds:  RLDS/TFDS 格式（可直接被 VLA-Adapter 训练读取）
"""

import os
import h5py
import pickle
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import random


class DataWriter:
    """Base class for data writers."""
    
    def __init__(self, output_dir: str, env_name: str):
        """
        Initialize data writer.
        
        Args:
            output_dir: Directory to save data
            env_name: Environment name
        """
        self.output_dir = Path(output_dir)
        self.env_name = env_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._logger = None
    
    def set_logger(self, logger: logging.Logger):
        """设置外部 logger，之后所有日志通过该 logger 实时写入文件。"""
        self._logger = logger
    
    def _log(self, msg: str, level: str = "info"):
        """统一日志输出：同时 print 到终端 + 写入 logger 文件。"""
        print(msg)
        if self._logger:
            getattr(self._logger, level)(msg)
        
    def write_episode(self, episode_data: Dict[str, Any], episode_idx: int):
        """Write a single episode to disk."""
        raise NotImplementedError
    
    def finalize(self):
        """Finalize writing (e.g., close files)."""
        pass


class HDF5DataWriter(DataWriter):
    """Writer for HDF5 format."""
    
    def __init__(self, output_dir: str, env_name: str):
        super().__init__(output_dir, env_name)
        self.hdf5_path = self.output_dir / f"{env_name}_data.hdf5"
        self.hdf5_file = h5py.File(self.hdf5_path, "w")
        self.episodes_written = 0
        
    def write_episode(self, episode_data: Dict[str, Any], episode_idx: int):
        """
        Write a single episode to HDF5 file.
        
        Args:
            episode_data: Dictionary containing episode data with keys:
                - observations: List of observation dicts
                - actions: List of action arrays
                - rewards: List of reward scalars
                - dones: List of done flags
                - info: Episode info dict
            episode_idx: Episode index
        """
        grp = self.hdf5_file.create_group(f"episode_{episode_idx}")
        
        # Write actions
        actions = np.array(episode_data["actions"])
        grp.create_dataset("actions", data=actions, compression="gzip")
        
        # Write rewards
        rewards = np.array(episode_data["rewards"])
        grp.create_dataset("rewards", data=rewards, compression="gzip")
        
        # Write dones
        dones = np.array(episode_data["dones"], dtype=bool)
        grp.create_dataset("dones", data=dones, compression="gzip")
        
        # Write observations
        obs_grp = grp.create_group("observations")
        
        # Get first observation to determine structure
        first_obs = episode_data["observations"][0]
        
        # Handle state observations
        if "state" in first_obs:
            states = np.array([obs["state"] for obs in episode_data["observations"]])
            obs_grp.create_dataset("state", data=states, compression="gzip")
        
        # Handle image observations
        for key in first_obs.keys():
            if key.endswith("_image"):
                images = np.array([obs[key] for obs in episode_data["observations"]])
                obs_grp.create_dataset(key, data=images, compression="gzip")
            elif key.endswith("_depth"):
                depths = np.array([obs[key] for obs in episode_data["observations"]])
                obs_grp.create_dataset(key, data=depths, compression="gzip")
        
        # Write metadata
        grp.attrs["episode_length"] = len(episode_data["actions"])
        grp.attrs["total_reward"] = np.sum(rewards)
        grp.attrs["success"] = episode_data.get("success", False)
        
        self.episodes_written += 1
        
    def finalize(self):
        """Close HDF5 file."""
        # Write global metadata
        self.hdf5_file.attrs["num_episodes"] = self.episodes_written
        self.hdf5_file.attrs["env_name"] = self.env_name
        
        self.hdf5_file.close()
        self._log(f"Saved {self.episodes_written} episodes to {self.hdf5_path}")


class PickleDataWriter(DataWriter):
    """Writer for pickle format (separate file per episode)."""
    
    def write_episode(self, episode_data: Dict[str, Any], episode_idx: int):
        """
        Write a single episode to pickle file.
        
        Args:
            episode_data: Episode data dictionary
            episode_idx: Episode index
        """
        episode_path = self.output_dir / f"{self.env_name}_ep{episode_idx:04d}.pkl"
        
        with open(episode_path, "wb") as f:
            pickle.dump(episode_data, f)
    
    def finalize(self):
        """Write metadata file."""
        metadata_path = self.output_dir / "metadata.json"
        metadata = {
            "env_name": self.env_name,
            "format": "pickle",
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)


class RLDSDataWriter(DataWriter):
    """
    Writer for RLDS/TFDS format, compatible with VLA-Adapter training pipeline.
    
    生成的数据结构遵循 LIBERO RLDS 格式：
      - 每条轨迹包含 steps，每个 step 有:
        observation/image (H, W, 3) uint8       — 第三人称相机 (agentview)
        observation/wrist_image (H, W, 3) uint8  — 腕部相机 (robot0_eye_in_hand)
        observation/state (8,) float32           — EEF位姿(6) + gripper_qpos(2)
        action (7,) float32                      — delta_pos(3) + delta_rot(3) + gripper(1)
        language_instruction: string             — 任务语言指令
    
    同时保存：
      - 每个 episode 两个视角的 mp4 视频
      - 随机挑选一个 episode 保存为检查用 HDF5
    
    注意:
      - action 是控制器归一化后的输入（与 LIBERO 一致）
      - 图像会翻转（robosuite OpenGL 坐标系 → 标准图像坐标系）
      - state 中的旋转使用 axis-angle 表示（与 LIBERO 一致）
    """
    
    def __init__(self, output_dir: str, env_name: str,
                 language_instruction: str = "",
                 image_key_primary: str = "agentview_image",
                 image_key_wrist: str = "robot0_eye_in_hand_image",
                 dataset_name: Optional[str] = None,
                 save_video: bool = True,
                 video_fps: int = 20):
        """
        Args:
            output_dir: 输出根目录
            env_name: 环境名称
            language_instruction: 任务语言指令（所有 episode 共用）
            image_key_primary: 主相机图像在 obs 中的 key
            image_key_wrist: 腕部相机图像在 obs 中的 key
            dataset_name: TFDS 数据集名称（默认使用 env_name 小写 + "_no_noops"）
            save_video: 是否保存渲染视频
            video_fps: 视频帧率
        """
        super().__init__(output_dir, env_name)
        
        self.language_instruction = language_instruction
        self.image_key_primary = image_key_primary
        self.image_key_wrist = image_key_wrist
        self.dataset_name = dataset_name or f"{env_name.lower()}_generated"
        self.save_video = save_video
        self.video_fps = video_fps
        
        # 收集所有 episodes 的数据，用于最后统一写入 TFRecord
        self.all_episodes = []
        self.episodes_written = 0
        
        # 随机选择一个 episode 用于检查（在 finalize 中确定）
        self._check_episode_idx = None
        
        # 创建视频输出目录
        if self.save_video:
            self.video_dir = self.output_dir / "videos"
            self.video_dir.mkdir(parents=True, exist_ok=True)
    
    def _extract_rlds_data(self, episode_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        从 collect_episode 的输出中提取 RLDS 格式的数据。
        
        关键对齐逻辑:
            observations 有 T+1 个（obs_0, obs_1, ..., obs_T），actions 有 T 个。
            RLDS 每个 step_t 对应 (obs_t, action_t)，共 T 个 step。
            最后一个 obs（obs_T）不对应任何 action，丢弃。
        
        Args:
            episode_data: collect_episode 返回的字典
            
        Returns:
            dict with keys: images, wrist_images, states, actions, language_instruction
        """
        import robosuite.utils.transform_utils as T
        
        observations = episode_data["observations"]
        actions = np.array(episode_data["actions"])  # (T, 7)
        num_steps = len(actions)
        
        # 对齐：只取前 T 个 observation（丢弃最后一个 next_obs）
        obs_aligned = observations[:num_steps]
        
        images = []
        wrist_images = []
        states = []
        
        for obs in obs_aligned:
            raw_obs = obs.get("raw_obs", obs)
            
            # 提取图像并翻转（robosuite OpenGL → 标准坐标系）
            primary_img = raw_obs.get(self.image_key_primary)
            if primary_img is not None:
                primary_img = np.flipud(primary_img).copy()
            else:
                raise ValueError(f"主相机图像 key '{self.image_key_primary}' 不存在于 obs 中。"
                               f"可用 keys: {list(raw_obs.keys())}")
            
            wrist_img = raw_obs.get(self.image_key_wrist)
            if wrist_img is not None:
                wrist_img = np.flipud(wrist_img).copy()
            else:
                raise ValueError(f"腕部相机图像 key '{self.image_key_wrist}' 不存在于 obs 中。"
                               f"可用 keys: {list(raw_obs.keys())}")
            
            images.append(primary_img)
            wrist_images.append(wrist_img)
            
            # 提取 state: eef_pos(3) + axisangle(3) + gripper_qpos(2) = 8D
            eef_pos = raw_obs.get("robot0_eef_pos", np.zeros(3))
            eef_quat = raw_obs.get("robot0_eef_quat", np.array([0., 0., 0., 1.]))
            
            # gripper_qpos_full为6维，且[0:3] = [3:6]对称分布，判断夹爪开闭只看gripper_qpos_full
            # gripper_qpos_full < 0为开,gripper_qpos_full > 0 为闭
            # 注意 -1张开 1为闭合
            gripper_qpos_full = raw_obs.get("robot0_gripper_qpos", np.zeros(6))
            gripper_qpos = np.zeros(2)
            gripper_qpos[1] = -1 if gripper_qpos_full[0] < 0 else 1

            # 四元数转欧拉角（与 LIBERO regenerate 脚本一致）
            eef_axisangle = T.quat2axisangle(eef_quat)
            
            state = np.concatenate([
                eef_pos,          # [0:3] EEF xyz
                eef_axisangle,    # [3:6] EEF axis-angle rotation
                gripper_qpos,     # [6:8] gripper joint positions (2D)
            ]).astype(np.float32)
            states.append(state)
        
        return {
            "images": np.array(images, dtype=np.uint8),           # (T, H, W, 3)
            "wrist_images": np.array(wrist_images, dtype=np.uint8),  # (T, H, W, 3)
            "states": np.array(states, dtype=np.float32),          # (T, 8)
            "actions": actions.astype(np.float32),                  # (T, 7)
            "language_instruction": self.language_instruction,
        }
    
    def _save_video(self, images: np.ndarray, path: str):
        """保存图像序列为 mp4 视频。"""
        import cv2
        
        h, w = images.shape[1], images.shape[2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(path), fourcc, self.video_fps, (w, h))
        
        for img in images:
            # RGB → BGR for OpenCV
            writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
        writer.release()
    
    def write_episode(self, episode_data: Dict[str, Any], episode_idx: int):
        """
        提取 RLDS 格式数据，暂存到内存中，并保存视频。
        """
        rlds_data = self._extract_rlds_data(episode_data)
        self.all_episodes.append(rlds_data)
        self.episodes_written += 1
        
        # 保存视频
        if self.save_video:
            self._save_video(
                rlds_data["images"],
                self.video_dir / f"ep{episode_idx:04d}_agentview.mp4"
            )
            self._save_video(
                rlds_data["wrist_images"],
                self.video_dir / f"ep{episode_idx:04d}_eye_in_hand.mp4"
            )
    
    def _save_check_hdf5(self, rlds_data: Dict[str, Any], episode_idx: int):
        """
        将单个 episode 的 RLDS 格式数据保存为检查用 HDF5。
        结构完全对应 LIBERO RLDS schema，方便直接查看验证。
        
        HDF5 结构：
            /observation/image          (T, H, W, 3)  uint8
            /observation/wrist_image    (T, H, W, 3)  uint8
            /observation/state          (T, 8)         float32
            /action                     (T, 7)         float32
            /language_instruction       scalar string
        """
        check_dir = self.output_dir / "check"
        check_dir.mkdir(parents=True, exist_ok=True)
        check_path = check_dir / f"check_episode_{episode_idx}.hdf5"
        
        with h5py.File(check_path, "w") as f:
            obs_grp = f.create_group("observation")
            obs_grp.create_dataset("image", data=rlds_data["images"], compression="gzip")
            obs_grp.create_dataset("wrist_image", data=rlds_data["wrist_images"], compression="gzip")
            obs_grp.create_dataset("state", data=rlds_data["states"], compression="gzip")
            f.create_dataset("action", data=rlds_data["actions"], compression="gzip")
            f.attrs["language_instruction"] = rlds_data["language_instruction"]
            
            # 同时记录维度信息便于检查
            f.attrs["num_steps"] = len(rlds_data["actions"])
            f.attrs["image_shape"] = rlds_data["images"].shape[1:]   # (H, W, 3)
            f.attrs["state_dim"] = rlds_data["states"].shape[1]      # 8
            f.attrs["action_dim"] = rlds_data["actions"].shape[1]    # 7
            f.attrs["state_format"] = "eef_pos(3) + eef_axisangle(3) + gripper_qpos(2)"
            f.attrs["action_format"] = "delta_pos(3) + delta_axisangle(3) + gripper(1)"
        
        self._log(f"✓ 检查 HDF5 已保存: {check_path}")
        self._log(f"  steps={len(rlds_data['actions'])}, "
                  f"image={rlds_data['images'].shape}, "
                  f"state={rlds_data['states'].shape}, "
                  f"action={rlds_data['actions'].shape}")
    
    def _write_tfrecords(self):
        """
        将所有 episodes 通过 TFDS builder API 写入标准 RLDS 格式。
        
        使用 tensorflow_datasets 的 GeneratorBasedBuilder，确保生成的数据
        可以被 tfds.builder_from_directory() 和 dl.DLataset.from_rlds() 正确读取。
        
        生成的目录结构：
            output_dir/
            └── {dataset_name}/
                └── 1.0.0/
                    ├── dataset_info.json
                    ├── features.json
                    └── {dataset_name}-train.tfrecord-*
        """
        import tensorflow_datasets as tfds
        import tensorflow as tf
        import os
        tfds.core.utils.gcs_utils._is_gcs_disabled = True
        os.environ['NO_GCE_CHECK'] = 'true'
 
        
        sample = self.all_episodes[0]
        img_h, img_w = sample["images"].shape[1], sample["images"].shape[2]
        
        # 将 all_episodes 暂存在类中供 builder 的 _generate_examples 访问
        episodes_ref = self.all_episodes
        dataset_name_ref = self.dataset_name
        
        class _InlineBuilder(tfds.core.GeneratorBasedBuilder):
            """内联 TFDS builder，用于将内存中的 episodes 写入标准 RLDS 格式。"""
            
            VERSION = tfds.core.Version("1.0.0")
            
            # 让 builder 使用指定的名字
            name = dataset_name_ref
            
            def _info(self):
                return self.dataset_info_from_configs(
                    features=tfds.features.FeaturesDict({
                        "steps": tfds.features.Dataset({
                            "observation": tfds.features.FeaturesDict({
                                "image": tfds.features.Image(
                                    shape=(img_h, img_w, 3),
                                    dtype=tf.uint8,
                                    encoding_format="png",
                                ),
                                "wrist_image": tfds.features.Image(
                                    shape=(img_h, img_w, 3),
                                    dtype=tf.uint8,
                                    encoding_format="png",
                                ),
                                "state": tfds.features.Tensor(
                                    shape=(8,),
                                    dtype=tf.float32,
                                ),
                            }),
                            "action": tfds.features.Tensor(
                                shape=(7,),
                                dtype=tf.float32,
                            ),
                            "reward": tfds.features.Scalar(dtype=tf.float32),
                            "discount": tfds.features.Scalar(dtype=tf.float32),
                            "is_first": tf.bool,
                            "is_last": tf.bool,
                            "is_terminal": tf.bool,
                            "language_instruction": tfds.features.Text(),
                        }),
                    }),
                )
            
            def _split_generators(self, dl_manager):
                return {
                    "train": self._generate_examples(),
                }
            
            def _generate_examples(self):
                for ep_idx, ep_data in enumerate(episodes_ref):
                    num_steps = len(ep_data["actions"])
                    steps = []
                    for t in range(num_steps):
                        steps.append({
                            "observation": {
                                "image": ep_data["images"][t],
                                "wrist_image": ep_data["wrist_images"][t],
                                "state": ep_data["states"][t],
                            },
                            "action": ep_data["actions"][t],
                            "reward": 1.0 if t == num_steps - 1 else 0.0,
                            "discount": 1.0,
                            "is_first": t == 0,
                            "is_last": t == num_steps - 1,
                            "is_terminal": t == num_steps - 1,
                            "language_instruction": ep_data["language_instruction"],
                        })
                    
                    yield f"episode_{ep_idx}", {"steps": steps}
        
        # 使用 TFDS builder 写入数据
        dataset_dir = str(self.output_dir)
        builder = _InlineBuilder(data_dir=dataset_dir)
        
        # download_and_prepare 会自动创建 TFRecord + metadata
        builder.download_and_prepare()
        
        num_transitions_total = sum(len(ep["actions"]) for ep in self.all_episodes)
        self._log(f"✓ TFDS 数据已保存: {builder.data_dir}")
        self._log(f"  episodes={len(self.all_episodes)}, transitions={num_transitions_total}")
    
    def finalize(self):
        """
        完成数据写入：
        1. 随机挑选一个 episode 保存为检查 HDF5
        2. 写入 TFRecord + metadata
        """
        if not self.all_episodes:
            self._log("⚠ 没有任何 episode 数据，跳过 finalize", level="warning")
            return
        
        # 1. 随机挑选一个 episode 保存检查 HDF5
        self._check_episode_idx = random.randint(0, len(self.all_episodes) - 1)
        self._log(f"随机挑选 episode {self._check_episode_idx} 作为检查数据...")
        self._save_check_hdf5(self.all_episodes[self._check_episode_idx], self._check_episode_idx)
        
        # 2. 写入 TFDS 格式 TFRecord
        self._log(f"正在写入 RLDS/TFDS 格式...")
        self._write_tfrecords()
        
        self._log(f"✓ RLDS 数据写入完成!")
        self._log(f"  数据集名称: {self.dataset_name}")
        self._log(f"  数据目录: {self.output_dir / self.dataset_name}")
        if self.save_video:
            self._log(f"  视频目录: {self.video_dir}")
        self._log(f"  检查文件: {self.output_dir / 'check' / f'check_episode_{self._check_episode_idx}.hdf5'}")


def create_data_writer(output_dir: str, env_name: str, format: str = "hdf5",
                       **kwargs) -> DataWriter:
    """
    Factory function to create appropriate data writer.
    
    Args:
        output_dir: Directory to save data
        env_name: Environment name
        format: Data format ("hdf5", "pickle", or "rlds")
        **kwargs: Additional arguments passed to writer constructor
            For "rlds" format:
                language_instruction: str — 任务语言指令
                image_key_primary: str — 主相机图像 key (default: "agentview_image")
                image_key_wrist: str — 腕部相机图像 key (default: "robot0_eye_in_hand_image")
                dataset_name: str — TFDS 数据集名称
                save_video: bool — 是否保存视频 (default: True)
                video_fps: int — 视频帧率 (default: 20)
        
    Returns:
        DataWriter instance
    """
    if format == "hdf5":
        return HDF5DataWriter(output_dir, env_name)
    elif format == "pickle":
        return PickleDataWriter(output_dir, env_name)
    elif format == "rlds":
        return RLDSDataWriter(output_dir, env_name, **kwargs)
    else:
        raise ValueError(f"Unknown format: {format}. Supported: hdf5, pickle, rlds")
