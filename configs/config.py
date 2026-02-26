"""
config.py

Configuration settings for robosuite data collection.
"""

import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from pathlib import Path


@dataclass
class DataCollectionConfig:
    """Configuration for robosuite data collection."""
    
    # Environment settings
    env_name: str = "PickPlaceCan"  # robosuite environment name
    robots: str = "Panda"  # Robot model: Panda, Sawyer, IIWA, etc.
    controller_config: Optional[Dict[str, Any]] = None
    # controller_type: 控制器类型简称，用于快速选择预设控制器。
    #   - null / "OSC_POSE"（默认）: 操作空间控制，输入为归一化的笛卡尔增量 (dx,dy,dz,drx,dry,drz)
    #   - "JOINT_POSITION": 关节位置控制，输入为目标关节角度
    #   - "JOINT_VELOCITY": 关节速度控制
    #   - "JOINT_TORQUE": 关节扭矩控制
    #   - "IK_POSE": 逆运动学控制器（仅支持 Panda/Sawyer/Baxter，UR5e 不支持）
    #   - "OSC_POSITION": 仅位置的操作空间控制（无姿态控制）
    # 若同时指定了 controller_config（完整 dict），则 controller_config 优先。
    controller_type: Optional[str] = None
    # controller_params: 控制器参数覆盖（可选）。
    # 仅当 controller_type 被指定且 controller_config 为 null 时生效。
    # 用于覆盖对应控制器类型的默认参数，例如：
    #   {"kp": 200, "output_max": [0.1, 0.1, 0.1, 1.0, 1.0, 1.0]}
    controller_params: Optional[Dict[str, Any]] = None
    has_renderer: bool = False  # Set to True for visualization
    has_offscreen_renderer: bool = True  # Required for camera observations
    use_camera_obs: bool = True
    use_object_obs: bool = True
    reward_shaping: bool = True
    control_freq: int = 20  # Control frequency in Hz
    horizon: int = 500  # Maximum episode length
    
    # Camera settings
    camera_names: List[str] = field(default_factory=lambda: ["agentview", "robot0_eye_in_hand"])
    camera_heights: int = 224
    camera_widths: int = 224
    camera_depths: bool = False  # Include depth images
    
    # Data collection settings
    num_episodes: int = 100
    output_dir: str = "data/robosuite_collected"
    save_format: str = "hdf5"  # "hdf5", "pickle", or "rlds"
    language_instruction: str = ""  # Task language instruction (required for rlds format)
    
    # Policy settings
    policy_type: str = "random"  # "random", "scripted", "human", "trajectory_gen"
    use_scripted_policy: bool = False
    
    # Trajectory generation settings (MimicGen-style)
    use_trajectory_generation: bool = False
    source_demo_path: Optional[str] = None  # Path to source HDF5 demo file
    
    # Speed limit settings (normalized action space, range [0, 1])
    # 归一化 action 的各分量绝对值不得超过此阈值，用于预处理和运行时限距检测
    limit_dpos: float = 0.85  # 归一化后位置增量上限
    limit_drot: float = 1.0   # 归一化后旋转增量上限
    
    # Additional robosuite settings
    ignore_done: bool = False
    hard_reset: bool = True
    
    # Reproducibility
    seed: Optional[int] = None  # Random seed for reproducibility (None = no seed)
    
    # Custom environment settings
    base_path: Optional[str] = None  # Base path for custom assets (e.g., external XML models)
    
    @classmethod
    def from_json(cls, json_path: str) -> "DataCollectionConfig":
        """
        Load configuration from JSON file.
        
        Args:
            json_path: Path to JSON configuration file
            
        Returns:
            DataCollectionConfig instance
        """
        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"Config file not found: {json_path}")
        
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        
        # Handle nested controller config
        if 'controller_config' in config_dict and config_dict['controller_config']:
            # Keep as dict
            pass
        
        return cls(**config_dict)
    
    def to_json(self, json_path: str):
        """
        Save configuration to JSON file.
        
        Args:
            json_path: Path to save JSON configuration
        """
        json_path = Path(json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = asdict(self)
        
        with open(json_path, 'w') as f:
            json.dump(config_dict, f, indent=4)
        
        print(f"Configuration saved to: {json_path}")
    
    def to_env_kwargs(self) -> Dict[str, Any]:
        """Convert config to robosuite environment kwargs."""
        env_kwargs = {
            "robots": self.robots,
            "has_renderer": self.has_renderer,
            "has_offscreen_renderer": self.has_offscreen_renderer,
            "use_camera_obs": self.use_camera_obs,
            "use_object_obs": self.use_object_obs,
            "reward_shaping": self.reward_shaping,
            "control_freq": self.control_freq,
            "horizon": self.horizon,
            "ignore_done": self.ignore_done,
            "hard_reset": self.hard_reset,
        }
        
        if self.use_camera_obs:
            env_kwargs.update({
                "camera_names": self.camera_names,
                "camera_heights": self.camera_heights,
                "camera_widths": self.camera_widths,
                "camera_depths": self.camera_depths,
            })
        
        # Add base_path if specified (for custom environments with external assets)
        if self.base_path is not None:
            env_kwargs["base_path"] = self.base_path
        
        # Note: controller_config is handled separately in RoboSuiteDataCollector
        # to use load_composite_controller_config function
            
        return env_kwargs
