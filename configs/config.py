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
