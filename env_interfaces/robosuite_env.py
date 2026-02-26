"""
robosuite_env.py

Wrapper for robosuite environment with data collection utilities.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from typing import Dict, Any, Tuple, List
import robosuite as suite
from robosuite.wrappers import GymWrapper
from robosuite.controllers import load_composite_controller_config

from configs.config import DataCollectionConfig

# Register custom environments
CUSTOM_ENVS = {}
try:
    from env.pouring_water_env import PouringWater
    CUSTOM_ENVS["PouringWater"] = PouringWater
except ImportError as e:
    print(f"Warning: Could not import custom environments: {e}")

class RoboSuiteDataCollector:
    """
    Wrapper class for robosuite environment with data collection functionality.
    """
    
    def __init__(self, config: DataCollectionConfig):
        """
        Initialize the robosuite environment.
        
        Args:
            config: DataCollectionConfig object with environment settings
        """
        self.config = config
        
        # Create robosuite environment
        env_kwargs = config.to_env_kwargs()
        
        # Handle controller configuration
        # 优先级: controller_config (完整 dict) > controller_type (简称 + 可选 params) > 默认
        controller_configs = self._build_controller_config(config)
        if controller_configs is not None:
            env_kwargs["controller_configs"] = controller_configs
        
        # Check if custom environment
        if config.env_name in CUSTOM_ENVS:
            print(f"Loading custom environment: {config.env_name}")
            env_class = CUSTOM_ENVS[config.env_name]
            self.env = env_class(**env_kwargs)
        else:
            self.env = suite.make(
                env_name=config.env_name,
                **env_kwargs
            )
        
        # Optionally wrap with Gym interface
        # self.env = GymWrapper(self.env)
        
        # Store action and observation specs
        self._setup_specs()
        
    def _setup_specs(self):
        """Setup action and observation specifications."""
        # Get action dimension
        self.action_dim = self.env.action_spec[0].shape[0]
        
        # Store observation keys
        obs = self.env.reset()
        self.obs_keys = list(obs.keys())
        
        # Separate state keys and image keys
        self.state_keys = [k for k in self.obs_keys 
                          if not k.endswith("_image") and not k.endswith("_depth")]
        self.image_keys = [k for k in self.obs_keys if k.endswith("_image")]
        self.depth_keys = [k for k in self.obs_keys if k.endswith("_depth")]
    
    @staticmethod
    def _build_controller_config(config: DataCollectionConfig):
        """
        根据配置构建 composite controller 配置。
        
        优先级:
          1. controller_config（完整 dict）: 直接使用
          2. controller_type（简称）+ controller_params（可选覆盖）: 从默认配置构建
          3. 都为 None: 返回 None，使用 robosuite 默认
          
        支持的 controller_type:
          - "OSC_POSE": 操作空间位姿控制（默认）
          - "OSC_POSITION": 操作空间仅位置控制
          - "JOINT_POSITION": 关节位置控制
          - "JOINT_VELOCITY": 关节速度控制
          - "JOINT_TORQUE": 关节扭矩控制
          - "IK_POSE": 逆运动学（仅 Panda/Sawyer/Baxter）
        
        Returns:
            dict or None: composite controller 配置，或 None 表示使用默认
        """
        # Case 1: 完整 controller_config 直接指定
        if config.controller_config is not None:
            if isinstance(config.controller_config, str):
                # 字符串视为 composite controller 名称（如 "BASIC"）
                return load_composite_controller_config(
                    controller=config.controller_config,
                    robot=config.robots
                )
            elif isinstance(config.controller_config, dict):
                return config.controller_config
            else:
                raise ValueError(
                    f"controller_config must be str or dict, got {type(config.controller_config)}"
                )
        
        # Case 2: 通过 controller_type 简称选择
        controller_type = getattr(config, 'controller_type', None)
        if controller_type is None:
            return None  # 使用 robosuite 默认

        # 合法的 arm controller 类型
        VALID_ARM_TYPES = {
            "OSC_POSE", "OSC_POSITION",
            "JOINT_POSITION", "JOINT_VELOCITY", "JOINT_TORQUE",
            "IK_POSE",
        }
        controller_type = controller_type.upper()
        if controller_type not in VALID_ARM_TYPES:
            raise ValueError(
                f"Unsupported controller_type: '{controller_type}'. "
                f"Valid options: {sorted(VALID_ARM_TYPES)}"
            )
        
        # IK_POSE 仅支持部分机器人
        if controller_type == "IK_POSE":
            supported_ik_robots = {"Panda", "Sawyer", "Baxter", "GR1FixedLowerBody"}
            if config.robots not in supported_ik_robots:
                raise ValueError(
                    f"IK_POSE controller is not supported for robot '{config.robots}'. "
                    f"Supported robots: {sorted(supported_ik_robots)}. "
                    f"Consider using OSC_POSE instead."
                )
        
        # 先加载该机器人的默认 composite controller 配置
        base_config = load_composite_controller_config(robot=config.robots)
        
        # 找到 arm part 并替换控制器类型
        body_parts = base_config.get('body_parts', {})
        # 单臂机器人可能是 body_parts['right']，双臂机器人是 body_parts['arms']['right']
        arm_part = None
        if 'right' in body_parts:
            arm_part = body_parts['right']
        elif 'arms' in body_parts and 'right' in body_parts['arms']:
            arm_part = body_parts['arms']['right']
        
        if arm_part is None:
            raise ValueError(
                f"Cannot find arm controller part in composite config. "
                f"body_parts keys: {list(body_parts.keys())}"
            )
        
        # 获取对应 controller_type 的默认参数模板
        type_to_part_file = {
            "OSC_POSE": "osc_pose",
            "OSC_POSITION": "osc_position",
            "JOINT_POSITION": "joint_position",
            "JOINT_VELOCITY": "joint_velocity",
            "JOINT_TORQUE": "joint_torque",
            "IK_POSE": "ik_pose",
        }
        import json
        import importlib
        robosuite_path = importlib.import_module('robosuite').__path__[0]
        part_json = f"{robosuite_path}/controllers/config/default/parts/{type_to_part_file[controller_type]}.json"
        
        try:
            with open(part_json, 'r') as f:
                new_arm_config = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Controller part config not found: {part_json}. "
                f"This controller type may not be available in your robosuite version."
            )
        
        # 保留 gripper 配置
        if 'gripper' in arm_part:
            new_arm_config['gripper'] = arm_part['gripper']
        
        # 应用用户自定义参数覆盖
        controller_params = getattr(config, 'controller_params', None)
        if controller_params:
            for key, value in controller_params.items():
                if key == 'type':
                    continue  # type 由 controller_type 决定，不允许覆盖
                new_arm_config[key] = value
        
        # 替换 arm part
        if 'right' in body_parts:
            body_parts['right'] = new_arm_config
        elif 'arms' in body_parts and 'right' in body_parts['arms']:
            body_parts['arms']['right'] = new_arm_config
        
        return base_config
        
    def reset(self) -> Dict[str, Any]:
        """
        Reset the environment.
        
        Returns:
            observation: Initial observation dictionary
        """
        obs = self.env.reset()
        return self._process_observation(obs)
    
    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: Action array
            
        Returns:
            observation: Observation dictionary
            reward: Reward scalar
            done: Done flag
            info: Info dictionary
        """
        obs, reward, done, info = self.env.step(action)
        obs = self._process_observation(obs)
        return obs, reward, done, info
    
    def _process_observation(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process raw observation from environment.
        
        Args:
            obs: Raw observation dictionary
            
        Returns:
            Processed observation dictionary
        """
        processed_obs = {}
        
        # Process state observations (flatten if needed)
        state_list = []
        for key in self.state_keys:
            if key in obs:
                val = obs[key]
                if isinstance(val, np.ndarray):
                    state_list.append(val.flatten())
                else:
                    state_list.append(np.array([val]))
        
        if state_list:
            processed_obs["state"] = np.concatenate(state_list)
        
        # Process image observations
        for key in self.image_keys:
            if key in obs:
                # Images are in shape (H, W, C), keep as is
                processed_obs[key] = obs[key]
        
        # Process depth observations
        for key in self.depth_keys:
            if key in obs:
                processed_obs[key] = obs[key]
        
        # Keep all original observations as well
        processed_obs["raw_obs"] = obs
        
        return processed_obs
    
    def get_camera_observations(self) -> Dict[str, np.ndarray]:
        """
        Get camera observations from current state.
        
        Returns:
            Dictionary of camera images
        """
        camera_obs = {}
        for camera_name in self.config.camera_names:
            img_key = f"{camera_name}_image"
            if img_key in self.env._get_observations():
                camera_obs[camera_name] = self.env._get_observations()[img_key]
        return camera_obs
    
    def render(self, mode: str = "human") -> np.ndarray:
        """
        Render the environment.
        
        Args:
            mode: Render mode ("human" or "rgb_array")
            
        Returns:
            Rendered image if mode="rgb_array"
        """
        if mode == "human":
            self.env.render()
            return None
        elif mode == "rgb_array":
            # Use the first camera for rendering
            camera_name = self.config.camera_names[0]
            img_key = f"{camera_name}_image"
            obs = self.env._get_observations()
            if img_key in obs:
                return obs[img_key]
            return None
    
    def render_multi_view(self) -> Dict[str, np.ndarray]:
        """
        Render all camera views.
        
        Returns:
            Dictionary mapping camera names to RGB images (H, W, 3)
            Images are flipped vertically to correct OpenGL coordinate system
        """
        camera_images = {}
        obs = self.env._get_observations()
        
        for camera_name in self.config.camera_names:
            img_key = f"{camera_name}_image"
            if img_key in obs:
                # Images from robosuite are in RGB format with OpenGL coordinates (origin at bottom-left)
                # Flip vertically to convert to standard image coordinates (origin at top-left)
                img = obs[img_key]
                img_flipped = np.flipud(img)
                camera_images[camera_name] = img_flipped
        
        return camera_images
    
    def get_robot_eef_pose(self) -> np.ndarray:
        """
        Get current end-effector pose.
        
        Returns:
            7D pose array (x, y, z, qx, qy, qz, qw) - xyzw quaternion format
        """
        obs = self.env._get_observations()
        # robosuite eef_quat 经过 convert_quat(to="xyzw") 处理，返回的是 xyzw 格式
        pos = obs.get(f"robot0_eef_pos", np.zeros(3))
        quat = obs.get(f"robot0_eef_quat", np.array([0.0, 0.0, 0.0, 1.0]))  # xyzw format
        return np.concatenate([pos, quat])
    
    def get_object_pose(self, obj_name: str) -> np.ndarray:
        """
        Get current object pose from simulation.
        
        Args:
            obj_name: 物体名称（如 "yellow_cup", "black_cup"）
            
        Returns:
            7D pose array (x, y, z, qx, qy, qz, qw) - xyzw quaternion format
        """
        import robosuite.utils.transform_utils as T
        
        # 通过 obj_body_id 获取物体的 body index
        if hasattr(self.env, 'obj_body_id') and obj_name in self.env.obj_body_id:
            body_id = self.env.obj_body_id[obj_name]
        else:
            # Fallback: 尝试通过 mujoco body name 查找
            # robosuite 通常在物体名后加 "_main" 后缀
            try:
                body_id = self.env.sim.model.body_name2id(f"{obj_name}_main")
            except ValueError:
                try:
                    body_id = self.env.sim.model.body_name2id(obj_name)
                except ValueError:
                    raise ValueError(
                        f"无法找到物体 '{obj_name}'。"
                        f"可用的 body names: {[self.env.sim.model.body_id2name(i) for i in range(self.env.sim.model.nbody)]}"
                    )
        
        pos = self.env.sim.data.body_xpos[body_id].copy()
        
        # body_xquat 返回 wxyz 格式，需要转换为 xyzw
        quat_wxyz = self.env.sim.data.body_xquat[body_id].copy()
        quat_xyzw = T.convert_quat(quat_wxyz, to="xyzw")
        
        return np.concatenate([pos, quat_xyzw])
    
    def close(self):
        """Close the environment."""
        self.env.close()
    
    def get_arm_controller_type(self) -> str:
        """
        获取当前 arm 控制器的类型名称。
        
        Returns:
            str: 控制器类型，如 "OSC_POSE", "JOINT_POSITION", "IK_POSE" 等
        """
        cfg = self.env.robots[0].composite_controller_config
        body_parts = cfg.get('body_parts', {})
        arm_part = body_parts.get('right') or (body_parts.get('arms', {}).get('right'))
        if arm_part is None:
            return "UNKNOWN"
        return arm_part.get('type', 'UNKNOWN')
    
    def get_arm_controller_config(self) -> Dict[str, Any]:
        """
        获取当前 arm 控制器的完整配置。
        
        Returns:
            dict: arm 控制器配置
        """
        cfg = self.env.robots[0].composite_controller_config
        body_parts = cfg.get('body_parts', {})
        arm_part = body_parts.get('right') or (body_parts.get('arms', {}).get('right'))
        return arm_part if arm_part is not None else {}
    
    @property
    def unwrapped(self):
        """Get the unwrapped environment."""
        return self.env
