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
        
        # Handle controller configuration if specified
        if config.controller_config is not None:
            if isinstance(config.controller_config, str):
                # If it's a string, treat it as a controller name (e.g., "BASIC")
                controller_configs = load_composite_controller_config(
                    controller=config.controller_config,
                    robot=config.robots
                )
            elif isinstance(config.controller_config, dict):
                # If it's a dict, pass it directly (assuming it's a full composite controller config)
                controller_configs = config.controller_config
            else:
                raise ValueError(f"controller_config must be str or dict, got {type(config.controller_config)}")
            
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
    
    def close(self):
        """Close the environment."""
        self.env.close()
    
    @property
    def unwrapped(self):
        """Get the unwrapped environment."""
        return self.env
