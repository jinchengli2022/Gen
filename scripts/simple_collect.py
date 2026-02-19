"""
simple_collect.py

Simple data collection script for robosuite 1.5.
Collects random policy demonstrations for testing.

Usage:
    python simple_collect.py --env_name PickPlaceCan --num_episodes 10
    python simple_collect.py --env_name Stack --num_episodes 20 --render
"""

import argparse
import numpy as np
from tqdm import tqdm
import time

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.config import DataCollectionConfig
from env_interfaces.robosuite_env import RoboSuiteDataCollector
from utils.data_writer import create_data_writer


class RandomPolicy:
    """Simple random policy for testing."""
    
    def __init__(self, action_dim: int, action_range: tuple = (-1.0, 1.0)):
        """
        Initialize random policy.
        
        Args:
            action_dim: Dimension of action space
            action_range: Tuple of (min, max) action values
        """
        self.action_dim = action_dim
        self.action_min, self.action_max = action_range
        
    def get_action(self, observation: dict) -> np.ndarray:
        """
        Sample a random action.
        
        Args:
            observation: Current observation (unused)
            
        Returns:
            Random action array
        """
        return np.random.uniform(
            self.action_min, 
            self.action_max, 
            size=self.action_dim
        )
    
    def reset(self):
        """Reset policy (no-op for random policy)."""
        pass


def collect_episode(env: RoboSuiteDataCollector, 
                   policy: RandomPolicy,
                   render: bool = False) -> dict:
    """
    Collect a single episode of data.
    
    Args:
        env: RoboSuiteDataCollector instance
        policy: Policy to generate actions
        render: Whether to render the environment
        
    Returns:
        Episode data dictionary
    """
    obs = env.reset()
    policy.reset()
    
    episode_data = {
        "observations": [obs],
        "actions": [],
        "rewards": [],
        "dones": [],
        "success": False,
    }
    
    done = False
    timestep = 0
    
    while not done:
        # Render if requested (before action)
        if render:
            env.render(mode="human")
        
        # Get action from policy
        action = policy.get_action(obs)
        
        # Take step in environment
        next_obs, reward, done, info = env.step(action)
        
        # Store transition
        episode_data["observations"].append(next_obs)
        episode_data["actions"].append(action)
        episode_data["rewards"].append(reward)
        episode_data["dones"].append(done)
        
        # Update for next iteration
        obs = next_obs
        timestep += 1
        
        # Small delay for smooth visualization
        if render:
            time.sleep(0.02)  # 20ms delay for smoother visualization
    
    # Check for success (if available in info)
    if "success" in info:
        episode_data["success"] = info["success"]
    
    return episode_data


def main(args):
    """Main data collection function."""
    
    # 获取基础配置
    config = DataCollectionConfig(
        env_name=args.env_name,
        robots=args.robot,
        num_episodes=args.num_episodes,
        output_dir=args.output_dir,
        save_format=args.format,
        has_renderer=args.render,
        has_offscreen_renderer=args.use_camera or args.render,  # Need offscreen for camera obs or rendering
        use_camera_obs=args.use_camera,
        camera_names=args.camera_names.split(",") if args.camera_names else ["agentview"],
        camera_heights=args.image_size,
        camera_widths=args.image_size,
        horizon=args.horizon,
        control_freq=args.control_freq,
    )
    
    print("="*60)
    print("Robosuite Data Collection")
    print("="*60)
    print(f"Environment: {config.env_name}")
    print(f"Robot: {config.robots}")
    print(f"Episodes: {config.num_episodes}")
    print(f"Output: {config.output_dir}")
    print(f"Format: {config.save_format}")
    print(f"Render: {args.render}")
    print(f"Use Camera: {args.use_camera}")
    print("="*60)
    
    # 初始化仿真环境
    print("\nInitializing environment...")
    env = RoboSuiteDataCollector(config)
    print(f"Action dimension: {env.action_dim}")
    print(f"State keys: {env.state_keys}")
    print(f"Image keys: {env.image_keys}")
    
    # 初始化策略
    policy = RandomPolicy(action_dim=env.action_dim)
    
    # 初始化数据写入器
    writer = create_data_writer(
        output_dir=config.output_dir,
        env_name=config.env_name,
        format=config.save_format
    )
    
    # 正式按照episode生成数据
    print(f"\nCollecting {config.num_episodes} episodes...")
    success_count = 0
    
    for episode_idx in tqdm(range(config.num_episodes)):
        # Collect episode
        episode_data = collect_episode(env, policy, render=args.render)
        
        # Write to disk
        writer.write_episode(episode_data, episode_idx)
        
        # Track success
        if episode_data["success"]:
            success_count += 1
        
        # Print progress
        if (episode_idx + 1) % 10 == 0:
            avg_reward = np.mean([np.sum(ep["rewards"]) for ep in [episode_data]])
            print(f"\nEpisode {episode_idx + 1}/{config.num_episodes}")
            print(f"  Avg reward: {avg_reward:.2f}")
            print(f"  Success rate: {success_count}/{episode_idx + 1}")
    
    # Finalize
    writer.finalize()
    env.close()
    
    print("\n" + "="*60)
    print("Data collection complete!")
    print(f"Total episodes: {config.num_episodes}")
    print(f"Success rate: {success_count}/{config.num_episodes} ({100*success_count/config.num_episodes:.1f}%)")
    print(f"Data saved to: {config.output_dir}")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect robosuite data with random policy")
    
    # Environment settings
    parser.add_argument("--env_name", type=str, default="PickPlaceCan",
                       help="Robosuite environment name")
    parser.add_argument("--robot", type=str, default="Panda",
                       help="Robot model (Panda, Sawyer, IIWA, etc.)")
    parser.add_argument("--horizon", type=int, default=500,
                       help="Maximum episode length")
    parser.add_argument("--control_freq", type=int, default=20,
                       help="Control frequency in Hz")
    
    # Data collection settings
    parser.add_argument("--num_episodes", type=int, default=10,
                       help="Number of episodes to collect")
    parser.add_argument("--output_dir", type=str, default="data/gen",
                       help="Output directory for collected data")
    parser.add_argument("--format", type=str, default="hdf5", choices=["hdf5", "pickle"],
                       help="Data format")
    
    # Observation settings
    parser.add_argument("--use_camera", action="store_true",
                       help="Use camera observations")
    parser.add_argument("--camera_names", type=str, default="agentview",
                       help="Comma-separated camera names")
    parser.add_argument("--image_size", type=int, default=224,
                       help="Image size (height and width)")
    
    # Visualization
    parser.add_argument("--render", action="store_true",
                       help="Render environment during collection")
    
    args = parser.parse_args()
    main(args)
