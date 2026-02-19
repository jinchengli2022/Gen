"""
gen.py

Generic data collection script for robosuite environments using JSON configuration.

Usage:
    python gen.py --config ../configs/examples/pickplace_demo.json
    python gen.py --config ../configs/examples/pouring_water.json --render
    python gen.py --config path/to/custom_config.json --render
"""

import argparse
import numpy as np
from tqdm import tqdm
import time
import cv2

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.config import DataCollectionConfig
from env_interfaces.robosuite_env import RoboSuiteDataCollector
from utils.data_writer import create_data_writer


class RandomPolicy:
    """Random policy for testing."""
    
    def __init__(self, action_dim: int, action_range: tuple = (-1.0, 1.0)):
        self.action_dim = action_dim
        self.action_min, self.action_max = action_range
        
    def get_action(self, observation: dict) -> np.ndarray:
        return np.random.uniform(
            self.action_min, 
            self.action_max, 
            size=self.action_dim
        )
    
    def reset(self):
        pass


def collect_episode(env: RoboSuiteDataCollector, 
                   policy: RandomPolicy,
                   render: bool = False,
                   verbose: bool = False) -> dict:
    """Collect a single episode."""
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
        if render:
            # Render all camera views
            camera_images = env.render_multi_view()
            if camera_images:
                # Display each camera view in separate windows
                for camera_name, img in camera_images.items():
                    # Convert RGB to BGR for OpenCV
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    cv2.imshow(f"{camera_name}", img_bgr)
                cv2.waitKey(1)  # Small delay for rendering
            else:
                # Fallback to default rendering
                env.render(mode="human")
        
        action = policy.get_action(obs)
        next_obs, reward, done, info = env.step(action)
        
        episode_data["observations"].append(next_obs)
        episode_data["actions"].append(action)
        episode_data["rewards"].append(reward)
        episode_data["dones"].append(done)
        
        obs = next_obs
        timestep += 1
        
        if render:
            time.sleep(0.02)
    
    # Check success
    if "success" in info:
        episode_data["success"] = info["success"]
    
    # Get failure reasons
    if hasattr(env.unwrapped, 'failure_reasons'):
        episode_data["failure_reasons"] = env.unwrapped.failure_reasons
    
    if verbose and episode_data.get("failure_reasons"):
        print(f"  Failure: {'; '.join(episode_data['failure_reasons'])}")
    
    return episode_data


def main(args):
    """Main collection function."""
    
    # Load config from JSON file
    print(f"Loading configuration from: {args.config}")
    try:
        config = DataCollectionConfig.from_json(args.config)
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        return
    except Exception as e:
        print(f"✗ Failed to load config: {e}")
        return
    
    # Override render setting if specified from command line
    if args.render:
        print("✓ Overriding has_renderer=True from command line")
        config.has_renderer = True
        # Also enable offscreen renderer if rendering
        if not config.has_offscreen_renderer:
            config.has_offscreen_renderer = True
    
    print("="*60)
    print("Robosuite Data Collection")
    print("="*60)
    print(f"Environment: {config.env_name}")
    print(f"Robot: {config.robots}")
    print(f"Episodes: {config.num_episodes}")
    print(f"Horizon: {config.horizon}")
    print(f"Control Freq: {config.control_freq} Hz")
    print(f"Output: {config.output_dir}")
    print(f"Format: {config.save_format}")
    print(f"Render: {config.has_renderer}")
    print(f"Use Camera: {config.use_camera_obs}")
    if config.use_camera_obs:
        print(f"  Cameras: {', '.join(config.camera_names)}")
        print(f"  Image size: {config.camera_heights}x{config.camera_widths}")
    if config.controller_config:
        print(f"Controller: {config.controller_config.get('type', 'default')}")
    print("="*60)
    
    # Initialize environment
    print(f"\nInitializing {config.env_name} environment...")
    try:
        env = RoboSuiteDataCollector(config)
        print(f"✓ Environment loaded successfully")
        print(f"  Action dimension: {env.action_dim}")
        print(f"  State keys: {env.state_keys}")
        print(f"  Image keys: {env.image_keys}")
    except Exception as e:
        print(f"✗ Failed to load environment: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Initialize policy
    policy = RandomPolicy(action_dim=env.action_dim)
    print(f"\n✓ Random policy initialized")
    
    # Initialize data writer
    writer = create_data_writer(
        output_dir=config.output_dir,
        env_name=config.env_name,
        format=config.save_format
    )
    print(f"✓ Data writer initialized")
    
    # Collect episodes
    print(f"\nCollecting {config.num_episodes} episodes...")
    success_count = 0
    failure_stats = {}
    total_rewards = []
    
    for episode_idx in tqdm(range(config.num_episodes)):
        episode_data = collect_episode(
            env, 
            policy, 
            render=config.has_renderer,
            verbose=False
        )
        
        writer.write_episode(episode_data, episode_idx)
        
        # Track statistics
        if episode_data["success"]:
            success_count += 1
        
        episode_reward = np.sum(episode_data["rewards"])
        total_rewards.append(episode_reward)
        
        # Track failure reasons
        if "failure_reasons" in episode_data:
            for reason in episode_data["failure_reasons"]:
                failure_stats[reason] = failure_stats.get(reason, 0) + 1
        
        # Print progress
        if (episode_idx + 1) % 10 == 0:
            avg_reward = np.mean(total_rewards[-10:])
            print(f"\nEpisode {episode_idx + 1}/{config.num_episodes}")
            print(f"  Recent avg reward: {avg_reward:.3f}")
            print(f"  Success rate: {success_count}/{episode_idx + 1} ({100*success_count/(episode_idx+1):.1f}%)")
    
    # Finalize
    writer.finalize()
    env.close()
    
    # Close any OpenCV windows
    if config.has_renderer:
        cv2.destroyAllWindows()
    
    # Print summary
    print("\n" + "="*60)
    print("Data Collection Complete!")
    print("="*60)
    print(f"Total episodes: {config.num_episodes}")
    print(f"Success rate: {success_count}/{config.num_episodes} ({100*success_count/config.num_episodes:.1f}%)")
    print(f"Average reward: {np.mean(total_rewards):.3f} ± {np.std(total_rewards):.3f}")
    print(f"Data saved to: {config.output_dir}")
    
    if failure_stats:
        print("\nFailure Analysis:")
        print("-" * 60)
        for reason, count in sorted(failure_stats.items(), key=lambda x: x[1], reverse=True):
            percentage = 100 * count / config.num_episodes
            print(f"  {reason}: {count} ({percentage:.1f}%)")
    
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect demonstration data from robosuite environments using JSON configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Use a config file
    python gen.py --config ../configs/examples/pickplace_demo.json
    
    # Use config with real-time rendering override
    python gen.py --config ../configs/examples/pouring_water.json --render
    
    # Use custom config
    python gen.py --config path/to/my_config.json
"""
    )
    
    # Required: config file path
    parser.add_argument("--config", type=str, required=True,
                       help="Path to JSON configuration file (required)")
    
    # Optional: render override
    parser.add_argument("--render", action="store_true",
                       help="Enable real-time rendering (overrides config file setting)")
    
    args = parser.parse_args()
    main(args)
