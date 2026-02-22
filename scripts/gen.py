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
from utils.trajectory_generator import TrajectoryGenerator, interpolate_poses, PoseUtils
from utils.source_loader import SourceDemoLoader
import robosuite.utils.transform_utils as T


class WaypointPolicy:
    """
    Policy that follows pre-computed waypoints (EEF poses).
    Uses inverse kinematics to convert poses to joint actions.
    """
    
    def __init__(self, env_interface, waypoint_poses, gripper_actions):
        """
        Args:
            env_interface: Environment interface for pose-to-action conversion
            waypoint_poses: (N, 4, 4) or (N, 7) array of target EEF poses
            gripper_actions: (N, 1) array of gripper commands
        """
        self.env_interface = env_interface
        self.gripper_actions = gripper_actions
        self.current_step = 0
        self.total_steps = len(waypoint_poses)
        
        # Convert all poses to 7D format (x,y,z, qw,qx,qy,qz) for consistent handling
        self.waypoint_poses_7d = []
        for pose in waypoint_poses:
            if pose.shape == (4, 4):
                # Extract position and quaternion from 4x4 matrix
                pos = pose[:3, 3]
                quat = T.mat2quat(pose[:3, :3])  # Returns (w,x,y,z)
                pose_7d = np.concatenate([pos, quat])
            elif pose.shape == (7,):
                pose_7d = pose
            else:
                raise ValueError(f"Invalid pose shape: {pose.shape}, expected (4,4) or (7,)")
            self.waypoint_poses_7d.append(pose_7d)
        
        self.waypoint_poses_7d = np.array(self.waypoint_poses_7d)
    
    def get_action(self, observation: dict) -> np.ndarray:
        """Get action for current timestep."""
        if self.current_step >= self.total_steps:
            # Return last action if exceeded
            self.current_step = self.total_steps - 1
        
        # Get target pose for this step (7D format)
        target_pose_7d = self.waypoint_poses_7d[self.current_step]
        target_pos = target_pose_7d[:3]
        target_quat = target_pose_7d[3:]  # (w,x,y,z)
        gripper_action = self.gripper_actions[self.current_step]
        
        # Get current pose (also ensure 7D format)
        current_pose = self.env_interface.get_robot_eef_pose()
        if current_pose.shape == (4, 4):
            current_pos = current_pose[:3, 3]
            current_quat = T.mat2quat(current_pose[:3, :3])
        elif current_pose.shape == (7,):
            current_pos = current_pose[:3]
            current_quat = current_pose[3:]
        else:
            raise ValueError(f"Invalid current pose shape: {current_pose.shape}")
        
        # Position delta (simple proportional control)
        pos_delta = target_pos - current_pos
        
        # Rotation delta (axis-angle approximation)
        # For small angle differences, this is a reasonable approximation
        quat_diff = target_quat - current_quat
        # Scale rotation (xyz components) for responsiveness
        rot_delta = quat_diff[1:] * 2.0  # Use xyz components, scale up
        
        # Combine into action (pos_delta + rot_delta + gripper)
        action = np.concatenate([pos_delta, rot_delta, gripper_action])
        
        self.current_step += 1
        return action
    
    def reset(self):
        """Reset policy to start of trajectory."""
        self.current_step = 0
    
    def is_done(self):
        """Check if all waypoints have been executed."""
        return self.current_step >= self.total_steps


def collect_episode(env: RoboSuiteDataCollector, 
                   policy,  # Can be RandomPolicy or WaypointPolicy
                   render: bool = False,
                   verbose: bool = False,
                   max_steps: int = None) -> dict:
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
    
    # Set max steps from horizon if not specified
    if max_steps is None:
        max_steps = env.env.horizon if hasattr(env.env, 'horizon') else 1000
    
    while not done and timestep < max_steps:
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
        
        # Check if waypoint policy is done
        if hasattr(policy, 'is_done') and policy.is_done():
            done = True
        
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
    
    # Initialize trajectory generation
    source_demo_path = getattr(config, 'source_demo_path', None)
    use_trajectory_generation = getattr(config, 'use_trajectory_generation', False)
    
    if not use_trajectory_generation or not source_demo_path:
        raise ValueError(
            "Trajectory generation is required. Please set:\n"
            "  - use_trajectory_generation: true\n"
            "  - source_demo_path: <path_to_demo_file>"
        )
    
    print(f"\n✓ Trajectory Generation Mode Enabled")
    print(f"  Loading source demo from: {source_demo_path}")
    
    demo_loader = SourceDemoLoader(source_demo_path)
    traj_generator = TrajectoryGenerator(env)
    print(f"✓ Loaded {demo_loader.num_demos} source demonstrations")
    
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
        # Get source demo
        src_demo = demo_loader.get_demo(0)  # Use first demo
        
        # Use target_poses if available (MimicGen format), otherwise use eef_poses
        if 'target_poses' in src_demo and src_demo['target_poses'] is not None:
            src_poses = src_demo['target_poses']  # Already (N, 4, 4) matrices
        elif 'eef_poses' in src_demo and src_demo['eef_poses'] is not None:
            # Check if eef_poses are already 4x4 matrices or 7D poses
            eef_poses = src_demo['eef_poses']
            if eef_poses.shape[1] == 7:
                # Convert 7D poses to 4x4 matrices
                src_poses = []
                for pose_7d in eef_poses:
                    pose_mat = PoseUtils.make_pose(pose_7d[:3], T.quat2mat(pose_7d[3:]))
                    src_poses.append(pose_mat)
                src_poses = np.array(src_poses)
            elif len(eef_poses.shape) == 3 and eef_poses.shape[1:] == (4, 4):
                # Already 4x4 matrices
                src_poses = eef_poses
            else:
                raise ValueError(f"Invalid eef_poses shape: {eef_poses.shape}")
        else:
            raise ValueError("No EEF or target poses found in source demo")
        
        # Create waypoint policy
        policy = WaypointPolicy(
            env_interface=env,
            waypoint_poses=src_poses,
            gripper_actions=src_demo['gripper_actions']
        )
        
        episode_data = collect_episode(
            env, 
            policy, 
            render=config.has_renderer,
            verbose=False,
            max_steps=config.horizon
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
