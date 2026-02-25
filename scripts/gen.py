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
import logging
import os
from datetime import datetime

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
        
        # Convert all poses to 7D format (x,y,z, x,y,z,w) for consistent handling
        self.waypoint_poses_7d = []
        for pose in waypoint_poses:
            if pose.shape == (4, 4):
                # Extract position and quaternion from 4x4 matrix
                pos = pose[:3, 3]
                quat = T.mat2quat(pose[:3, :3])  # Returns (x,y,z,w) in robosuite
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
        target_quat = target_pose_7d[3:]  # (x,y,z,w)
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
        
        # get maximum position and rotation action bounds
        max_dpos = 0.05
        max_drot = 0.15
        # max_dpos = self.env_interface.env.robots[0].composite_controller_config["body_parts"].right.output_max[0]
        # max_drot = self.env_interface.env.robots[0].composite_controller_config["body_parts"].right.output_max[3]

        # Position delta
        delta_position = target_pos - current_pos
        if np.any(np.abs(delta_position) > max_dpos):
            print(f"[Warning] Step {self.current_step}: Position delta {delta_position} exceeds max_dpos {max_dpos}. Clipping applied.")
        delta_position = np.clip(delta_position / max_dpos, -1.0, 1.0)
        
        # Rotation delta
        target_rot = T.quat2mat(target_quat)
        curr_rot = T.quat2mat(current_quat)
        delta_rot_mat = target_rot.dot(curr_rot.T)
        delta_quat = T.mat2quat(delta_rot_mat)
        delta_rotation = T.quat2axisangle(delta_quat)
        
        if np.any(np.abs(delta_rotation) > max_drot):
            print(f"[Warning] Step {self.current_step}: Rotation delta {delta_rotation} exceeds max_drot {max_drot}. Clipping applied.")
        delta_rotation = np.clip(delta_rotation / max_drot, -1.0, 1.0)
        
        # Combine into action (pos_delta + rot_delta + gripper)
        action = np.concatenate([delta_position, delta_rotation, gripper_action])
        
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
                   max_steps: int = None,
                   skip_reset: bool = False,
                   initial_obs: dict = None) -> dict:
    """Collect a single episode.
    
    Args:
        skip_reset: 若为 True，不再调用 env.reset()，使用 initial_obs 作为初始观测
        initial_obs: 当 skip_reset=True 时提供的初始观测
    """
    if skip_reset and initial_obs is not None:
        obs = initial_obs
    else:
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
    
    # Check success - 优先使用 _check_success() 的返回值，因为 robosuite 的 info 字典不含 "success" 键
    if hasattr(env.unwrapped, '_check_success'):
        episode_data["success"] = env.unwrapped._check_success()
        episode_data["failure_reasons"] = getattr(env.unwrapped, 'failure_reasons', [])
    elif "success" in info:
        episode_data["success"] = info["success"]
    
    if verbose and episode_data.get("failure_reasons"):
        print(f"  Failure: {'; '.join(episode_data['failure_reasons'])}")
    
    return episode_data


def setup_logger(output_dir, env_name, seed=None):
    """
    设置实时日志记录器，每条记录立即 flush 到文件。
    
    Args:
        output_dir: 输出目录
        env_name: 环境名称
        seed: 随机种子（用于日志文件名）
    
    Returns:
        logger: 配置好的 Logger 实例
        log_path: 日志文件路径
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    seed_str = f"_seed{seed}" if seed is not None else ""
    log_filename = f"gen_{env_name}{seed_str}_{timestamp}.log"
    log_path = os.path.join(output_dir, log_filename)
    
    logger = logging.getLogger(f"gen_{timestamp}")
    logger.setLevel(logging.DEBUG)
    # 防止重复 handler
    logger.handlers.clear()
    
    # 文件 handler — 实时 flush（通过自定义 emit 在每条日志后自动 flush）
    class FlushFileHandler(logging.FileHandler):
        def emit(self, record):
            super().emit(record)
            if self.stream:
                self.stream.flush()
    
    file_handler = FlushFileHandler(log_path, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    
    logger.addHandler(file_handler)
    
    return logger, log_path


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
    
    # Set random seed for reproducibility
    if config.seed is not None:
        np.random.seed(config.seed)
        print(f"✓ Random seed set to: {config.seed}")
    
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
    if config.seed is not None:
        print(f"Seed: {config.seed}")
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
    writer_kwargs = {}
    if config.save_format == "rlds":
        writer_kwargs["language_instruction"] = config.language_instruction
        writer_kwargs["image_key_primary"] = "agentview_image"
        writer_kwargs["image_key_wrist"] = "robot0_eye_in_hand_image"
    
    writer = create_data_writer(
        output_dir=config.output_dir,
        env_name=config.env_name,
        format=config.save_format,
        **writer_kwargs
    )
    print(f"✓ Data writer initialized")
    
    # Initialize real-time logger
    logger, log_path = setup_logger(config.output_dir, config.env_name, config.seed)
    writer.set_logger(logger)
    logger.info("=" * 60)
    logger.info("Trajectory Generation Log")
    logger.info("=" * 60)
    logger.info(f"Environment: {config.env_name}")
    logger.info(f"Robot: {config.robots}")
    logger.info(f"Episodes: {config.num_episodes}")
    logger.info(f"Horizon: {config.horizon}")
    logger.info(f"Seed: {config.seed}")
    logger.info(f"Source demo: {source_demo_path}")
    logger.info(f"Output dir: {config.output_dir}")
    logger.info("=" * 60)
    print(f"✓ Real-time log initialized: {log_path}")
    
    # Collect episodes
    print(f"\nCollecting {config.num_episodes} episodes...")
    success_count = 0
    failure_stats = {}
    total_rewards = []
    
    # 具体的生成过程
    for episode_idx in tqdm(range(config.num_episodes)):
        # Get source demo
        src_demo = demo_loader.get_demo(0)  # Use first demo

        # Reset 环境，获取新场景中的物体位姿
        obs = env.reset()
        
        # 获取当前 EEF 位姿
        current_eef_pose = env.get_robot_eef_pose()
        
        # 获取新场景中的物体位姿
        # 从 subtask_object_signals 确定操作对象和非操作对象
        obj_signals = src_demo.get('subtask_object_signals', {})
        obj_poses_src = src_demo.get('object_poses', {})
        obj_names = list(obj_poses_src.keys())
        
        if len(obj_names) < 2:
            raise ValueError(f"需要至少 2 个物体，当前只有: {obj_names}")
        
        # 确定操作对象和非操作对象名称
        operated_obj_name = None
        non_operated_obj_name = None
        if obj_signals:
            for obj_name, signal in obj_signals.items():
                if np.any(signal == 1):
                    operated_obj_name = obj_name
                else:
                    non_operated_obj_name = obj_name
        if operated_obj_name is None:
            operated_obj_name = obj_names[0]
        if non_operated_obj_name is None:
            for n in obj_names:
                if n != operated_obj_name:
                    non_operated_obj_name = n
                    break
        
        # 从环境中获取新物体位姿 (7D xyzw 格式)
        new_operated_obj_pose = env.get_object_pose(operated_obj_name)
        new_non_operated_obj_pose = env.get_object_pose(non_operated_obj_name)
        
        print(f"\n[Episode {episode_idx}] 操作对象: {operated_obj_name}, 非操作对象: {non_operated_obj_name}")
        print(f"  新操作物体位姿: pos={new_operated_obj_pose[:3]}")
        print(f"  新非操作物体位姿: pos={new_non_operated_obj_pose[:3]}")
        
        # 实时日志：记录场景配置
        logger.info(f"--- Episode {episode_idx}/{config.num_episodes} ---")
        logger.info(f"操作对象: {operated_obj_name}, 非操作对象: {non_operated_obj_name}")
        logger.info(f"EEF 初始位姿: pos={np.array2string(current_eef_pose[:3], precision=4)}")
        logger.info(f"新 {operated_obj_name} 位姿: pos={np.array2string(new_operated_obj_pose[:3], precision=4)}, "
                     f"quat={np.array2string(new_operated_obj_pose[3:], precision=4)}")
        logger.info(f"新 {non_operated_obj_name} 位姿: pos={np.array2string(new_non_operated_obj_pose[:3], precision=4)}, "
                     f"quat={np.array2string(new_non_operated_obj_pose[3:], precision=4)}")
        
        # 使用分段式轨迹变换
        new_target_poses, new_gripper_actions = traj_generator.transform_demo_to_new_scene(
            src_demo=src_demo,
            new_operated_obj_pose=new_operated_obj_pose,
            new_non_operated_obj_pose=new_non_operated_obj_pose,
            current_eef_pose=current_eef_pose,
        )
        
        # 实时日志：记录轨迹分段信息
        logger.info(f"轨迹生成完成: 总步数={len(new_target_poses)}")
        
        # Create waypoint policy
        policy = WaypointPolicy(
            env_interface=env,
            waypoint_poses=new_target_poses,
            gripper_actions=new_gripper_actions,
        )
        
        episode_data = collect_episode(
            env, 
            policy, 
            render=config.has_renderer,
            verbose=False,
            max_steps=config.horizon,
            skip_reset=True,
            initial_obs=obs,
        )
        
        writer.write_episode(episode_data, episode_idx)
        
        # Track statistics
        is_success = episode_data["success"]
        if is_success:
            success_count += 1
        
        episode_reward = np.sum(episode_data["rewards"])
        total_rewards.append(episode_reward)
        actual_steps = len(episode_data["actions"])
        
        # Track failure reasons（collect_episode 已调用 _check_success() 并填充 failure_reasons）
        failure_reasons = episode_data.get("failure_reasons", [])
        if failure_reasons:
            for reason in failure_reasons:
                failure_stats[reason] = failure_stats.get(reason, 0) + 1
        
        # 实时日志：记录执行结果（每条轨迹完成后立即写入）
        status_str = "✓ SUCCESS" if is_success else "✗ FAILED"
        logger.info(f"执行结果: {status_str} | 实际步数={actual_steps} | reward={episode_reward:.4f}")
        if not is_success:
            if failure_reasons:
                for reason in failure_reasons:
                    logger.info(f"  失败原因: {reason}")
            else:
                logger.info(f"  失败原因: (未知，环境未提供具体原因)")
        
        # 实时日志：当前累计统计
        current_success_rate = 100 * success_count / (episode_idx + 1)
        logger.info(f"累计统计: 成功={success_count}/{episode_idx + 1} ({current_success_rate:.1f}%) | "
                     f"平均reward={np.mean(total_rewards):.4f}")
        
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
    
    # 日志：最终汇总
    logger.info("=" * 60)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total episodes: {config.num_episodes}")
    logger.info(f"Success rate: {success_count}/{config.num_episodes} "
                 f"({100*success_count/config.num_episodes:.1f}%)")
    logger.info(f"Average reward: {np.mean(total_rewards):.3f} ± {np.std(total_rewards):.3f}")
    
    if failure_stats:
        print("\nFailure Analysis:")
        print("-" * 60)
        logger.info("Failure Analysis:")
        for reason, count in sorted(failure_stats.items(), key=lambda x: x[1], reverse=True):
            percentage = 100 * count / config.num_episodes
            print(f"  {reason}: {count} ({percentage:.1f}%)")
            logger.info(f"  {reason}: {count} ({percentage:.1f}%)")
    
    logger.info(f"Data saved to: {config.output_dir}")
    logger.info(f"Log saved to: {log_path}")
    logger.info("=" * 60)
    
    # 关闭 logger
    for handler in logger.handlers[:]:
        try:
            handler.close()
        except Exception:
            pass
    logger.handlers.clear()
    
    print(f"Log saved to: {log_path}")
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
