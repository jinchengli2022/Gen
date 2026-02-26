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
import matplotlib.pyplot as plt

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.config import DataCollectionConfig
from env_interfaces.robosuite_env import RoboSuiteDataCollector
from utils.data_writer import create_data_writer
from utils.trajectory_generator import TrajectoryGenerator, interpolate_poses, PoseUtils
from utils.source_loader import SourceDemoLoader
from utils.source_preprocessor import SourceDemoPreprocessor
import robosuite.utils.transform_utils as T


class WaypointPolicy:
    """
    Policy that follows pre-computed waypoints (EEF poses).
    Supports multiple controller types:
      - OSC_POSE / OSC_POSITION: 归一化笛卡尔增量 action
      - IK_POSE: 笛卡尔增量 action（由 IK 控制器内部求逆解）
      - JOINT_POSITION: 直接关节角度目标（需要外部 IK 求解，暂不支持）
    """
    
    # 支持笛卡尔增量输入的控制器类型
    CARTESIAN_DELTA_CONTROLLERS = {"OSC_POSE", "OSC_POSITION", "IK_POSE"}
    
    def __init__(self, env_interface, waypoint_poses, gripper_actions,
                 limit_dpos=0.85, limit_drot=1.0):
        """
        Args:
            env_interface: Environment interface for pose-to-action conversion
            waypoint_poses: (N, 4, 4) or (N, 7) array of target EEF poses
            gripper_actions: (N, 1) array of gripper commands
            limit_dpos: 归一化后位置增量上限 (0~1)，超过即视为超距
            limit_drot: 归一化后旋转增量上限 (0~1)，超过即视为超距
        """
        self.env_interface = env_interface
        self.gripper_actions = gripper_actions
        self.current_step = 0
        self.total_steps = len(waypoint_poses)
        self.limit_dpos = limit_dpos
        self.limit_drot = limit_drot
        self.clip_warnings = []  # 收集限距警告信息
        
        # 获取当前控制器类型
        self.controller_type = env_interface.get_arm_controller_type()
        if self.controller_type not in self.CARTESIAN_DELTA_CONTROLLERS:
            raise ValueError(
                f"WaypointPolicy 当前仅支持笛卡尔增量控制器: {self.CARTESIAN_DELTA_CONTROLLERS}. "
                f"当前控制器类型: '{self.controller_type}'. "
                f"如需使用 JOINT_POSITION 等控制器，需先实现关节空间轨迹规划。"
            )
        
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
        """
        Get action for current timestep.
        
        根据控制器类型生成不同格式的 action:
          - OSC_POSE:     [norm_dx, norm_dy, norm_dz, norm_drx, norm_dry, norm_drz, gripper]
          - OSC_POSITION: [norm_dx, norm_dy, norm_dz, gripper]（无旋转分量）
          - IK_POSE:      [norm_dx, norm_dy, norm_dz, norm_drx, norm_dry, norm_drz, gripper]
        """
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
        
        # 根据控制器类型获取 action bounds 和生成 action
        arm_config = self.env_interface.get_arm_controller_config()
        
        if self.controller_type in ("OSC_POSE", "OSC_POSITION"):
            # OSC 系列: 归一化到 [-1, 1]，由 output_max/output_min 缩放
            output_max = arm_config.get('output_max', [0.05, 0.05, 0.05, 0.5, 0.5, 0.5])
            output_max_pos = output_max[0] if isinstance(output_max, list) else output_max
            
            # Position delta → normalize to [-1, 1]
            delta_position = target_pos - current_pos
            normalized_dpos = delta_position / output_max_pos
            if np.any(np.abs(normalized_dpos) > self.limit_dpos):
                warn_msg = (f"Step {self.current_step}: Normalized pos delta "
                            f"{np.round(normalized_dpos, 4)} exceeds limit_dpos {self.limit_dpos}.")
                self.clip_warnings.append(warn_msg)
            normalized_dpos = np.clip(normalized_dpos, -1.0, 1.0)
            
            if self.controller_type == "OSC_POSITION":
                # 仅位置控制，无旋转分量
                action = np.concatenate([normalized_dpos, gripper_action])
            else:
                # OSC_POSE: 含旋转分量
                output_max_rot = output_max[3] if isinstance(output_max, list) else output_max
                
                target_rot = T.quat2mat(target_quat)
                curr_rot = T.quat2mat(current_quat)
                delta_rot_mat = target_rot.dot(curr_rot.T)
                delta_quat = T.mat2quat(delta_rot_mat)
                delta_rotation = T.quat2axisangle(delta_quat)
                normalized_drot = delta_rotation / output_max_rot
                
                if np.any(np.abs(normalized_drot) > self.limit_drot):
                    warn_msg = (f"Step {self.current_step}: Normalized rot delta "
                                f"{np.round(normalized_drot, 4)} exceeds limit_drot {self.limit_drot}.")
                    self.clip_warnings.append(warn_msg)
                normalized_drot = np.clip(normalized_drot, -1.0, 1.0)
                
                action = np.concatenate([normalized_dpos, normalized_drot, gripper_action])
        
        elif self.controller_type == "IK_POSE":
            # IK_POSE: 输入也是笛卡尔增量 (dx,dy,dz,ax,ay,az)，但由 ik_pos_limit/ik_ori_limit 裁剪
            # IK 内部使用 user_sensitivity=0.3 缩放，并自行进行 clip
            ik_pos_limit = arm_config.get('ik_pos_limit', 0.02)
            ik_ori_limit = arm_config.get('ik_ori_limit', 0.05)
            
            delta_position = target_pos - current_pos
            # IK 控制器的输入不需要归一化，但会被 ik_pos_limit clip
            # 用 ik_pos_limit 作为归一化基准进行限距检测
            normalized_dpos_for_check = delta_position / ik_pos_limit if ik_pos_limit > 0 else delta_position
            if np.any(np.abs(normalized_dpos_for_check) > self.limit_dpos):
                warn_msg = (f"Step {self.current_step}: IK pos delta "
                            f"{np.round(delta_position, 6)} exceeds ik_pos_limit * limit_dpos.")
                self.clip_warnings.append(warn_msg)
            
            target_rot = T.quat2mat(target_quat)
            curr_rot = T.quat2mat(current_quat)
            delta_rot_mat = target_rot.dot(curr_rot.T)
            delta_quat = T.mat2quat(delta_rot_mat)
            delta_rotation = T.quat2axisangle(delta_quat)
            
            # IK 输入: (dx, dy, dz, ax, ay, az)，不需要归一化
            action = np.concatenate([delta_position, delta_rotation, gripper_action])
        
        self.current_step += 1
        return action
    
    def reset(self):
        """Reset policy to start of trajectory."""
        self.current_step = 0
        self.clip_warnings = []
    
    def is_done(self):
        """Check if all waypoints have been executed."""
        return self.current_step >= self.total_steps


def collect_episode(env: RoboSuiteDataCollector, 
                   policy,  # Can be RandomPolicy or WaypointPolicy
                   render: bool = False,
                   verbose: bool = False,
                   max_steps: int = None,
                   skip_reset: bool = False,
                   initial_obs: dict = None,
                   debug: bool = False,
                   episode_idx: int = None) -> dict:
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
    
    # For debug: collect executed and planned EEF positions per step
    executed_positions = []  # list of (3,) positions
    planned_positions = []   # if policy exposes planned waypoints
    joint_positions_log = []  # list of (n_joints,) joint angles
    joint_limits = None       # (n_joints, 2) lower/upper limits, fetched once

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
        
        # Record executed (current) EEF pose before taking action
        if debug:
            try:
                cur_pose = env.get_robot_eef_pose()
                if cur_pose is not None:
                    if hasattr(cur_pose, 'shape') and tuple(cur_pose.shape) == (4, 4):
                        cur_pos = cur_pose[:3, 3]
                    else:
                        cur_pos = np.array(cur_pose[:3])
                    executed_positions.append(np.array(cur_pos))
            except Exception:
                # best-effort; don't break episode on debug collection
                pass

            # planned position from policy (if available)
            if hasattr(policy, 'waypoint_poses_7d'):
                try:
                    if policy.current_step < len(policy.waypoint_poses_7d):
                        planned_positions.append(np.array(policy.waypoint_poses_7d[policy.current_step][:3]))
                except Exception:
                    pass

            # Record joint positions & limits
            try:
                robot = env.env.robots[0]
                # qpos indices for robot arm joints
                q_idx = robot.joint_indexes  # list of joint qpos indices
                q = np.array([env.env.sim.data.qpos[idx] for idx in q_idx])
                joint_positions_log.append(q)
                # Fetch joint limits once
                if joint_limits is None:
                    jnt_range = env.env.sim.model.jnt_range  # (n_all_joints, 2)
                    joint_limits = np.array([jnt_range[idx] for idx in q_idx])  # (n_arm, 2)
            except Exception:
                pass

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
        pass  # failure_reasons 已记录到日志
    
    # 附加 debug 轨迹数据（如果收集到了）
    if len(executed_positions) > 0:
        episode_data['_debug_executed_positions'] = np.stack(executed_positions)
    else:
        episode_data['_debug_executed_positions'] = None
    if len(planned_positions) > 0:
        episode_data['_debug_planned_positions'] = np.stack(planned_positions)
    else:
        episode_data['_debug_planned_positions'] = None
    if len(joint_positions_log) > 0:
        episode_data['_debug_joint_positions'] = np.stack(joint_positions_log)
        episode_data['_debug_joint_limits'] = joint_limits
    else:
        episode_data['_debug_joint_positions'] = None
        episode_data['_debug_joint_limits'] = None

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
    try:
        config = DataCollectionConfig.from_json(args.config)
        config.output_dir = os.path.join(config.output_dir, config.env_name)
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        return
    except Exception as e:
        print(f"✗ Failed to load config: {e}")
        return
    
    # Override render setting if specified from command line
    if args.render:
        config.has_renderer = True
        # Also enable offscreen renderer if rendering
        if not config.has_offscreen_renderer:
            config.has_offscreen_renderer = True
    
    # Set random seed for reproducibility
    if config.seed is not None:
        np.random.seed(config.seed)
    
    print(f"[Gen] {config.env_name} | {config.robots} | {config.num_episodes} episodes | seed={config.seed}")
    
    # Initialize environment
    try:
        env = RoboSuiteDataCollector(config)
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
    
    demo_loader = SourceDemoLoader(source_demo_path)
    traj_generator = TrajectoryGenerator(env)
    
    # Read robot arm controller type and action bounds
    actual_controller_type = env.get_arm_controller_type()
    arm_config = env.get_arm_controller_config()
    
    # 根据控制器类型确定物理空间限距阈值
    if actual_controller_type in ("OSC_POSE", "OSC_POSITION"):
        output_max = arm_config.get('output_max', [0.05, 0.05, 0.05, 0.5, 0.5, 0.5])
        output_max_pos = output_max[0] if isinstance(output_max, list) else output_max
        output_max_rot = output_max[3] if isinstance(output_max, list) and len(output_max) > 3 else 0.5
        physical_limit_dpos = config.limit_dpos * output_max_pos
        physical_limit_drot = config.limit_drot * output_max_rot
    elif actual_controller_type == "IK_POSE":
        ik_pos_limit = arm_config.get('ik_pos_limit', 0.02)
        ik_ori_limit = arm_config.get('ik_ori_limit', 0.05)
        physical_limit_dpos = config.limit_dpos * ik_pos_limit
        physical_limit_drot = config.limit_drot * ik_ori_limit
    else:
        raise ValueError(
            f"Unsupported controller type for trajectory generation: '{actual_controller_type}'. "
            f"Supported: OSC_POSE, OSC_POSITION, IK_POSE"
        )
    
    # Preprocess source demo: interpolate overspeed segments
    # 物理空间限距阈值 = limit * output_max
    physical_limit_dpos = config.limit_dpos * output_max_pos
    physical_limit_drot = config.limit_drot * output_max_rot
    preprocessor = SourceDemoPreprocessor(
        physical_limit_dpos=physical_limit_dpos,
        physical_limit_drot=physical_limit_drot,
        verbose=True,
    )
    raw_demo = demo_loader.get_demo(0)
    processed_demo = preprocessor.process(raw_demo)
    
    # Debug mode: enable trajectory visualization
    if args.debug:
        traj_generator.debug = True
        traj_generator.debug_output_dir = config.output_dir
        traj_generator.physical_limit_dpos = physical_limit_dpos   # 物理空间限距阈值
        traj_generator.physical_limit_drot = physical_limit_drot
    
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
    logger.info(f"Action limits: limit_dpos={config.limit_dpos}, limit_drot={config.limit_drot}")
    logger.info(f"Arm controller: {actual_controller_type}")
    if actual_controller_type in ("OSC_POSE", "OSC_POSITION"):
        logger.info(f"Robot output_max: pos={output_max_pos}, rot={output_max_rot}")
    elif actual_controller_type == "IK_POSE":
        logger.info(f"IK limits: ik_pos_limit={ik_pos_limit}, ik_ori_limit={ik_ori_limit}")
    logger.info(f"Physical limits: dpos={physical_limit_dpos:.4f}m, drot={physical_limit_drot:.4f}rad")
    # 记录预处理信息
    if 'target_poses' in raw_demo and raw_demo['target_poses'] is not None:
        raw_len = len(raw_demo['target_poses'])
    elif 'eef_poses' in raw_demo and raw_demo['eef_poses'] is not None:
        raw_len = len(raw_demo['eef_poses'])
    else:
        raw_len = 0
    if 'target_poses' in processed_demo and processed_demo['target_poses'] is not None:
        proc_len = len(processed_demo['target_poses'])
    elif 'eef_poses' in processed_demo and processed_demo['eef_poses'] is not None:
        proc_len = len(processed_demo['eef_poses'])
    else:
        proc_len = 0
    logger.info(f"Source demo preprocessing: {raw_len} → {proc_len} frames "
                f"(inserted {proc_len - raw_len} frames for speed compliance)")
    logger.info("=" * 60)
    print(f"  Output: {config.output_dir}")
    print(f"  Log: {log_path}")
    
    # Collect episodes
    success_count = 0
    saved_count = 0  # 实际保存的 episode 数量（仅成功的）
    failure_stats = {}
    total_rewards = []
    
    # 具体的生成过程
    pbar = tqdm(range(config.num_episodes), desc="Generating", ncols=90)
    for episode_idx in pbar:
        # Get preprocessed source demo (overspeed segments already interpolated)
        src_demo = processed_demo

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
        
        # 从四元数提取欧拉角，获取绕 Z 轴的旋转角度（placement sampler 的 rotation）
        operated_euler = T.mat2euler(T.quat2mat(new_operated_obj_pose[3:]))
        non_operated_euler = T.mat2euler(T.quat2mat(new_non_operated_obj_pose[3:]))
        operated_z_rot = np.rad2deg(operated_euler[2])
        non_operated_z_rot = np.rad2deg(non_operated_euler[2])
        
        # 实时日志：记录场景配置
        logger.info(f"--- Episode {episode_idx}/{config.num_episodes} ---")
        logger.info(f"操作对象: {operated_obj_name}, 非操作对象: {non_operated_obj_name}")
        logger.info(f"EEF 初始位姿: pos={np.array2string(current_eef_pose[:3], precision=4)}")
        logger.info(f"新 {operated_obj_name} 位姿: pos={np.array2string(new_operated_obj_pose[:3], precision=4)}, "
                     f"quat={np.array2string(new_operated_obj_pose[3:], precision=4)}, "
                     f"euler={np.array2string(np.rad2deg(operated_euler), precision=1)}°, z_rot={operated_z_rot:.1f}°")
        logger.info(f"新 {non_operated_obj_name} 位姿: pos={np.array2string(new_non_operated_obj_pose[:3], precision=4)}, "
                     f"quat={np.array2string(new_non_operated_obj_pose[3:], precision=4)}, "
                     f"euler={np.array2string(np.rad2deg(non_operated_euler), precision=1)}°, z_rot={non_operated_z_rot:.1f}°")
        
        # 使用分段式轨迹变换
        new_target_poses, new_gripper_actions = traj_generator.transform_demo_to_new_scene(
            src_demo=src_demo,
            new_operated_obj_pose=new_operated_obj_pose,
            new_non_operated_obj_pose=new_non_operated_obj_pose,
            current_eef_pose=current_eef_pose,
            episode_idx=episode_idx,
        )
        
        # 实时日志：记录轨迹分段信息
        logger.info(f"轨迹生成完成: 总步数={len(new_target_poses)}")
        
        # Create waypoint policy
        policy = WaypointPolicy(
            env_interface=env,
            waypoint_poses=new_target_poses,
            gripper_actions=new_gripper_actions,
            limit_dpos=config.limit_dpos,
            limit_drot=config.limit_drot,
        )
        
        episode_data = collect_episode(
            env, 
            policy, 
            render=config.has_renderer,
            verbose=False,
            max_steps=config.horizon,
            skip_reset=True,
            initial_obs=obs,
            debug=args.debug,
            episode_idx=episode_idx,
        )
        
        # Track statistics
        is_success = episode_data["success"]
        if is_success:
            success_count += 1
            # 只保存成功的 episode，使用 saved_count 作为连续索引
            writer.write_episode(episode_data, saved_count)
            saved_count += 1

        # Debug: show executed vs planned EEF trajectories for this episode
        if args.debug:
            exec_pos = episode_data.get('_debug_executed_positions')
            plan_pos = episode_data.get('_debug_planned_positions')
            if exec_pos is not None or plan_pos is not None:
                fig = plt.figure(figsize=(12, 8))
                ax = fig.add_subplot(111, projection='3d')
                if plan_pos is not None:
                    ax.plot(plan_pos[:, 0], plan_pos[:, 1], plan_pos[:, 2],
                            '-o', color='C1', markersize=3, linewidth=1.2,
                            label=f'Planned ({len(plan_pos)} pts)', alpha=0.8)
                    # 标注每个点的 step 编号
                    step_interval = 1
                    for s in range(0, len(plan_pos), step_interval):
                        ax.text(plan_pos[s, 0], plan_pos[s, 1], plan_pos[s, 2],
                                f'{s}', fontsize=6, color='C1', alpha=0.9)
                    # 始终标注最后一个点
                    if (len(plan_pos) - 1) % step_interval != 0:
                        s = len(plan_pos) - 1
                        ax.text(plan_pos[s, 0], plan_pos[s, 1], plan_pos[s, 2],
                                f'{s}', fontsize=6, color='C1', alpha=0.9)
                if exec_pos is not None:
                    ax.plot(exec_pos[:, 0], exec_pos[:, 1], exec_pos[:, 2],
                            '-o', color='C0', markersize=3, linewidth=1.2,
                            label=f'Executed ({len(exec_pos)} pts)', alpha=0.8)
                    step_interval = 1
                    for s in range(0, len(exec_pos), step_interval):
                        ax.text(exec_pos[s, 0], exec_pos[s, 1], exec_pos[s, 2],
                                f'{s}', fontsize=6, color='C0', alpha=0.9)
                    if (len(exec_pos) - 1) % step_interval != 0:
                        s = len(exec_pos) - 1
                        ax.text(exec_pos[s, 0], exec_pos[s, 1], exec_pos[s, 2],
                                f'{s}', fontsize=6, color='C0', alpha=0.9)
                ax.set_xlabel('X (m)')
                ax.set_ylabel('Y (m)')
                ax.set_zlabel('Z (m)')
                ax.set_title(f'Episode {episode_idx}: Executed vs Planned EEF Trajectories')
                ax.legend(fontsize=9)
                plt.tight_layout()
                print(f"  [Debug] 显示 Episode {episode_idx} 的执行轨迹对比（阻塞，关闭窗口继续）")
                plt.show()

                # 保存逐步坐标为 CSV，方便离线分析
                csv_path = os.path.join(config.output_dir,
                                        f"debug_traj_ep{episode_idx}.csv")
                joint_pos_arr = episode_data.get('_debug_joint_positions')
                jnt_limits = episode_data.get('_debug_joint_limits')
                n_exec = len(exec_pos) if exec_pos is not None else 0
                n_plan = len(plan_pos) if plan_pos is not None else 0
                n_jnt = joint_pos_arr.shape[1] if joint_pos_arr is not None else 0
                n_rows = max(n_exec, n_plan)
                with open(csv_path, 'w') as f:
                    # --- Header ---
                    header_parts = [
                        "step",
                        "exec_x,exec_y,exec_z",
                        "plan_x,plan_y,plan_z",
                        "delta_x,delta_y,delta_z,delta_norm",
                    ]
                    for j in range(n_jnt):
                        header_parts.append(f"q{j}")
                    for j in range(n_jnt):
                        header_parts.append(f"q{j}_margin")  # min(q-lo, hi-q)
                    f.write(",".join(header_parts) + "\n")

                    # --- Joint limits comment line ---
                    if jnt_limits is not None:
                        lo_str = ",".join(f"{jnt_limits[j,0]:.4f}" for j in range(n_jnt))
                        hi_str = ",".join(f"{jnt_limits[j,1]:.4f}" for j in range(n_jnt))
                        f.write(f"# joint_lower:,,,,,,,,,,,{lo_str}\n")
                        f.write(f"# joint_upper:,,,,,,,,,,,{hi_str}\n")

                    # --- Data rows ---
                    for s in range(n_rows):
                        ex = exec_pos[s] if (exec_pos is not None and s < n_exec) else None
                        pl = plan_pos[s] if (plan_pos is not None and s < n_plan) else None
                        ex_str = f"{ex[0]:.6f},{ex[1]:.6f},{ex[2]:.6f}" if ex is not None else ",,"
                        pl_str = f"{pl[0]:.6f},{pl[1]:.6f},{pl[2]:.6f}" if pl is not None else ",,"
                        if ex is not None and pl is not None:
                            d = ex - pl
                            d_str = f"{d[0]:.6f},{d[1]:.6f},{d[2]:.6f},{np.linalg.norm(d):.6f}"
                        else:
                            d_str = ",,,"
                        # Joint data
                        if joint_pos_arr is not None and s < len(joint_pos_arr):
                            q = joint_pos_arr[s]
                            q_str = ",".join(f"{q[j]:.6f}" for j in range(n_jnt))
                            if jnt_limits is not None:
                                margins = [min(q[j] - jnt_limits[j, 0],
                                               jnt_limits[j, 1] - q[j])
                                           for j in range(n_jnt)]
                                m_str = ",".join(f"{m:.6f}" for m in margins)
                            else:
                                m_str = ",".join("" for _ in range(n_jnt))
                        else:
                            q_str = ",".join("" for _ in range(n_jnt))
                            m_str = ",".join("" for _ in range(n_jnt))
                        f.write(f"{s},{ex_str},{pl_str},{d_str},{q_str},{m_str}\n")
                print(f"  [Debug] 轨迹数据已保存: {csv_path}")
                if jnt_limits is not None:
                    print(f"  [Debug] 关节数: {n_jnt}, 关节范围已记录")
        
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
        saved_str = f" | 已保存={saved_count}" if is_success else ""
        clip_count = len(policy.clip_warnings)
        clip_str = f" | ⚠ 限距clipping={clip_count}次" if clip_count > 0 else ""
        logger.info(f"执行结果: {status_str} | 实际步数={actual_steps} | reward={episode_reward:.4f}{saved_str}{clip_str}")
        if clip_count > 0:
            logger.info(f"  限距警告 ({clip_count}次):")
            for warn in policy.clip_warnings:
                logger.info(f"    {warn}")
        if not is_success:
            if failure_reasons:
                for reason in failure_reasons:
                    logger.info(f"  失败原因: {reason}")
            else:
                logger.info(f"  失败原因: (未知，环境未提供具体原因)")
        
        # 实时日志：当前累计统计
        current_success_rate = 100 * success_count / (episode_idx + 1)
        logger.info(f"累计统计: 成功={success_count}/{episode_idx + 1} ({current_success_rate:.1f}%) | "
                     f"已保存={saved_count} | 平均reward={np.mean(total_rewards):.4f}")
        
        # 更新进度条
        pbar.set_postfix_str(
            f"saved={saved_count} succ={current_success_rate:.0f}%"
        )
    
    # Finalize
    writer.finalize()
    env.close()
    
    # Close any OpenCV windows
    if config.has_renderer:
        cv2.destroyAllWindows()
    
    # Print summary
    success_rate = 100 * success_count / config.num_episodes
    print(f"\n✓ Done! saved={saved_count}/{config.num_episodes} "
          f"succ={success_rate:.1f}% avg_r={np.mean(total_rewards):.1f}")
    print(f"  {config.output_dir}")
    
    # 日志：最终汇总
    logger.info("=" * 60)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total episodes attempted: {config.num_episodes}")
    logger.info(f"Successful episodes saved: {saved_count}/{config.num_episodes} "
                 f"({success_rate:.1f}%)")
    logger.info(f"Average reward: {np.mean(total_rewards):.3f} ± {np.std(total_rewards):.3f}")
    
    if failure_stats:
        logger.info("Failure Analysis:")
        for reason, count in sorted(failure_stats.items(), key=lambda x: x[1], reverse=True):
            percentage = 100 * count / config.num_episodes
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
    
    # Optional: debug mode (trajectory visualization)
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode: save trajectory visualizations for each episode")
    
    args = parser.parse_args()
    main(args)
