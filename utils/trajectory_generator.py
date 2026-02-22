"""
trajectory_generator.py

Trajectory generation utilities inspired by MimicGen.
Implements pose transformation and interpolation for data generation.
"""

import numpy as np
import robosuite.utils.transform_utils as T


class PoseUtils:
    """Utility functions for pose manipulation."""
    
    @staticmethod
    def make_pose(pos, rot_mat):
        """Create 4x4 pose matrix from position and rotation."""
        pose = np.eye(4)
        pose[:3, :3] = rot_mat
        pose[:3, 3] = pos
        return pose
    
    @staticmethod
    def unmake_pose(pose):
        """Extract position and rotation from 4x4 pose matrix."""
        if pose.shape == (7,):  # (x,y,z, qw,qx,qy,qz)
            return pose[:3], T.quat2mat(pose[3:])
        return pose[:3, 3], pose[:3, :3]
    
    @staticmethod
    def pose_inv(pose):
        """Compute inverse of 4x4 pose matrix."""
        inv_pose = np.eye(4)
        inv_pose[:3, :3] = pose[:3, :3].T
        inv_pose[:3, 3] = -pose[:3, :3].T @ pose[:3, 3]
        return inv_pose


def interpolate_poses(start_pose, end_pose, steps):
    """
    Interpolate between two poses using linear interpolation for position
    and Slerp for rotation.
    
    Args:
        start_pose: Starting 4x4 pose matrix
        end_pose: Ending 4x4 pose matrix
        steps: Number of interpolation steps
        
    Returns:
        Array of interpolated poses (steps, 4, 4)
    """
    pos_start, rot_start = PoseUtils.unmake_pose(start_pose)
    pos_end, rot_end = PoseUtils.unmake_pose(end_pose)
    
    # Linear position interpolation
    pos_seq = np.linspace(pos_start, pos_end, steps)
    
    # Quaternion Slerp for rotation
    quat_start = T.mat2quat(rot_start)
    quat_end = T.mat2quat(rot_end)
    
    # Handle quaternion double cover (choose shortest path)
    if np.dot(quat_start, quat_end) < 0:
        quat_end = -quat_end
    
    # NLERP (Normalized Linear Interpolation)
    quats = []
    for t in np.linspace(0, 1, steps):
        q = (1 - t) * quat_start + t * quat_end
        q_norm = np.linalg.norm(q)
        if q_norm > 1e-6:
            q = q / q_norm
        quats.append(q)
    
    # Reconstruct poses
    poses = []
    for i in range(steps):
        poses.append(PoseUtils.make_pose(pos_seq[i], T.quat2mat(quats[i])))
    
    return np.array(poses)


def slerp(q1, q2, t):
    """
    Spherical linear interpolation between two quaternions.
    
    Args:
        q1, q2: Quaternions (w, x, y, z)
        t: Interpolation parameter [0, 1]
        
    Returns:
        Interpolated quaternion
    """
    dot_product = np.dot(q1, q2)
    
    # Take shorter path
    if dot_product < 0.0:
        q2 = -q2
        dot_product = -dot_product
    
    dot_product = np.clip(dot_product, -1.0, 1.0)
    theta = np.arccos(dot_product)
    sin_theta = np.sin(theta)
    
    # Handle nearly parallel quaternions
    if sin_theta < 1e-6:
        return (1 - t) * q1 + t * q2
    
    # Slerp formula
    w1 = np.sin((1 - t) * theta) / sin_theta
    w2 = np.sin(t * theta) / sin_theta
    
    return w1 * q1 + w2 * q2


class TrajectoryGenerator:
    """
    Generates robot trajectories by transforming source demonstrations
    to adapt to new object configurations.
    """
    
    def __init__(self, env_interface):
        """
        Args:
            env_interface: Environment interface for getting poses
        """
        self.env_interface = env_interface
    
    def generate_grasp_trajectory(
        self, 
        target_object_pose,
        grasp_pose_in_object,
        pre_grasp_height=0.3,
        num_approach_steps=100,
        num_grasp_steps=100,
        num_wait_steps=30
    ):
        """
        Generate a grasping trajectory (approach -> grasp -> close gripper).
        
        Args:
            target_object_pose: 4x4 pose of object to grasp
            grasp_pose_in_object: 4x4 relative grasp pose in object frame
            pre_grasp_height: Height above object for pre-grasp pose (meters)
            num_approach_steps: Steps for approach motion
            num_grasp_steps: Steps for grasping motion
            num_wait_steps: Steps to wait after grasping
            
        Returns:
            poses: (N, 4, 4) array of EEF poses
            gripper_actions: (N, 1) array of gripper commands (-1: open, 1: close)
        """
        current_eef_pose = self.env_interface.get_robot_eef_pose()
        if current_eef_pose.shape == (7,):
            current_eef_pose = PoseUtils.make_pose(
                current_eef_pose[:3], 
                T.quat2mat(current_eef_pose[3:])
            )
        
        if target_object_pose.shape == (7,):
            target_object_pose = PoseUtils.make_pose(
                target_object_pose[:3],
                T.quat2mat(target_object_pose[3:])
            )
        
        # Pre-grasp pose (above object)
        pre_grasp_in_object = grasp_pose_in_object.copy()
        pre_grasp_in_object[2, 3] += pre_grasp_height
        pre_grasp_world = target_object_pose @ pre_grasp_in_object
        
        # Grasp pose
        grasp_world = target_object_pose @ grasp_pose_in_object
        
        # Generate trajectory segments
        approach_traj = interpolate_poses(current_eef_pose, pre_grasp_world, num_approach_steps)
        grasp_traj = interpolate_poses(pre_grasp_world, grasp_world, num_grasp_steps)
        wait_traj = np.tile(grasp_world, (num_wait_steps, 1, 1))
        
        # Combine trajectories
        full_traj = np.concatenate([approach_traj, grasp_traj, wait_traj], axis=0)
        
        # Gripper actions: open during approach/grasp, close during wait
        gripper_actions = np.concatenate([
            np.full((num_approach_steps + num_grasp_steps, 1), -1.0),  # Open
            np.full((num_wait_steps, 1), 1.0)  # Close
        ], axis=0)
        
        return full_traj, gripper_actions
    
    def transform_trajectory_to_new_scene(
        self,
        src_eef_poses,
        src_object_pose,
        target_object_pose,
        current_eef_pose,
        target_end_eef_pose=None
    ):
        """
        Transform a source trajectory to adapt to new object configuration
        using vector scaling algorithm.
        
        Args:
            src_eef_poses: (N, 4, 4) source EEF trajectory
            src_object_pose: 4x4 source object pose
            target_object_pose: 4x4 target object pose in new scene
            current_eef_pose: 4x4 current robot EEF pose
            target_end_eef_pose: 4x4 desired end pose (optional, computed if None)
            
        Returns:
            transformed_poses: (N, 4, 4) transformed EEF trajectory
        """
        # Ensure matrix format
        def ensure_mat(pose):
            if pose.shape == (7,):
                return PoseUtils.make_pose(pose[:3], T.quat2mat(pose[3:]))
            return pose
        
        src_eef_poses = np.array([ensure_mat(p) for p in src_eef_poses])
        src_object_pose = ensure_mat(src_object_pose)
        target_object_pose = ensure_mat(target_object_pose)
        current_eef_pose = ensure_mat(current_eef_pose)
        
        # Compute target end pose if not provided
        if target_end_eef_pose is None:
            # Maintain relative pose from source
            src_end_pose = src_eef_poses[-1]
            rel_pose = PoseUtils.pose_inv(src_object_pose) @ src_end_pose
            target_end_eef_pose = target_object_pose @ rel_pose
        else:
            target_end_eef_pose = ensure_mat(target_end_eef_pose)
        
        num_steps = len(src_eef_poses)
        
        # Extract positions
        src_positions = np.array([p[:3, 3] for p in src_eef_poses])
        src_start_pos = src_positions[0]
        src_end_pos = src_positions[-1]
        target_start_pos = current_eef_pose[:3, 3]
        target_end_pos = target_end_eef_pose[:3, 3]
        
        # Compute XY plane transformation
        src_vec = src_end_pos[:2] - src_start_pos[:2]
        target_vec = target_end_pos[:2] - target_start_pos[:2]
        
        src_dist = np.linalg.norm(src_vec)
        target_dist = np.linalg.norm(target_vec)
        
        scale = 1.0
        angle_diff = 0.0
        
        if src_dist > 1e-6:
            scale = target_dist / src_dist
            src_angle = np.arctan2(src_vec[1], src_vec[0])
            target_angle = np.arctan2(target_vec[1], target_vec[0])
            angle_diff = target_angle - src_angle
        
        # 2D rotation matrix
        cos_a, sin_a = np.cos(angle_diff), np.sin(angle_diff)
        R_2d = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        
        # Transform positions
        target_delta_z = target_end_pos[2] - target_start_pos[2]
        src_delta_z = src_end_pos[2] - src_start_pos[2]
        
        transformed_positions = np.zeros((num_steps, 3))
        for i in range(num_steps):
            rel_pos = src_positions[i] - src_start_pos
            
            # XY: scale and rotate
            rel_pos_xy = R_2d @ rel_pos[:2] * scale
            
            # Z: preserve relative motion + linear correction
            progress = i / (num_steps - 1) if num_steps > 1 else 1.0
            z_correction = (target_delta_z - src_delta_z) * progress
            rel_pos_z = rel_pos[2] + z_correction
            
            transformed_positions[i] = target_start_pos + np.array([
                rel_pos_xy[0], rel_pos_xy[1], rel_pos_z
            ])
        
        # Transform rotations using Slerp
        start_quat = T.mat2quat(current_eef_pose[:3, :3])
        end_quat = T.mat2quat(target_end_eef_pose[:3, :3])
        
        transformed_quats = []
        for i in range(num_steps):
            t = i / (num_steps - 1) if num_steps > 1 else 1.0
            transformed_quats.append(slerp(start_quat, end_quat, t))
        
        # Reconstruct poses
        transformed_poses = []
        for i in range(num_steps):
            transformed_poses.append(PoseUtils.make_pose(
                transformed_positions[i],
                T.quat2mat(transformed_quats[i])
            ))
        
        return np.array(transformed_poses)
