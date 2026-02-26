"""
trajectory_generator.py

基于分段策略的轨迹变换工具。
将源 demo 的轨迹根据新场景中物体位置进行变换，分为三段：
  - Approach: 根据新物体位置重算终点，保持源 demo 速度线性插值
  - Grasp:    根据新物体位置进行齐次旋转矩阵变换
  - Move:     根据源 demo 中两物体最终相对位置计算新终点，XY 缩放 + Z 保高 + Slerp 旋转
"""

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
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
        if pose.shape == (7,):  # (x,y,z, qx,qy,qz,qw) — robosuite xyzw
            return pose[:3].copy(), T.quat2mat(pose[3:])
        return pose[:3, 3].copy(), pose[:3, :3].copy()
    
    @staticmethod
    def pose_inv(pose):
        """Compute inverse of 4x4 pose matrix."""
        inv_pose = np.eye(4)
        inv_pose[:3, :3] = pose[:3, :3].T
        inv_pose[:3, 3] = -pose[:3, :3].T @ pose[:3, 3]
        return inv_pose
    
    @staticmethod
    def ensure_mat(pose):
        """确保输入为 4x4 矩阵格式。"""
        if pose.shape == (4, 4):
            return pose.copy()
        elif pose.shape == (7,):
            return PoseUtils.make_pose(pose[:3], T.quat2mat(pose[3:]))
        else:
            raise ValueError(f"Invalid pose shape: {pose.shape}, expected (4,4) or (7,)")
    
    @staticmethod
    def pose_to_7d(pose):
        """将 4x4 矩阵转换为 7D 向量 [x,y,z, qx,qy,qz,qw] (robosuite xyzw 格式)。"""
        pos = pose[:3, 3]
        quat = T.mat2quat(pose[:3, :3])  # robosuite returns xyzw
        return np.concatenate([pos, quat])


def interpolate_poses_slerp(start_pose, end_pose, steps):
    """
    在两个 4x4 位姿之间进行插值（位置线性，旋转 Slerp）。
    
    Args:
        start_pose: 起始 4x4 位姿矩阵
        end_pose: 终点 4x4 位姿矩阵
        steps: 插值步数
        
    Returns:
        (steps, 4, 4) 插值后的位姿序列
    """
    pos_start, rot_start = PoseUtils.unmake_pose(start_pose)
    pos_end, rot_end = PoseUtils.unmake_pose(end_pose)
    
    # 位置线性插值
    pos_seq = np.linspace(pos_start, pos_end, steps)
    
    # 旋转 Slerp 插值（使用 scipy）
    r_start = Rotation.from_matrix(rot_start)
    r_end = Rotation.from_matrix(rot_end)
    slerp_func = Slerp([0, 1], Rotation.concatenate([r_start, r_end]))
    times = np.linspace(0, 1, steps)
    rot_seq = slerp_func(times).as_matrix()
    
    # 组装位姿
    poses = np.zeros((steps, 4, 4))
    poses[:, 3, 3] = 1.0
    poses[:, :3, :3] = rot_seq
    poses[:, :3, 3] = pos_seq
    return poses


# 保留旧名称的向后兼容别名
interpolate_poses = interpolate_poses_slerp


def slerp(q1, q2, t):
    """
    Spherical linear interpolation between two quaternions.
    
    Args:
        q1, q2: Quaternions (xyzw format, robosuite convention)
        t: Interpolation parameter [0, 1]
        
    Returns:
        Interpolated quaternion (xyzw)
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
        q = (1 - t) * q1 + t * q2
        return q / np.linalg.norm(q)
    
    # Slerp formula
    w1 = np.sin((1 - t) * theta) / sin_theta
    w2 = np.sin(t * theta) / sin_theta
    
    return w1 * q1 + w2 * q2


class TrajectoryGenerator:
    """
    基于分段策略的轨迹生成器。
    
    从源 demo 中读取三段轨迹（Approach/Grasp/Move），
    根据新场景中的物体位置变换生成新轨迹。
    """
    
    def __init__(self, env_interface=None):
        """
        Args:
            env_interface: 环境接口（用于获取当前 EEF 位姿等），可选
        """
        self.env_interface = env_interface
        self.debug = False          # 是否启用 debug 可视化
        self.debug_output_dir = ""  # debug 图片输出目录
        self.physical_limit_dpos = None  # 物理空间限距阈值（米，用于 debug 可视化标注超距点）
        self.physical_limit_drot = None  # 物理空间限距阈值（弧度）
    
    # ----------------------------------------------------------------
    #  核心入口：分段式轨迹变换
    # ----------------------------------------------------------------
    
    def transform_demo_to_new_scene(
        self,
        src_demo: dict,
        new_operated_obj_pose,
        new_non_operated_obj_pose,
        current_eef_pose=None,
        episode_idx: int = 0,
    ):
        """
        将源 demo 变换到新的场景配置。
        
        流程：
            1. 从 subtask_term_signals 和 subtask_object_signals 解析各段边界及操作对象
            2. 对 Approach / Grasp / Move 分别应用变换策略
            3. 拼接并返回完整新轨迹
        
        Args:
            src_demo: SourceDemoLoader.get_demo() 返回的 dict，包含：
                - target_poses / eef_poses: (N, 4, 4)
                - gripper_actions: (N, 1)
                - object_poses: {obj_name: (N, 7)}  xyzw 格式
                - subtask_term_signals: {approach: (N,), grasp: (N,), move: (N,)}
                - subtask_object_signals: {obj_name: (N,)}
            new_operated_obj_pose: 新场景中被操作物体的位姿 (4x4 或 7D)
            new_non_operated_obj_pose: 新场景中非操作物体的位姿 (4x4 或 7D)
            current_eef_pose: 当前 EEF 位姿 (4x4 或 7D)，若 None 则从 env_interface 获取
            
        Returns:
            new_target_poses: (M, 4, 4) 变换后的目标位姿序列
            new_gripper_actions: (M, 1) 夹爪动作序列
        """
        # --- 0. 准备输入 ---
        new_operated_obj_pose = PoseUtils.ensure_mat(new_operated_obj_pose)
        new_non_operated_obj_pose = PoseUtils.ensure_mat(new_non_operated_obj_pose)
        
        if current_eef_pose is None:
            if self.env_interface is not None:
                current_eef_pose = self.env_interface.get_robot_eef_pose()
            else:
                raise ValueError("current_eef_pose 未提供，且 env_interface 为 None")
        current_eef_pose = PoseUtils.ensure_mat(current_eef_pose)
        
        # 获取源 demo 数据
        src_poses = self._get_src_poses(src_demo)        # (N, 4, 4)
        src_gripper = src_demo['gripper_actions']         # (N, 1)
        
        # --- 1. 解析段边界 ---
        seg_info = self._parse_segments(src_demo)
        approach_range = seg_info['approach']   # (start, end) 索引
        grasp_range = seg_info['grasp']
        move_range = seg_info['move']
        
        operated_obj_name = seg_info['operated_obj_name']
        non_operated_obj_name = seg_info['non_operated_obj_name']
        
        # print(f"[TrajectoryGenerator] 分段解析完成:")
        # print(f"  操作对象: {operated_obj_name}, 非操作对象: {non_operated_obj_name}")
        # print(f"  Approach: steps [{approach_range[0]}, {approach_range[1]})")
        # print(f"  Grasp:    steps [{grasp_range[0]}, {grasp_range[1]})")
        # print(f"  Move:     steps [{move_range[0]}, {move_range[1]})")
        
        # --- 2. 提取源 demo 中物体位姿 ---
        src_operated_obj_poses = src_demo['object_poses'][operated_obj_name]      # (N, 7)
        src_non_operated_obj_poses = src_demo['object_poses'][non_operated_obj_name]  # (N, 7)
        
        # 源 demo 中操作物体初始位姿 (取 approach 段开头)
        src_operated_obj_init = PoseUtils.ensure_mat(src_operated_obj_poses[approach_range[0]])
        # 源 demo 中非操作物体初始位姿
        src_non_operated_obj_init = PoseUtils.ensure_mat(src_non_operated_obj_poses[approach_range[0]])
        
        # --- 3. 分段变换 ---
        # 3a. Approach 段
        src_approach_poses = src_poses[approach_range[0]:approach_range[1]]
        src_approach_gripper = src_gripper[approach_range[0]:approach_range[1]]
        
        new_approach_poses, new_approach_gripper = self._transform_approach(
            src_approach_poses=src_approach_poses,
            src_approach_gripper=src_approach_gripper,
            src_operated_obj_pose=src_operated_obj_init,
            new_operated_obj_pose=new_operated_obj_pose,
            current_eef_pose=current_eef_pose,
        )
        
        # 3b. Grasp 段
        src_grasp_poses = src_poses[grasp_range[0]:grasp_range[1]]
        src_grasp_gripper = src_gripper[grasp_range[0]:grasp_range[1]]
        
        new_grasp_poses, new_grasp_gripper = self._transform_grasp(
            src_grasp_poses=src_grasp_poses,
            src_grasp_gripper=src_grasp_gripper,
            src_operated_obj_pose=src_operated_obj_init,
            new_operated_obj_pose=new_operated_obj_pose,
        )
        
        # 3c. Move 段
        src_move_poses = src_poses[move_range[0]:move_range[1]]
        src_move_gripper = src_gripper[move_range[0]:move_range[1]]
        
        # 源 demo 中两物体在 Move 段末尾的位姿
        src_operated_end = PoseUtils.ensure_mat(src_operated_obj_poses[move_range[1] - 1])
        src_non_operated_end = PoseUtils.ensure_mat(src_non_operated_obj_poses[move_range[1] - 1])
        
        new_move_poses, new_move_gripper = self._transform_move(
            src_move_poses=src_move_poses,
            src_move_gripper=src_move_gripper,
            src_operated_obj_start=src_operated_obj_init,
            src_non_operated_obj_start=src_non_operated_obj_init,
            src_operated_obj_end=src_operated_end,
            src_non_operated_obj_end=src_non_operated_end,
            new_operated_obj_start=new_operated_obj_pose,
            new_non_operated_obj_start=new_non_operated_obj_pose,
        )
        
        # --- 4. 拼接 ---
        new_target_poses = np.concatenate([
            new_approach_poses, new_grasp_poses, new_move_poses
        ], axis=0)
        new_gripper_actions = np.concatenate([
            new_approach_gripper, new_grasp_gripper, new_move_gripper
        ], axis=0)
        
        # print(f"[TrajectoryGenerator] 新轨迹生成完成: "
        #       f"Approach {len(new_approach_poses)} + "
        #       f"Grasp {len(new_grasp_poses)} + "
        #       f"Move {len(new_move_poses)} = "
        #       f"{len(new_target_poses)} steps")
        
        # Debug 可视化
        if self.debug:
            from utils.trajectory_visualizer import visualize_trajectory
            visualize_trajectory(
                approach_poses=new_approach_poses,
                grasp_poses=new_grasp_poses,
                move_poses=new_move_poses,
                approach_gripper=new_approach_gripper,
                grasp_gripper=new_grasp_gripper,
                move_gripper=new_move_gripper,
                new_operated_obj_pose=new_operated_obj_pose,
                new_non_operated_obj_pose=new_non_operated_obj_pose,
                current_eef_pose=current_eef_pose,
                episode_idx=episode_idx,
                output_dir=self.debug_output_dir,
                operated_obj_name=operated_obj_name,
                non_operated_obj_name=non_operated_obj_name,
                src_poses=src_poses,
                physical_limit_dpos=self.physical_limit_dpos,
                physical_limit_drot=self.physical_limit_drot,
            )
        
        return new_target_poses, new_gripper_actions
    
    # ----------------------------------------------------------------
    #  Approach 段变换
    # ----------------------------------------------------------------
    
    def _transform_approach(
        self,
        src_approach_poses,
        src_approach_gripper,
        src_operated_obj_pose,
        new_operated_obj_pose,
        current_eef_pose,
    ):
        """
        Approach 段变换逻辑：
        
        1. 源 demo 中 Approach 终点相对于操作物体的相对位姿 = T_obj^{-1} @ T_end
        2. 新 Approach 终点 = T_obj_new @ 相对位姿
        3. 计算源 demo 速度 = 总距离 / 轨迹点数
        4. 计算新 Approach 的总距离，按相同速度得到新的点数
        5. 线性位置 + Slerp 旋转插值
        
        Returns:
            new_poses: (M, 4, 4), new_gripper: (M, 1)
        """
        n_src = len(src_approach_poses)
        
        # 源 Approach 终点
        src_approach_end = src_approach_poses[-1]
        
        # 终点相对于操作物体的相对位姿
        rel_end_to_obj = PoseUtils.pose_inv(src_operated_obj_pose) @ src_approach_end
        
        # 新 Approach 终点
        new_approach_end = new_operated_obj_pose @ rel_end_to_obj
        
        # 源 Approach 速度（位置总距离 / 点数）
        src_positions = np.array([p[:3, 3] for p in src_approach_poses])
        src_dists = np.linalg.norm(np.diff(src_positions, axis=0), axis=1)
        src_total_dist = np.sum(src_dists)
        src_speed = src_total_dist / n_src if n_src > 0 else 0.01
        
        # 新 Approach 总距离
        new_total_dist = np.linalg.norm(new_approach_end[:3, 3] - current_eef_pose[:3, 3])
        
        # 新步数（保持速度一致）
        new_steps = max(int(round(new_total_dist / src_speed)), 2)
        
        # print(f"  [Approach] 源距离={src_total_dist:.4f}, 源步数={n_src}, 速度={src_speed:.6f}")
        # print(f"  [Approach] 新距离={new_total_dist:.4f}, 新步数={new_steps}")
        
        # 插值
        new_poses = interpolate_poses_slerp(current_eef_pose, new_approach_end, new_steps)
        
        # 夹爪：Approach 段一般全程张开，用源 demo 最多出现的值填充
        dominant_gripper = np.median(src_approach_gripper)
        new_gripper = np.full((new_steps, 1), dominant_gripper)
        
        return new_poses, new_gripper
    
    # ----------------------------------------------------------------
    #  Grasp 段变换
    # ----------------------------------------------------------------
    
    def _transform_grasp(
        self,
        src_grasp_poses,
        src_grasp_gripper,
        src_operated_obj_pose,
        new_operated_obj_pose,
    ):
        """
        Grasp 段变换逻辑：
        
        根据操作物体新旧位姿的齐次变换矩阵，将 Grasp 段的每个 EEF 位姿
        进行刚体变换：
            T_new = T_obj_new @ T_obj_old^{-1} @ T_old
        
        这样保持了 EEF 相对于物体的相对关系不变。
        
        Grasp 段步数不变，夹爪动作直接复制。
        
        Returns:
            new_poses: (N, 4, 4), new_gripper: (N, 1)
        """
        n_src = len(src_grasp_poses)
        
        # 变换矩阵：T_obj_new @ T_obj_old^{-1}
        transform = new_operated_obj_pose @ PoseUtils.pose_inv(src_operated_obj_pose)
        
        new_poses = np.zeros_like(src_grasp_poses)
        for i in range(n_src):
            new_poses[i] = transform @ src_grasp_poses[i]
        
        # print(f"  [Grasp] 步数={n_src}，刚体变换完成")
        
        # 夹爪动作直接复制
        new_gripper = src_grasp_gripper.copy()
        
        return new_poses, new_gripper
    
    # ----------------------------------------------------------------
    #  Move 段变换
    # ----------------------------------------------------------------
    
    def _transform_move(
        self,
        src_move_poses,
        src_move_gripper,
        src_operated_obj_start,
        src_non_operated_obj_start,
        src_operated_obj_end,
        src_non_operated_obj_end,
        new_operated_obj_start,
        new_non_operated_obj_start,
    ):
        """
        Move 段变换逻辑：
        
        1. 计算源 demo 中两物体最终相对位姿：
           rel_final = T_non_operated_end^{-1} @ T_operated_end
        2. 新 Move 终点的操作物体位姿：
           T_operated_new_end = T_non_operated_new @ rel_final
        3. 从 operated 的新终点推算 EEF 的新终点：
           保持 EEF 相对操作物体的相对位姿不变
        4. 位置变换 — XY 缩放 + Z 保高：
           - XY 平面：按 new_xy_dist / src_xy_dist 等比缩放
           - Z 轴：保留源 demo 中 Move 段的最高点高度（arc_height_max），
             但起止点 Z 根据实际场景调整
        5. 旋转变换 — Slerp 插值（起点旋转→终点旋转）
        6. 步数：保持源 demo 速度，按新距离缩放步数
        
        Returns:
            new_poses: (M, 4, 4), new_gripper: (M, 1)
        """
        n_src = len(src_move_poses)
        
        # --- 1. 计算源 demo 中两物体最终相对位姿 ---
        rel_final = PoseUtils.pose_inv(src_non_operated_obj_end) @ src_operated_obj_end
        
        # --- 2. 新操作物体的 Move 终点 ---
        new_operated_obj_end = new_non_operated_obj_start @ rel_final
        
        # --- 3. EEF 新起点 & 终点 ---
        # EEF 相对于操作物体的相对位姿（在 Move 段起点和终点处）
        rel_eef_start = PoseUtils.pose_inv(src_operated_obj_start) @ src_move_poses[0]
        rel_eef_end = PoseUtils.pose_inv(src_operated_obj_end) @ src_move_poses[-1]
        
        new_move_start = new_operated_obj_start @ rel_eef_start
        new_move_end = new_operated_obj_end @ rel_eef_end
        
        # --- 4. 分析源 demo Move 段的位置特征 ---
        src_positions = np.array([p[:3, 3] for p in src_move_poses])
        src_start_pos = src_positions[0]
        src_end_pos = src_positions[-1]
        
        new_start_pos = new_move_start[:3, 3]
        new_end_pos = new_move_end[:3, 3]
        
        # 源 demo XY 平面距离
        src_xy_vec = src_end_pos[:2] - src_start_pos[:2]
        src_xy_dist = np.linalg.norm(src_xy_vec)
        
        # 新 XY 平面距离
        new_xy_vec = new_end_pos[:2] - new_start_pos[:2]
        new_xy_dist = np.linalg.norm(new_xy_vec)
        
        # XY 缩放比例
        xy_scale = new_xy_dist / src_xy_dist if src_xy_dist > 1e-6 else 1.0
        
        # XY 旋转角
        if src_xy_dist > 1e-6 and new_xy_dist > 1e-6:
            src_angle = np.arctan2(src_xy_vec[1], src_xy_vec[0])
            new_angle = np.arctan2(new_xy_vec[1], new_xy_vec[0])
            angle_diff = new_angle - src_angle
        else:
            angle_diff = 0.0
        
        cos_a, sin_a = np.cos(angle_diff), np.sin(angle_diff)
        R_2d = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        
        # --- Z 轴处理：保留 arc 最高点 ---
        # 源 demo 中 Z 的最高点（绝对值）
        src_z_max = np.max(src_positions[:, 2])
        # 源 demo Z 的 profile（相对于线性基线的偏差）
        src_z_baseline = np.linspace(src_start_pos[2], src_end_pos[2], n_src)
        src_z_arc = src_positions[:, 2] - src_z_baseline  # arc 形状
        # arc 峰值
        src_arc_peak = np.max(src_z_arc)
        
        # --- 5. 源 demo 速度 ---
        src_dists = np.linalg.norm(np.diff(src_positions, axis=0), axis=1)
        src_total_dist = np.sum(src_dists)
        src_speed = src_total_dist / n_src if n_src > 0 else 0.01
        
        # 粗略估算新距离（XY 缩放 + Z 不变）
        new_total_dist_approx = src_total_dist * xy_scale  # 近似
        new_steps = max(int(round(new_total_dist_approx / src_speed)), 2)
        
        # print(f"  [Move] 源XY距离={src_xy_dist:.4f}, 新XY距离={new_xy_dist:.4f}, "
        #       f"缩放={xy_scale:.4f}, 旋转={np.degrees(angle_diff):.1f}°")
        # print(f"  [Move] 源Z最高={src_z_max:.4f}, arc峰值={src_arc_peak:.4f}")
        # print(f"  [Move] 源步数={n_src}, 新步数={new_steps}")
        
        # --- 6. 生成新位置 ---
        new_positions = np.zeros((new_steps, 3))
        new_z_baseline = np.linspace(new_start_pos[2], new_end_pos[2], new_steps)
        
        for i in range(new_steps):
            t = i / (new_steps - 1) if new_steps > 1 else 1.0
            
            # XY：对源 demo 的相对位移进行缩放和旋转
            # 在源 demo 中找到对应的归一化时间位置
            src_idx_f = t * (n_src - 1)
            src_idx_lo = int(np.floor(src_idx_f))
            src_idx_hi = min(src_idx_lo + 1, n_src - 1)
            alpha = src_idx_f - src_idx_lo
            
            # 源 demo 在此归一化时间的相对 XY 位移
            src_rel_xy_lo = src_positions[src_idx_lo, :2] - src_start_pos[:2]
            src_rel_xy_hi = src_positions[src_idx_hi, :2] - src_start_pos[:2]
            src_rel_xy = (1 - alpha) * src_rel_xy_lo + alpha * src_rel_xy_hi
            
            # 缩放 + 旋转
            new_rel_xy = R_2d @ (src_rel_xy * xy_scale)
            new_positions[i, 0] = new_start_pos[0] + new_rel_xy[0]
            new_positions[i, 1] = new_start_pos[1] + new_rel_xy[1]
            
            # Z：baseline + arc 形状（保留源 demo 的 arc 形状）
            src_arc_lo = src_z_arc[src_idx_lo]
            src_arc_hi = src_z_arc[src_idx_hi]
            src_arc_val = (1 - alpha) * src_arc_lo + alpha * src_arc_hi
            new_positions[i, 2] = new_z_baseline[i] + src_arc_val
        
        # --- 7. 生成新旋转（Slerp） ---
        start_quat = T.mat2quat(new_move_start[:3, :3])
        end_quat = T.mat2quat(new_move_end[:3, :3])
        
        new_quats = []
        for i in range(new_steps):
            t = i / (new_steps - 1) if new_steps > 1 else 1.0
            new_quats.append(slerp(start_quat, end_quat, t))
        
        # --- 8. 组装位姿 ---
        new_poses = np.zeros((new_steps, 4, 4))
        new_poses[:, 3, 3] = 1.0
        for i in range(new_steps):
            new_poses[i, :3, :3] = T.quat2mat(new_quats[i])
            new_poses[i, :3, 3] = new_positions[i]
        
        # --- 9. 夹爪 ---
        # Move 段夹爪通常恒为闭合，直接用源 demo 的模式
        dominant_gripper = np.median(src_move_gripper)
        new_gripper = np.full((new_steps, 1), dominant_gripper)
        
        return new_poses, new_gripper
    
    # ----------------------------------------------------------------
    #  辅助方法
    # ----------------------------------------------------------------
    
    def _get_src_poses(self, src_demo):
        """从 demo dict 中获取 (N, 4, 4) 的源位姿。优先使用 target_poses。"""
        if 'target_poses' in src_demo and src_demo['target_poses'] is not None:
            poses = src_demo['target_poses']
        elif 'eef_poses' in src_demo and src_demo['eef_poses'] is not None:
            poses = src_demo['eef_poses']
        else:
            raise ValueError("源 demo 中未找到 target_poses 或 eef_poses")
        
        # 确保为 4x4 矩阵格式
        if len(poses.shape) == 2 and poses.shape[1] == 7:
            poses = np.array([PoseUtils.ensure_mat(p) for p in poses])
        
        return poses
    
    def _parse_segments(self, src_demo):
        """
        从 subtask_term_signals 和 subtask_object_signals 解析各段边界及操作对象。
        
        Returns:
            dict: {
                'approach': (start, end),  # 前闭后开索引
                'grasp': (start, end),
                'move': (start, end),
                'operated_obj_name': str,
                'non_operated_obj_name': str,
            }
        """
        term_signals = src_demo.get('subtask_term_signals', {})
        obj_signals = src_demo.get('subtask_object_signals', {})
        
        if not term_signals:
            raise ValueError("源 demo 中缺少 subtask_term_signals，无法解析分段")
        
        result = {}
        
        # 解析各段索引
        for seg_name in ['approach', 'grasp', 'move']:
            if seg_name not in term_signals:
                raise ValueError(f"subtask_term_signals 中缺少 '{seg_name}' 信号")
            signal = term_signals[seg_name]
            indices = np.where(signal == 1)[0]
            if len(indices) == 0:
                raise ValueError(f"'{seg_name}' 信号中没有 step 取值为 1")
            result[seg_name] = (int(indices[0]), int(indices[-1]) + 1)
        
        # 解析操作对象
        if obj_signals:
            operated_obj_name = None
            non_operated_obj_name = None
            for obj_name, signal in obj_signals.items():
                if np.any(signal == 1):
                    operated_obj_name = obj_name
                else:
                    non_operated_obj_name = obj_name
            
            if operated_obj_name is None:
                # fallback: 第一个有信号的是操作对象
                obj_names = list(obj_signals.keys())
                operated_obj_name = obj_names[0]
                non_operated_obj_name = obj_names[1] if len(obj_names) > 1 else None
        else:
            # 从 object_poses 的 key 推断（简单假设第一个是操作对象）
            obj_names = list(src_demo.get('object_poses', {}).keys())
            if len(obj_names) < 2:
                raise ValueError(f"需要至少 2 个物体的位姿，当前只有: {obj_names}")
            operated_obj_name = obj_names[0]
            non_operated_obj_name = obj_names[1]
            # print(f"  [Warning] 未找到 subtask_object_signals，"
            #       f"默认 '{operated_obj_name}' 为操作对象")
        
        if non_operated_obj_name is None:
            obj_names = list(src_demo.get('object_poses', {}).keys())
            for name in obj_names:
                if name != operated_obj_name:
                    non_operated_obj_name = name
                    break
        
        if non_operated_obj_name is None:
            raise ValueError("无法确定非操作物体名称")
        
        result['operated_obj_name'] = operated_obj_name
        result['non_operated_obj_name'] = non_operated_obj_name
        
        return result
    
    # ----------------------------------------------------------------
    #  旧接口保留（向后兼容）
    # ----------------------------------------------------------------
    
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
        生成一个抓取轨迹（approach -> grasp -> close gripper）。
        保留旧接口，供独立使用。
        """
        current_eef_pose = self.env_interface.get_robot_eef_pose()
        current_eef_pose = PoseUtils.ensure_mat(current_eef_pose)
        target_object_pose = PoseUtils.ensure_mat(target_object_pose)
        
        # Pre-grasp pose (above object)
        pre_grasp_in_object = grasp_pose_in_object.copy()
        pre_grasp_in_object[2, 3] += pre_grasp_height
        pre_grasp_world = target_object_pose @ pre_grasp_in_object
        
        # Grasp pose
        grasp_world = target_object_pose @ grasp_pose_in_object
        
        # Generate trajectory segments
        approach_traj = interpolate_poses_slerp(current_eef_pose, pre_grasp_world, num_approach_steps)
        grasp_traj = interpolate_poses_slerp(pre_grasp_world, grasp_world, num_grasp_steps)
        wait_traj = np.tile(grasp_world, (num_wait_steps, 1, 1))
        
        # Combine trajectories
        full_traj = np.concatenate([approach_traj, grasp_traj, wait_traj], axis=0)
        
        # Gripper actions: open during approach/grasp, close during wait
        gripper_actions = np.concatenate([
            np.full((num_approach_steps + num_grasp_steps, 1), -1.0),
            np.full((num_wait_steps, 1), 1.0)
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
        旧接口：对整段轨迹进行向量缩放变换。
        保留向后兼容，新代码请使用 transform_demo_to_new_scene()。
        """
        def ensure_mat(pose):
            return PoseUtils.ensure_mat(pose)
        
        src_eef_poses = np.array([ensure_mat(p) for p in src_eef_poses])
        src_object_pose = ensure_mat(src_object_pose)
        target_object_pose = ensure_mat(target_object_pose)
        current_eef_pose = ensure_mat(current_eef_pose)
        
        if target_end_eef_pose is None:
            src_end_pose = src_eef_poses[-1]
            rel_pose = PoseUtils.pose_inv(src_object_pose) @ src_end_pose
            target_end_eef_pose = target_object_pose @ rel_pose
        else:
            target_end_eef_pose = ensure_mat(target_end_eef_pose)
        
        num_steps = len(src_eef_poses)
        src_positions = np.array([p[:3, 3] for p in src_eef_poses])
        src_start_pos = src_positions[0]
        src_end_pos = src_positions[-1]
        target_start_pos = current_eef_pose[:3, 3]
        target_end_pos = target_end_eef_pose[:3, 3]
        
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
        
        cos_a, sin_a = np.cos(angle_diff), np.sin(angle_diff)
        R_2d = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        
        target_delta_z = target_end_pos[2] - target_start_pos[2]
        src_delta_z = src_end_pos[2] - src_start_pos[2]
        
        transformed_positions = np.zeros((num_steps, 3))
        for i in range(num_steps):
            rel_pos = src_positions[i] - src_start_pos
            rel_pos_xy = R_2d @ rel_pos[:2] * scale
            progress = i / (num_steps - 1) if num_steps > 1 else 1.0
            z_correction = (target_delta_z - src_delta_z) * progress
            rel_pos_z = rel_pos[2] + z_correction
            transformed_positions[i] = target_start_pos + np.array([
                rel_pos_xy[0], rel_pos_xy[1], rel_pos_z
            ])
        
        start_quat = T.mat2quat(current_eef_pose[:3, :3])
        end_quat = T.mat2quat(target_end_eef_pose[:3, :3])
        transformed_quats = []
        for i in range(num_steps):
            t = i / (num_steps - 1) if num_steps > 1 else 1.0
            transformed_quats.append(slerp(start_quat, end_quat, t))
        
        transformed_poses = []
        for i in range(num_steps):
            transformed_poses.append(PoseUtils.make_pose(
                transformed_positions[i],
                T.quat2mat(transformed_quats[i])
            ))
        
        return np.array(transformed_poses)
