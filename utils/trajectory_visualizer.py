"""
trajectory_visualizer.py

用 Matplotlib 可视化生成的轨迹（debug 模式）。
支持：
  - 3D 分段轨迹（Approach / Grasp / Move）
  - 物体位置标注
  - EEF 朝向箭头
  - 夹爪状态色带
  - 超距点标红高亮
"""

import numpy as np
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import robosuite.utils.transform_utils as T


def _detect_overdist_points(all_poses, physical_limit_dpos, physical_limit_drot):
    """
    检测相邻帧之间超过物理空间限距阈值的轨迹点。
    
    对于第 i 段 [poses[i], poses[i+1]]，如果超距，则将 poses[i+1]
    （即"跳到"的那个点）标记为超距点。
    
    Args:
        all_poses: (N, 4, 4) 位姿矩阵序列
        physical_limit_dpos: 物理空间位置限距阈值（米），None 则不检查
        physical_limit_drot: 物理空间旋转限距阈值（弧度），None 则不检查
        
    Returns:
        overdist_mask: (N,) bool 数组，True 表示该点是超距目标点
        overdist_info: list of dict，每个超距点的详细信息
    """
    N = len(all_poses)
    overdist_mask = np.zeros(N, dtype=bool)
    overdist_info = []
    
    for i in range(N - 1):
        pos_delta = all_poses[i + 1][:3, 3] - all_poses[i][:3, 3]
        delta_rot_mat = all_poses[i + 1][:3, :3] @ all_poses[i][:3, :3].T
        delta_quat = T.mat2quat(delta_rot_mat)
        delta_aa = T.quat2axisangle(delta_quat)
        
        pos_exceed = False
        rot_exceed = False
        
        if physical_limit_dpos is not None:
            pos_max_component = np.max(np.abs(pos_delta))
            if pos_max_component > physical_limit_dpos:
                pos_exceed = True
        
        if physical_limit_drot is not None:
            rot_max_component = np.max(np.abs(delta_aa))
            if rot_max_component > physical_limit_drot:
                rot_exceed = True
        
        if pos_exceed or rot_exceed:
            overdist_mask[i + 1] = True
            overdist_info.append({
                'index': i + 1,
                'pos_delta': pos_delta,
                'rot_delta_aa': delta_aa,
                'pos_max': np.max(np.abs(pos_delta)),
                'rot_max': np.max(np.abs(delta_aa)),
                'pos_exceed': pos_exceed,
                'rot_exceed': rot_exceed,
            })
    
    return overdist_mask, overdist_info


def visualize_trajectory(
    approach_poses,
    grasp_poses,
    move_poses,
    approach_gripper,
    grasp_gripper,
    move_gripper,
    new_operated_obj_pose,
    new_non_operated_obj_pose,
    current_eef_pose,
    episode_idx: int,
    output_dir: str,
    operated_obj_name: str = "operated_obj",
    non_operated_obj_name: str = "non_operated_obj",
    src_poses=None,
    physical_limit_dpos=None,
    physical_limit_drot=None,
):
    """
    实时可视化单个 episode 的生成轨迹。
    
    弹出 Matplotlib 窗口，进程阻塞直到用户关闭窗口后继续。

    功能：
      - 3D 轨迹总览（分段着色 + 物体位置 + EEF 朝向）
      - 超距点标红高亮（物理空间增量超过 physical_limit_dpos / physical_limit_drot 的点）
    """

    # 提取位置
    approach_pos = np.array([p[:3, 3] for p in approach_poses])
    grasp_pos = np.array([p[:3, 3] for p in grasp_poses])
    move_pos = np.array([p[:3, 3] for p in move_poses])
    all_pos = np.concatenate([approach_pos, grasp_pos, move_pos], axis=0)
    all_poses_concat = np.concatenate([approach_poses, grasp_poses, move_poses], axis=0)

    operated_obj_pos = new_operated_obj_pose[:3, 3]
    non_operated_obj_pos = new_non_operated_obj_pose[:3, 3]
    eef_start_pos = current_eef_pose[:3, 3]

    # 检测超距点
    overdist_mask, overdist_info = _detect_overdist_points(
        all_poses_concat, physical_limit_dpos, physical_limit_drot)
    n_overdist = int(np.sum(overdist_mask))

    if n_overdist > 0:
        print(f"  [Debug] ⚠ 检测到 {n_overdist} 个超距点")
        for info in overdist_info:
            reasons = []
            if info['pos_exceed']:
                reasons.append(f"pos={info['pos_max']:.4f}>{physical_limit_dpos}")
            if info['rot_exceed']:
                reasons.append(f"rot={info['rot_max']:.4f}>{physical_limit_drot}")
            print(f"    Step {info['index']}: {', '.join(reasons)}")
    else:
        print("  [Debug] ⚠ 未检测到超距点")

    # ================================================================
    #  图 1: 3D 轨迹总览
    # ================================================================
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制分段轨迹
    ax.plot(approach_pos[:, 0], approach_pos[:, 1], approach_pos[:, 2],
            'o-', color='#2196F3', markersize=2, linewidth=1.5,
            label=f'Approach ({len(approach_pos)} steps)')
    ax.plot(grasp_pos[:, 0], grasp_pos[:, 1], grasp_pos[:, 2],
            'o-', color='#FF9800', markersize=2, linewidth=1.5,
            label=f'Grasp ({len(grasp_pos)} steps)')
    ax.plot(move_pos[:, 0], move_pos[:, 1], move_pos[:, 2],
            'o-', color='#4CAF50', markersize=2, linewidth=1.5,
            label=f'Move ({len(move_pos)} steps)')

    # 超距点标红高亮
    if n_overdist > 0:
        overdist_pos = all_pos[overdist_mask]
        ax.scatter(overdist_pos[:, 0], overdist_pos[:, 1], overdist_pos[:, 2],
                   color='red', s=60, marker='o', edgecolors='darkred',
                   linewidths=1.5, zorder=10, alpha=0.9,
                   label=f'Over-distance ({n_overdist} pts)')

    # 源 demo 轨迹（灰色虚线）
    if src_poses is not None:
        src_pos = np.array([p[:3, 3] for p in src_poses])
        ax.plot(src_pos[:, 0], src_pos[:, 1], src_pos[:, 2],
                '--', color='gray', linewidth=1.0, alpha=0.5,
                label=f'Source demo ({len(src_pos)} steps)')

    # 标注关键点
    ax.scatter(*eef_start_pos, color='blue', s=120, marker='^',
               edgecolors='black', linewidths=1, zorder=5, label='EEF Start')
    ax.scatter(*approach_pos[-1], color='#2196F3', s=80, marker='s',
               edgecolors='black', linewidths=1, zorder=5, label='Approach End')
    ax.scatter(*move_pos[-1], color='#4CAF50', s=80, marker='D',
               edgecolors='black', linewidths=1, zorder=5, label='Move End')

    # 标注物体位置
    ax.scatter(*operated_obj_pos, color='#FFD700', s=200, marker='*',
               edgecolors='black', linewidths=1, zorder=5,
               label=f'{operated_obj_name}')
    ax.scatter(*non_operated_obj_pos, color='#333333', s=200, marker='*',
               edgecolors='red', linewidths=1, zorder=5,
               label=f'{non_operated_obj_name}')

    # 绘制 EEF 朝向箭头（每隔 N 个点绘制一次）
    arrow_interval = max(len(all_pos) // 15, 1)
    for i in range(0, len(all_poses_concat), arrow_interval):
        pos = all_poses_concat[i][:3, 3]
        z_axis = all_poses_concat[i][:3, 2]  # EEF z 轴方向
        arrow_len = 0.02
        ax.quiver(pos[0], pos[1], pos[2],
                  z_axis[0] * arrow_len, z_axis[1] * arrow_len, z_axis[2] * arrow_len,
                  color='red', alpha=0.4, arrow_length_ratio=0.3, linewidth=0.8)

    # 绘制段之间的连接线（虚线）
    if len(approach_pos) > 0 and len(grasp_pos) > 0:
        ax.plot([approach_pos[-1, 0], grasp_pos[0, 0]],
                [approach_pos[-1, 1], grasp_pos[0, 1]],
                [approach_pos[-1, 2], grasp_pos[0, 2]],
                ':', color='gray', linewidth=1.0, alpha=0.6)
    if len(grasp_pos) > 0 and len(move_pos) > 0:
        ax.plot([grasp_pos[-1, 0], move_pos[0, 0]],
                [grasp_pos[-1, 1], move_pos[0, 1]],
                [grasp_pos[-1, 2], move_pos[0, 2]],
                ':', color='gray', linewidth=1.0, alpha=0.6)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    
    # 标题中包含超距信息
    overdist_title = f'  ⚠ Over-distance: {n_overdist} pts' if n_overdist > 0 else '  ✓ No over-distance'
    limit_title = ''
    if physical_limit_dpos is not None or physical_limit_drot is not None:
        parts = []
        if physical_limit_dpos is not None:
            parts.append(f'dpos≤{physical_limit_dpos:.4f}m')
        if physical_limit_drot is not None:
            parts.append(f'drot≤{physical_limit_drot:.4f}rad')
        limit_title = f'  Limits: {", ".join(parts)}'
    
    ax.set_title(f'Episode {episode_idx} — Generated Trajectory (3D)\n'
                 f'Total: {len(all_pos)} steps  |  '
                 f'Approach: {len(approach_pos)}  Grasp: {len(grasp_pos)}  Move: {len(move_pos)}\n'
                 f'{overdist_title}{limit_title}')
    ax.legend(loc='upper left', fontsize=8)

    # 设置合理的坐标范围
    _set_equal_aspect_3d(ax, all_pos, operated_obj_pos, non_operated_obj_pos, eef_start_pos)

    fig.tight_layout()

    # 实时弹窗显示，阻塞直到用户关闭窗口
    print(f"  [Debug] Episode {episode_idx} 轨迹可视化已弹出，关闭窗口后继续...")
    plt.show()  # 阻塞等待


def _set_equal_aspect_3d(ax, *point_arrays):
    """设置 3D 坐标轴等比例缩放，确保不变形。"""
    all_points = np.vstack([np.atleast_2d(p) for p in point_arrays])
    center = all_points.mean(axis=0)
    max_range = (all_points.max(axis=0) - all_points.min(axis=0)).max() / 2
    max_range = max(max_range, 0.05)  # 至少 5cm 范围
    margin = max_range * 0.15

    ax.set_xlim(center[0] - max_range - margin, center[0] + max_range + margin)
    ax.set_ylim(center[1] - max_range - margin, center[1] + max_range + margin)
    ax.set_zlim(center[2] - max_range - margin, center[2] + max_range + margin)
