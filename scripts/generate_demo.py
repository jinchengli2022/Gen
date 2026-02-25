import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import os

def make_pose(pos, rot_mat):
    """构建 4x4 齐次变换矩阵"""
    pose = np.eye(4)
    pose[:3, :3] = rot_mat
    pose[:3, 3] = pos
    return pose

def interp_poses(pose1, pose2, steps):
    """在两个 4x4 位姿之间进行线性插值 (位置线性，旋转 SLERP)"""
    pos1, pos2 = pose1[:3, 3], pose2[:3, 3]
    r1, r2 = R.from_matrix(pose1[:3, :3]), R.from_matrix(pose2[:3, :3])
    
    times = np.linspace(0, 1, steps)
    pos_interp = np.linspace(pos1, pos2, steps)
    
    # SLERP 插值
    slerp = R.concatenate([r1, r2])
    # scipy Slerp 需要时间点和对应的draw旋转
    from scipy.spatial.transform import Slerp
    slerp_func = Slerp([0, 1], slerp)
    rot_interp = slerp_func(times).as_matrix()
    
    poses = np.zeros((steps, 4, 4))
    poses[:, 3, 3] = 1.0
    poses[:, :3, :3] = rot_interp
    poses[:, :3, 3] = pos_interp
    return poses

def pose_to_7d(pose):
    """将 4x4 矩阵转换为 7D 向量 [x, y, z, x, y, z, w]"""
    pos = pose[:3, 3]
    quat_xyzw = R.from_matrix(pose[:3, :3]).as_quat()
    # 直接返回 xyzw 格式，这是 robosuite 默认的标准
    return np.concatenate([pos, quat_xyzw])

def draw_gripper(ax, pose, gripper_state, color='gray'):
    """
    按照以下约定绘制：
    Z 轴：夹爪朝向（指向指尖）
    X 轴：掌心方向（垂直于手指内侧平面）
    Y 轴：手指开合方向
    """
    # 夹爪基础尺寸
    finger_length = 0.08
    
    # 根据状态计算手指张开宽度
    width = 0.08 if gripper_state < 0 else 0.02
    half_w = width / 2.0
    
    # --- 关键修改：定义夹爪在局部坐标系下的关键点 ---
    # 底座中心
    base_center = np.array([0, 0, 0, 1])
    
    # 【修改点】底座横梁两端：现在沿 Y 轴展开（因为 X 是掌心方向，Y 才是开合方向）
    base_left = np.array([0, -half_w, 0, 1])
    base_right = np.array([0, half_w, 0, 1])
    
    # 【修改点】手指尖端：沿 Z 轴正方向延伸，位置保持在 Y 轴的两侧
    tip_left = np.array([0, -half_w, finger_length, 1])
    tip_right = np.array([0, half_w, finger_length, 1])
    
    # 转换到世界坐标系
    pts = np.array([base_center, base_left, base_right, tip_left, tip_right])
    pts_world = (pose @ pts.T).T
    
    # 绘制夹爪底座横梁 (连接两个手指的基部)
    ax.plot([pts_world[1, 0], pts_world[2, 0]], 
            [pts_world[1, 1], pts_world[2, 1]], 
            [pts_world[1, 2], pts_world[2, 2]], color=color, linewidth=3)
    
    # 绘制左手指 (从基部到尖端)
    ax.plot([pts_world[1, 0], pts_world[3, 0]], 
            [pts_world[1, 1], pts_world[3, 1]], 
            [pts_world[1, 2], pts_world[3, 2]], color=color, linewidth=3)
            
    # 绘制右手指 (从基部到尖端)
    ax.plot([pts_world[2, 0], pts_world[4, 0]], 
            [pts_world[2, 1], pts_world[4, 1]], 
            [pts_world[2, 2], pts_world[4, 2]], color=color, linewidth=3)
            
    # --- 绘制坐标轴辅助验证 ---
    axis_len = 0.05
    # 计算各轴在世界空间的方向
    # 这里的 pose[:3, 0] 是 X 轴，pose[:3, 1] 是 Y 轴，pose[:3, 2] 是 Z 轴
    origin = pts_world[0, :3]
    x_end = origin + pose[:3, 0] * axis_len
    y_end = origin + pose[:3, 1] * axis_len
    z_end = origin + pose[:3, 2] * axis_len
    
    ax.plot([origin[0], x_end[0]], [origin[1], x_end[1]], [origin[2], x_end[2]], color='r', linewidth=2, label='X (Palm)') 
    ax.plot([origin[0], y_end[0]], [origin[1], y_end[1]], [origin[2], y_end[2]], color='g', linewidth=2, label='Y (Open)') 
    ax.plot([origin[0], z_end[0]], [origin[1], z_end[1]], [origin[2], z_end[2]], color='b', linewidth=2, label='Z (Forward)')


def main():
    # ==========================================
    # 1. 定义输入参数 (统一管理)
    # ==========================================
    # 物体名称
    obj1_name = "yellow_cup"
    obj2_name = "black_cup"
    
    # 物体初始 4x4 位姿
    obj1_init_pose = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.15],
        [0.0, 0.0, 1.0, 0.8319],
        [0.0, 0.0, 0.0, 1.0]
    ])
    
    obj2_init_pose = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, -0.15],
        [0.0, 0.0, 1.0, 0.86],
        [0.0, 0.0, 0.0, 1.0]
    ])
    
    # 机械臂初始 6D 位姿 (4x4)
    # 注意：此值应与仿真环境中 UR5e 在 init_qpos 下的实际 EEF 位姿一致
    # 通过在仿真中 env.reset() 后读取 get_robot_eef_pose() 获得
    # 这里很奇怪？？？robosuite中关于机械爪的坐标系定义与之前遇到的不一样，它的X方向是垂直掌心的
    init_pose = np.array([
        [ 0.9953, -0.0672, -0.0703, -0.2332],
        [-0.0667, -0.9977,  0.0084, -0.0219],
        [-0.0707, -0.0037, -0.9975,  0.9692],
        [ 0.0,     0.0,     0.0,     1.0   ]
    ])
    
    # 相对于待操作物体的抓取姿态 (4x4)
    # 注意：这里的旋转是抓取时的旋转，与 init_pose 的旋转独立
    rel_grasp_pose = np.array([
        [ 0.0,  -1.0,  0.0,  0.0],
        [ -0.86,  0.0,  -0.5,  0.06],
        [ 0.5,  0.0, -0.86,  0.0],
        [ 0.0,  0.0,  0.0,  1.0]
    ])

    # 这个是最终两个物体的相对坐标
    rel_final_pose = np.array([
        [ 1.0,  0.0,  0.0,  0.0],
        [ 0.0,  0.707,  -0.707,  0.08],
        [ 0.0,  0.707, 0.707,  0.05],
        [ 0.0,  0.0,  0.0,  1.0]
    ])
    
    # 抓取高度 (标量，例如在物体上方 0.15m 处准备)
    grasp_height = 0.15
    
    # 轨迹步数设置
    steps_approach = 100
    steps_grasp = 100
    steps_move = 200
    
    # 移动阶段的弧形高度
    arc_height = 0.13
    
    # 输出路径
    out_dir = "source"
    out_filename = "gen_demo.hdf5"
    
    # ==========================================
    # 2. 计算关键点位姿
    # ==========================================
    # 目标抓取位姿 = 物体位姿 * 相对抓取位姿
    grasp_pose = obj1_init_pose @ rel_grasp_pose
    
    # 准备抓取位姿 (在抓取位姿正上方 grasp_height 处)
    pre_grasp_pose = grasp_pose.copy()
    pre_grasp_pose[2, 3] += grasp_height
    
    # 放置位姿 (移动到 obj2 上方)
    drop_pose = obj2_init_pose @ rel_final_pose @ rel_grasp_pose
    drop_pose[2, 3] += 0.1  # 放在 obj2 上方 10cm 处
    
    # ==========================================
    # 3. 生成轨迹 (分 3 个阶段)
    # ==========================================
    # 阶段 1: Approach (初始 -> 准备抓取)
    poses_approach = interp_poses(init_pose, pre_grasp_pose, steps_approach)
    gripper_approach = np.full((steps_approach, 1), -1.0)  # 张开
    
    # 阶段 2: Grasp (准备抓取 -> 抓取)
    poses_grasp = interp_poses(pre_grasp_pose, grasp_pose, steps_grasp)
    gripper_grasp = np.full((steps_grasp, 1), -1.0)
    gripper_grasp[-5:, 0] = 1.0  # 最后 5 步闭合夹爪
    
    # 阶段 3: Move (抓取 -> 放置，带弧形轨迹)
    poses_move = interp_poses(grasp_pose, drop_pose, steps_move)
    # 添加弧形高度 (Z 轴附加一个正弦波)
    times = np.linspace(0, 1, steps_move)
    poses_move[:, 2, 3] += arc_height * np.sin(np.pi * times)
    gripper_move = np.full((steps_move, 1), 1.0)  # 保持闭合
    
    # 合并轨迹
    target_poses = np.concatenate([poses_approach, poses_grasp, poses_move], axis=0)
    gripper_actions = np.concatenate([gripper_approach, gripper_grasp, gripper_move], axis=0)
    total_steps = target_poses.shape[0]
    
    # 转换为 7D 格式
    target_poses_7d = np.array([pose_to_7d(p) for p in target_poses])
    eef_poses = target_poses.copy()  # 假设实际执行位姿完美跟踪目标位姿
    
    # ==========================================
    # 4. 计算物体轨迹
    # ==========================================
    obj1_poses_7d = np.zeros((total_steps, 7))
    obj2_poses_7d = np.zeros((total_steps, 7))
    
    # obj2 始终静止
    obj2_poses_7d[:] = pose_to_7d(obj2_init_pose)
    
    # obj1 在前两个阶段静止
    idx_grasp_end = steps_approach + steps_grasp
    obj1_poses_7d[:idx_grasp_end] = pose_to_7d(obj1_init_pose)
    
    # obj1 在第三阶段随夹爪移动 (obj_pose = eef_pose * rel_grasp_pose^-1)
    rel_grasp_inv = np.linalg.inv(rel_grasp_pose)
    for i in range(idx_grasp_end, total_steps):
        current_obj_pose = eef_poses[i] @ rel_grasp_inv
        obj1_poses_7d[i] = pose_to_7d(current_obj_pose)
        
    # ==========================================
    # 5. 生成 Signals
    # ==========================================
    # subtask_term_signals: 各个阶段对应的 step 取值为 1
    term_approach = np.zeros(total_steps, dtype=np.int32)
    term_approach[:steps_approach] = 1
    
    term_grasp = np.zeros(total_steps, dtype=np.int32)
    term_grasp[steps_approach:idx_grasp_end] = 1
    
    term_move = np.zeros(total_steps, dtype=np.int32)
    term_move[idx_grasp_end:] = 1
    
    # subtask_object_signals: 操作物体的 step 取值为 1
    sig_obj1 = np.ones(total_steps, dtype=np.int32)   # yellow_cup 是操作对象
    sig_obj2 = np.zeros(total_steps, dtype=np.int32)  # black_cup 不是
    
    # ==========================================
    # 6. 保存为 HDF5
    # ==========================================
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, out_filename)
    
    with h5py.File(out_path, 'w') as f:
        demo_grp = f.create_group('data/demo_0/datagen_info')
        
        # 基础数据
        demo_grp.create_dataset('eef_pose', data=eef_poses)
        demo_grp.create_dataset('gripper_action', data=gripper_actions)
        demo_grp.create_dataset('target_pose', data=target_poses)
        demo_grp.create_dataset('target_pose_7d', data=target_poses_7d)
        
        # 物体位姿35
        obj_grp = demo_grp.create_group('object_poses')
        obj_grp.create_dataset(obj1_name, data=obj1_poses_7d)
        obj_grp.create_dataset(obj2_name, data=obj2_poses_7d)
        
        # Subtask Info
        sub_grp = demo_grp.create_group('subtask_info')
        
        term_grp = sub_grp.create_group('subtask_term_signals')
        term_grp.create_dataset('approach', data=term_approach)
        term_grp.create_dataset('grasp', data=term_grasp)
        term_grp.create_dataset('move', data=term_move)
        
        obj_sig_grp = sub_grp.create_group('subtask_object_signals')
        obj_sig_grp.create_dataset(obj1_name, data=sig_obj1)
        obj_sig_grp.create_dataset(obj2_name, data=sig_obj2)
        
    print(f"成功生成合成 Demo，已保存至: {out_path}")
    
    # ==========================================
    # 7. 3D 可视化
    # ==========================================
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制三个阶段的轨迹
    ax.plot(poses_approach[:, 0, 3], poses_approach[:, 1, 3], poses_approach[:, 2, 3], 
            c='r', linewidth=2, label='Phase 1: Approach')
    ax.plot(poses_grasp[:, 0, 3], poses_grasp[:, 1, 3], poses_grasp[:, 2, 3], 
            c='g', linewidth=2, label='Phase 2: Grasp')
    ax.plot(poses_move[:, 0, 3], poses_move[:, 1, 3], poses_move[:, 2, 3], 
            c='b', linewidth=2, label='Phase 3: Move (Arc)')
            
    # 绘制夹爪示意图 (每隔一定步数绘制一个)
    step_interval = 15
    for i in range(0, total_steps, step_interval):
        draw_gripper(ax, target_poses[i], gripper_actions[i, 0], color='gray')
    # 绘制最后一个位姿的夹爪
    draw_gripper(ax, target_poses[-1], gripper_actions[-1, 0], color='black')
    
    # 绘制物体初始位置
    ax.scatter(*obj1_init_pose[:3, 3], c='y', s=200, marker='*', label=f'{obj1_name} (Start)')
    ax.scatter(*obj2_init_pose[:3, 3], c='k', s=200, marker='s', label=f'{obj2_name}')
    
    # 绘制物体1的最终位置
    ax.scatter(*obj1_poses_7d[-1, :3], c='orange', s=200, marker='*', alpha=0.5, label=f'{obj1_name} (End)')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Synthetic Demonstration Trajectory')
    ax.legend()
    
    # 确保坐标轴比例一致，避免夹爪变形
    ax.set_box_aspect([1, 1, 1])
    
    # 获取当前坐标轴的范围并设置为相同比例，保证 X、Y、Z 轴刻度一致
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    
    # 调整视角
    ax.view_init(elev=20, azim=45)
    
    plt.savefig(os.path.join(out_dir, "trajectory_vis.png"))
    print(f"轨迹可视化已保存至: {os.path.join(out_dir, 'trajectory_vis.png')}")
    plt.show()

if __name__ == "__main__":
    main()
