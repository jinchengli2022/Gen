import h5py
import numpy as np
import robosuite.utils.transform_utils as T

# 直接以读写模式打开原文件
with h5py.File('source/pouring_demo.hdf5', 'a') as f:
    # 遍历所有的 demo 组
    for demo_name in f['data'].keys():
        demo_group = f['data'][demo_name]
        
        # 检查是否已经存在 target_poses_7d，如果存在则跳过或删除
        if 'target_poses_7d' in demo_group['datagen_info']:
            print(f"[{demo_name}] target_poses_7d 已存在，正在覆盖...")
            del demo_group['datagen_info']['target_poses_7d']
            
        # 读取原来的 4x4 格式数据
        target_poses = demo_group['datagen_info']['target_pose'][:]  # (N, 4, 4)

        # 将 4x4 格式转换为 7D 格式 (x, y, z, qw, qx, qy, qz)
        N = target_poses.shape[0]
        new_target_poses_7d = np.zeros((N, 7))
        
        for i in range(N):
            # 提取位置 (x, y, z)
            pos = target_poses[i][:3, 3]
            
            # 提取旋转矩阵并转换为四元数 (qw, qx, qy, qz)
            rot_mat = target_poses[i][:3, :3]
            quat = T.mat2quat(rot_mat)  # robosuite 的 mat2quat 返回 (w, x, y, z)
            
            # 拼接为 7D 向量
            new_target_poses_7d[i] = np.concatenate([pos, quat])

        # 将新的 7D 数据保存回原文件的 datagen_info 组中
        demo_group['datagen_info'].create_dataset('target_pose_7d', data=new_target_poses_7d)
        print(f"[{demo_name}] 成功添加 target_pose_7d，形状: {new_target_poses_7d.shape}")

print("所有 demo 处理完成！")