"""
source_preprocessor.py

对源 demo 数据进行预处理：检测超距段并插值，使其满足限距约束。

限距体系：
    机器人控制器有固有的 output_max（pos/rot），用于将物理空间增量归一化到 [-1, 1]。
    用户在 config 中设置 limit_dpos / limit_drot（归一化空间阈值，范围 [0, 1]）。
    物理空间限距 = limit * output_max。
    本预处理器接收的 physical_limit_dpos / physical_limit_drot
    即为已计算好的物理空间限距阈值（单位：米/弧度）。

工作原理：
    遍历 target_poses 的每对相邻帧，计算位置增量和旋转增量。
    若任一分量超过对应物理空间限距阈值，则在该段内均匀插入额外的中间帧，
    使插值后每帧间的增量均不超过限距。
    同时对 eef_poses、gripper_actions、object_poses 以及
    subtask_term_signals / subtask_object_signals 做同步插值 / 扩展。
"""

import numpy as np
from copy import deepcopy
from scipy.spatial.transform import Rotation, Slerp
import robosuite.utils.transform_utils as T


class SourceDemoPreprocessor:
    """
    对源 demo 进行限距预处理。
    
    检查 target_poses 中相邻帧之间的位移 / 旋转增量是否超过
    物理空间限距阈值（physical_limit = limit * output_max），
    如果超距则在该段内做细分插值，
    保证处理后的轨迹在每一步都满足限距约束。
    
    Usage:
        # physical limits = config.limit_dpos * output_max_pos
        preprocessor = SourceDemoPreprocessor(
            physical_limit_dpos=0.0425,
            physical_limit_drot=0.15,
        )
        processed_demo = preprocessor.process(src_demo)
    """

    def __init__(self, physical_limit_dpos: float = 0.05,
                 physical_limit_drot: float = 0.15,
                 verbose: bool = True):
        """
        Args:
            physical_limit_dpos: 每步最大位置增量（米），= limit_dpos * output_max_pos
            physical_limit_drot: 每步最大旋转增量（弧度），= limit_drot * output_max_rot
            verbose: 是否打印处理信息
        """
        self.physical_limit_dpos = physical_limit_dpos
        self.physical_limit_drot = physical_limit_drot
        self.verbose = verbose

    # ----------------------------------------------------------------
    #  公开接口
    # ----------------------------------------------------------------

    def process(self, src_demo: dict) -> dict:
        """
        对源 demo 进行限距预处理。
        
        Args:
            src_demo: SourceDemoLoader.get_demo() 返回的 dict，包含：
                - target_poses: (N, 4, 4) 或 (N, 7)
                - eef_poses: (N, 4, 4) 或 (N, 7)   [可选]
                - gripper_actions: (N, 1)
                - object_poses: {obj_name: (N, 7)}
                - subtask_term_signals: {name: (N,)}
                - subtask_object_signals: {obj_name: (N,)}
                - actions, rewards, dones 等（若存在也会同步处理）
                
        Returns:
            processed_demo: 深拷贝后经过插值的新 demo dict，
                           结构与输入一致，帧数 >= 原始帧数
        """
        demo = deepcopy(src_demo)

        # --- 获取 target_poses（统一为 4x4 矩阵） ---
        target_poses = self._ensure_mat_array(demo.get('target_poses'))
        if target_poses is None:
            target_poses = self._ensure_mat_array(demo.get('eef_poses'))
        if target_poses is None:
            raise ValueError("源 demo 中未找到 target_poses 或 eef_poses，无法进行预处理")

        N = len(target_poses)
        if N < 2:
            if self.verbose:
                print("[SourceDemoPreprocessor] 轨迹长度 < 2，无需预处理")
            return demo

        # --- 计算每段需要的细分数 ---
        subdivisions = self._compute_subdivisions(target_poses)

        total_new_frames = int(np.sum(subdivisions)) + 1  # subdivisions[i] 是第 i 段的帧数（不含终点），最后 +1 补上末帧
        total_inserted = total_new_frames - N

        if total_inserted == 0:
            if self.verbose:
                print(f"[SourceDemoPreprocessor] 所有帧均满足限距 "
                      f"(physical_limit_dpos={self.physical_limit_dpos}, "
                      f"physical_limit_drot={self.physical_limit_drot})，无需插值")
            return demo

        if self.verbose:
            overspeed_count = int(np.sum(subdivisions > 1))
            print(f"[SourceDemoPreprocessor] 检测到 {overspeed_count}/{N-1} 段超距")
            print(f"  原始帧数: {N} → 插值后帧数: {total_new_frames} (插入 {total_inserted} 帧)")
            print(f"  限距: physical_limit_dpos={self.physical_limit_dpos}, "
                  f"physical_limit_drot={self.physical_limit_drot}")

        # --- 构建帧映射：new_idx -> (src_seg_idx, alpha) ---
        # 其中 src_seg_idx 是源 demo 中的段索引 [i, i+1]，alpha ∈ [0, 1)
        frame_map = self._build_frame_map(subdivisions, N)

        # --- 插值各数据通道 ---
        # 1. target_poses
        new_target_poses = self._interpolate_poses(target_poses, frame_map)
        if 'target_poses' in demo and demo['target_poses'] is not None:
            demo['target_poses'] = self._to_original_format(
                new_target_poses, src_demo.get('target_poses'))

        # 2. eef_poses
        if 'eef_poses' in demo and demo['eef_poses'] is not None:
            eef_poses = self._ensure_mat_array(demo['eef_poses'])
            new_eef_poses = self._interpolate_poses(eef_poses, frame_map)
            demo['eef_poses'] = self._to_original_format(
                new_eef_poses, src_demo.get('eef_poses'))

        # 3. gripper_actions — 阶梯式复制（每段内保持源帧的值）
        if 'gripper_actions' in demo and demo['gripper_actions'] is not None:
            demo['gripper_actions'] = self._expand_step_hold(
                demo['gripper_actions'], frame_map)

        # 4. object_poses — 对每个物体做位姿插值
        if 'object_poses' in demo:
            for obj_name, obj_poses in demo['object_poses'].items():
                if obj_poses is not None and len(obj_poses) == N:
                    obj_mat = self._ensure_mat_array(obj_poses)
                    new_obj_mat = self._interpolate_poses(obj_mat, frame_map)
                    demo['object_poses'][obj_name] = self._to_original_format(
                        new_obj_mat, obj_poses)

        # 5. subtask_term_signals — 阶梯式扩展（保持段归属）
        if 'subtask_term_signals' in demo:
            for sig_name, signal in demo['subtask_term_signals'].items():
                if signal is not None and len(signal) == N:
                    demo['subtask_term_signals'][sig_name] = self._expand_signal(
                        signal, frame_map)

        # 6. subtask_object_signals — 同上
        if 'subtask_object_signals' in demo:
            for obj_name, signal in demo['subtask_object_signals'].items():
                if signal is not None and len(signal) == N:
                    demo['subtask_object_signals'][obj_name] = self._expand_signal(
                        signal, frame_map)

        # 7. actions — 阶梯式细分（每段内复制源帧 action，幅度按细分数缩放）
        if 'actions' in demo and demo['actions'] is not None and len(demo['actions']) == N:
            demo['actions'] = self._expand_actions(demo['actions'], frame_map, subdivisions)

        # 8. rewards / dones — 阶梯式扩展
        for key in ['rewards', 'dones']:
            if key in demo and demo[key] is not None and len(demo[key]) == N:
                demo[key] = self._expand_step_hold(demo[key], frame_map)

        # --- 打印超距段详情 ---
        if self.verbose:
            self._print_overspeed_details(target_poses, subdivisions)

        return demo

    # ----------------------------------------------------------------
    #  内部方法
    # ----------------------------------------------------------------

    def _compute_subdivisions(self, poses):
        """
        计算每对相邻帧之间需要的细分段数。
        
        Args:
            poses: (N, 4, 4) 位姿矩阵序列
            
        Returns:
            subdivisions: (N-1,) int 数组，subdivisions[i] 表示第 i 段
                          [poses[i], poses[i+1]] 需要被分成多少小段。
                          若 =1 则无需细分。
        """
        N = len(poses)
        subdivisions = np.ones(N - 1, dtype=int)

        for i in range(N - 1):
            pos_delta = poses[i + 1][:3, 3] - poses[i][:3, 3]
            # 旋转增量：轴角表示
            delta_rot_mat = poses[i + 1][:3, :3] @ poses[i][:3, :3].T
            delta_quat = T.mat2quat(delta_rot_mat)
            delta_axisangle = T.quat2axisangle(delta_quat)

            # 所需细分数 = max(各分量 / 限距) 的 ceil
            pos_ratio = np.max(np.abs(pos_delta) / self.physical_limit_dpos) if self.physical_limit_dpos > 0 else 0
            rot_ratio = np.max(np.abs(delta_axisangle) / self.physical_limit_drot) if self.physical_limit_drot > 0 else 0

            ratio = max(pos_ratio, rot_ratio)
            subdivisions[i] = max(int(np.ceil(ratio)), 1)

        return subdivisions

    def _build_frame_map(self, subdivisions, N):
        """
        构建帧映射列表。
        
        Returns:
            frame_map: list of (src_seg_idx, alpha)
                       长度 = sum(subdivisions) + 1
                       src_seg_idx: 源段索引 i, 对应 [poses[i], poses[i+1]]
                       alpha: 在该段内的归一化位置 [0, 1]
        """
        frame_map = []
        for i, n_sub in enumerate(subdivisions):
            for j in range(n_sub):
                alpha = j / n_sub
                frame_map.append((i, alpha))
        # 最后一帧：末尾 pose
        frame_map.append((len(subdivisions) - 1, 1.0))
        return frame_map

    def _interpolate_poses(self, poses, frame_map):
        """
        根据 frame_map 对 (N, 4, 4) 位姿序列进行插值。
        位置线性插值，旋转 Slerp 插值。
        
        Returns:
            new_poses: (M, 4, 4) 插值后的位姿序列
        """
        M = len(frame_map)
        new_poses = np.zeros((M, 4, 4))
        new_poses[:, 3, 3] = 1.0

        for k, (seg_idx, alpha) in enumerate(frame_map):
            i = seg_idx
            j = min(i + 1, len(poses) - 1)

            if alpha == 0.0:
                new_poses[k] = poses[i]
            elif alpha == 1.0:
                new_poses[k] = poses[j]
            else:
                # 位置线性插值
                pos = (1 - alpha) * poses[i][:3, 3] + alpha * poses[j][:3, 3]
                # 旋转 Slerp
                r_start = Rotation.from_matrix(poses[i][:3, :3])
                r_end = Rotation.from_matrix(poses[j][:3, :3])
                slerp_func = Slerp([0, 1], Rotation.concatenate([r_start, r_end]))
                rot = slerp_func(alpha).as_matrix()
                new_poses[k, :3, :3] = rot
                new_poses[k, :3, 3] = pos

        return new_poses

    def _expand_step_hold(self, data, frame_map):
        """
        阶梯式扩展：每个新帧取其所属源段起始帧的值。
        适用于 gripper_actions、rewards、dones 等离散量。
        
        Args:
            data: (N, ...) 原始数据
            frame_map: 帧映射列表
            
        Returns:
            new_data: (M, ...) 扩展后数据
        """
        M = len(frame_map)
        shape = list(data.shape)
        shape[0] = M
        new_data = np.zeros(shape, dtype=data.dtype)

        for k, (seg_idx, alpha) in enumerate(frame_map):
            if alpha == 1.0:
                # 段终点 → 取下一帧的值（即 seg_idx + 1）
                src_idx = min(seg_idx + 1, len(data) - 1)
            else:
                src_idx = seg_idx
            new_data[k] = data[src_idx]

        return new_data

    def _expand_signal(self, signal, frame_map):
        """
        扩展 subtask 信号（0/1 值）。
        
        策略：每段内的所有插入帧都继承源段起始帧的信号值。
        最后一帧（alpha=1.0 的）继承源段终点帧的信号值。
        
        Args:
            signal: (N,) 原始信号
            frame_map: 帧映射列表
            
        Returns:
            new_signal: (M,) 扩展后信号
        """
        M = len(frame_map)
        new_signal = np.zeros(M, dtype=signal.dtype)

        for k, (seg_idx, alpha) in enumerate(frame_map):
            if alpha == 1.0:
                src_idx = min(seg_idx + 1, len(signal) - 1)
            else:
                src_idx = seg_idx
            new_signal[k] = signal[src_idx]

        return new_signal

    def _expand_actions(self, actions, frame_map, subdivisions):
        """
        扩展 actions：每段内的 action 按细分数均分。
        
        如果源 demo 的第 i 帧 action 被细分为 n_sub 段，
        则每个子段的 action = 原 action / n_sub（位移/旋转部分均分）。
        
        Args:
            actions: (N, D) 原始 action
            frame_map: 帧映射列表
            subdivisions: (N-1,) 细分数
            
        Returns:
            new_actions: (M, D) 扩展后 action（M = sum(subdivisions) + 1）
        """
        M = len(frame_map)
        D = actions.shape[1]
        new_actions = np.zeros((M, D), dtype=actions.dtype)

        for k, (seg_idx, alpha) in enumerate(frame_map):
            if alpha == 1.0:
                src_idx = min(seg_idx + 1, len(actions) - 1)
                new_actions[k] = actions[src_idx]
            else:
                n_sub = subdivisions[seg_idx]
                # 位移/旋转类的 action 均分
                new_actions[k] = actions[seg_idx] / n_sub

        return new_actions

    def _print_overspeed_details(self, poses, subdivisions):
        """打印超距段的详细信息。"""
        for i in range(len(subdivisions)):
            if subdivisions[i] > 1:
                pos_delta = poses[i + 1][:3, 3] - poses[i][:3, 3]
                delta_rot_mat = poses[i + 1][:3, :3] @ poses[i][:3, :3].T
                delta_quat = T.mat2quat(delta_rot_mat)
                delta_aa = T.quat2axisangle(delta_quat)

                pos_max = np.max(np.abs(pos_delta))
                rot_max = np.max(np.abs(delta_aa))
                print(f"  超距段 [{i}→{i+1}]: "
                      f"pos_max={pos_max:.4f}(>{self.physical_limit_dpos}), "
                      f"rot_max={rot_max:.4f}(>{self.physical_limit_drot}), "
                      f"细分为 {subdivisions[i]} 段")

    # ----------------------------------------------------------------
    #  格式转换辅助
    # ----------------------------------------------------------------

    def _ensure_mat_array(self, data):
        """
        将位姿数据统一转换为 (N, 4, 4) 矩阵数组。
        
        Args:
            data: (N, 4, 4) 或 (N, 7) 或 None
            
        Returns:
            (N, 4, 4) np.ndarray 或 None
        """
        if data is None:
            return None
        data = np.array(data)
        if data.ndim == 3 and data.shape[1:] == (4, 4):
            return data
        elif data.ndim == 2 and data.shape[1] == 7:
            # (N, 7) → (N, 4, 4)
            mats = np.zeros((len(data), 4, 4))
            mats[:, 3, 3] = 1.0
            for i in range(len(data)):
                mats[i, :3, :3] = T.quat2mat(data[i, 3:])
                mats[i, :3, 3] = data[i, :3]
            return mats
        else:
            raise ValueError(f"无法将 shape={data.shape} 转换为 (N, 4, 4) 位姿矩阵")

    def _to_original_format(self, new_mat_poses, original_data):
        """
        将 (M, 4, 4) 位姿矩阵转回原始数据格式。
        
        如果原始数据是 (N, 7) 则转为 7D，否则保持 4x4。
        """
        original_data = np.array(original_data)
        if original_data.ndim == 2 and original_data.shape[1] == 7:
            # 转回 (M, 7) 格式
            M = len(new_mat_poses)
            result = np.zeros((M, 7))
            for i in range(M):
                result[i, :3] = new_mat_poses[i, :3, 3]
                result[i, 3:] = T.mat2quat(new_mat_poses[i, :3, :3])
            return result
        else:
            return new_mat_poses
