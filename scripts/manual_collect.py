"""
manual_collect.py

交互式数据采集脚本：通过外部输入末端执行器 (EEF) 的目标位姿来控制机械手，
并将采集到的轨迹保存为 HDF5 数据集。

使用方法:
    python scripts/manual_collect.py --config configs/examples/pouring_water_trajgen.json --render
    python scripts/manual_collect.py --config configs/examples/pouring_water_trajgen.json --output data/manual_collected

操作说明:
    - 每一步输入末端夹爪的目标位置 (x y z) 和姿态四元数 (qw qx qy qz)，以及夹爪开合 (-1=张开, 1=闭合)
    - 输入 'c' 查看当前末端位姿
    - 输入 'r' 重置环境（放弃本条轨迹）
    - 输入 's' 保存当前轨迹并开始下一条
    - 输入 'q' 退出程序
"""

import argparse
import numpy as np
import cv2
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.config import DataCollectionConfig
from env_interfaces.robosuite_env import RoboSuiteDataCollector
from utils.data_writer import create_data_writer
import robosuite.utils.transform_utils as T


# ──────────────────────────────────────────────
# 辅助：把 7D 目标位姿转换为控制动作
# ──────────────────────────────────────────────

def pose_to_delta_action(env: RoboSuiteDataCollector,
                         target_pos: np.ndarray,
                         target_quat: np.ndarray,
                         gripper: float) -> np.ndarray:
    """
    将目标末端位姿转换为增量控制动作。

    Args:
        env: 环境接口
        target_pos: 目标位置 (3,)  世界坐标系
        target_quat: 目标四元数 (4,) 格式 (qw, qx, qy, qz)
        gripper: 夹爪命令 float，-1=张开，1=闭合

    Returns:
        action: 控制动作向量
    """
    current_pose = env.get_robot_eef_pose()   # (7,) = [x,y,z, qw,qx,qy,qz]
    current_pos  = current_pose[:3]
    current_quat = current_pose[3:]            # (qw,qx,qy,qz)

    # 位置增量
    pos_delta = target_pos - current_pos

    # 旋转增量（轴角近似，适用于小角度偏差）
    quat_diff = target_quat - current_quat
    rot_delta = quat_diff[1:] * 2.0            # 取 (qx,qy,qz) 分量并放大

    action = np.concatenate([pos_delta, rot_delta, [gripper]])
    return action


# ──────────────────────────────────────────────
# 辅助：渲染当前帧到 OpenCV 窗口
# ──────────────────────────────────────────────

def render_frame(env: RoboSuiteDataCollector):
    """渲染所有相机视角到 OpenCV 窗口。"""
    camera_images = env.render_multi_view()
    for cam_name, img in camera_images.items():
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow(cam_name, img_bgr)
    cv2.waitKey(1)


# ──────────────────────────────────────────────
# 辅助：打印当前末端位姿
# ──────────────────────────────────────────────

def print_current_pose(env: RoboSuiteDataCollector):
    pose = env.get_robot_eef_pose()   # [x,y,z, qw,qx,qy,qz]
    pos  = pose[:3]
    quat = pose[3:]
    rot_mat = T.quat2mat(quat)
    euler = T.mat2euler(rot_mat)   # (rx, ry, rz) 弧度
    print(f"\n  当前末端位置   : x={pos[0]:.4f}  y={pos[1]:.4f}  z={pos[2]:.4f}")
    print(f"  当前末端四元数 : qw={quat[0]:.4f}  qx={quat[1]:.4f}  qy={quat[2]:.4f}  qz={quat[3]:.4f}")
    print(f"  当前末端欧拉角 : rx={np.degrees(euler[0]):.2f}°  ry={np.degrees(euler[1]):.2f}°  rz={np.degrees(euler[2]):.2f}°")


# ──────────────────────────────────────────────
# 解析单行用户输入
# ──────────────────────────────────────────────

def parse_user_input(raw: str):
    """
    解析一行用户输入，返回 (target_pos, target_quat, gripper) 或特殊指令字符串。

    支持格式：
        <x> <y> <z>                        —— 仅指定位置，姿态保持不变，夹爪保持不变
        <x> <y> <z> <g>                    —— 指定位置和夹爪，姿态保持不变
        <x> <y> <z> <qw> <qx> <qy> <qz>   —— 完整位置+四元数，夹爪保持不变
        <x> <y> <z> <qw> <qx> <qy> <qz> <g> —— 完整 8 个数值
        c / r / s / q                      —— 特殊指令
    """
    raw = raw.strip()
    if raw.lower() in ('c', 'r', 's', 'q'):
        return raw.lower()

    parts = raw.split()
    try:
        vals = [float(p) for p in parts]
    except ValueError:
        return None   # 无法解析

    if len(vals) == 3:
        # 仅位置
        return np.array(vals), None, None
    elif len(vals) == 4:
        # 位置 + 夹爪
        return np.array(vals[:3]), None, float(vals[3])
    elif len(vals) == 7:
        # 位置 + 四元数
        return np.array(vals[:3]), np.array(vals[3:7]), None
    elif len(vals) == 8:
        # 位置 + 四元数 + 夹爪
        return np.array(vals[:3]), np.array(vals[3:7]), float(vals[7])
    else:
        return None


# ──────────────────────────────────────────────
# 主采集循环
# ──────────────────────────────────────────────

def collect_manual(args):
    # ---- 加载配置 ----
    print(f"加载配置文件: {args.config}")
    try:
        config = DataCollectionConfig.from_json(args.config)
    except Exception as e:
        print(f"✗ 加载配置失败: {e}")
        return

    # 命令行参数覆盖
    if args.render:
        config.has_renderer = True
        config.has_offscreen_renderer = True

    if args.output:
        config.output_dir = args.output

    config.num_episodes = args.num_episodes

    # ---- 打印信息 ----
    print("=" * 60)
    print("手动末端位姿控制数据采集")
    print("=" * 60)
    print(f"  环境       : {config.env_name}")
    print(f"  机器人     : {config.robots}")
    print(f"  目标条数   : {config.num_episodes}")
    print(f"  最大步数   : {config.horizon}")
    print(f"  输出目录   : {config.output_dir}")
    print(f"  渲染       : {config.has_renderer}")
    print("=" * 60)
    print("\n操作说明:")
    print("  输入位置和姿态来控制夹爪末端 (格式见文件顶部注释)")
    print("  c  —— 显示当前末端位姿")
    print("  r  —— 重置环境（放弃本条轨迹）")
    print("  s  —— 保存当前轨迹并开始下一条")
    print("  q  —— 退出程序\n")

    # ---- 初始化环境 ----
    print("初始化仿真环境...")
    try:
        env = RoboSuiteDataCollector(config)
        print(f"✓ 环境加载成功  |  动作维度: {env.action_dim}")
    except Exception as e:
        print(f"✗ 环境初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # ---- 初始化数据写入器 ----
    writer = create_data_writer(
        output_dir=config.output_dir,
        env_name=config.env_name,
        format=config.save_format,
    )
    print(f"✓ 数据写入器初始化完成，保存路径: {config.output_dir}\n")

    episode_idx   = 0
    saved_count   = 0

    while saved_count < config.num_episodes:
        print(f"\n{'─'*60}")
        print(f"  第 {saved_count + 1} / {config.num_episodes} 条轨迹  （已保存 {saved_count} 条）")
        print(f"{'─'*60}")

        # 重置环境
        obs = env.reset()

        # 渲染初始帧
        if config.has_renderer:
            render_frame(env)

        # 打印初始位姿
        print_current_pose(env)

        # 本条轨迹缓存
        episode_data = {
            "observations": [obs],
            "actions"     : [],
            "rewards"     : [],
            "dones"       : [],
            "success"     : False,
        }

        # 记录上一次夹爪状态（方便省略时继承）
        last_gripper = -1.0   # 默认张开
        # 记录上一次四元数（方便省略时继承）
        last_quat    = env.get_robot_eef_pose()[3:]   # (qw,qx,qy,qz)

        timestep = 0
        abort    = False

        while timestep < config.horizon:
            try:
                raw = input(f"\n[步骤 {timestep:04d}] 输入目标位姿 / 指令 > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n检测到 Ctrl+C / EOF，退出程序。")
                env.close()
                cv2.destroyAllWindows()
                return

            if not raw:
                continue

            result = parse_user_input(raw)

            # ── 特殊指令 ──
            if result == 'c':
                print_current_pose(env)
                continue

            elif result == 'r':
                print("  ↩  重置环境，放弃本条轨迹。")
                abort = True
                break

            elif result == 's':
                if len(episode_data["actions"]) == 0:
                    print("  ⚠  当前轨迹为空，请至少执行一步再保存。")
                    continue
                print(f"  💾  保存本条轨迹 (共 {len(episode_data['actions'])} 步)。")
                writer.write_episode(episode_data, episode_idx)
                episode_idx  += 1
                saved_count  += 1
                abort = False
                break

            elif result == 'q':
                print("  👋  用户退出。")
                # 如果有未保存的轨迹，询问是否保存
                if len(episode_data["actions"]) > 0:
                    ans = input("  当前轨迹未保存，是否保存？(y/n) > ").strip().lower()
                    if ans == 'y':
                        writer.write_episode(episode_data, episode_idx)
                        saved_count += 1
                        print(f"  💾  已保存，共保存 {saved_count} 条轨迹。")
                env.close()
                cv2.destroyAllWindows()
                writer.finalize()
                print(f"\n数据采集结束，共保存 {saved_count} 条轨迹，路径: {config.output_dir}")
                return

            elif result is None:
                print("  ✗  输入格式错误，请重新输入。")
                print("     格式示例: 0.45 0.1 0.85 1.0 0.0 0.0 0.0 -1")
                continue

            # ── 正常位姿输入 ──
            target_pos, target_quat, gripper_val = result

            # 继承上次四元数 / 夹爪
            if target_quat is None:
                target_quat = last_quat.copy()
            else:
                # 归一化四元数
                norm = np.linalg.norm(target_quat)
                if norm > 1e-6:
                    target_quat = target_quat / norm
                last_quat = target_quat.copy()

            if gripper_val is None:
                gripper_val = last_gripper
            else:
                last_gripper = gripper_val

            # 构建动作
            action = pose_to_delta_action(env, target_pos, target_quat, gripper_val)

            # 打印目标 vs 当前
            cur_pose = env.get_robot_eef_pose()
            print(f"  目标位置: ({target_pos[0]:.4f}, {target_pos[1]:.4f}, {target_pos[2]:.4f})  "
                  f"当前位置: ({cur_pose[0]:.4f}, {cur_pose[1]:.4f}, {cur_pose[2]:.4f})")
            print(f"  夹爪命令: {gripper_val:+.1f}")

            # 执行动作
            next_obs, reward, done, info = env.step(action)

            # 渲染
            if config.has_renderer:
                render_frame(env)
                time.sleep(0.02)

            # 存储
            episode_data["observations"].append(next_obs)
            episode_data["actions"].append(action)
            episode_data["rewards"].append(reward)
            episode_data["dones"].append(done)

            # 打印执行后的实际位置
            actual_pose = env.get_robot_eef_pose()
            print(f"  执行后位置: ({actual_pose[0]:.4f}, {actual_pose[1]:.4f}, {actual_pose[2]:.4f})")

            if "success" in info and info["success"]:
                episode_data["success"] = True
                print("  🎉  任务成功！")

            obs = next_obs
            timestep += 1

            if done:
                print("  ⚡  环境返回 done=True，本条轨迹结束。")
                # 自动提示保存
                ans = input("  是否保存本条轨迹？(y/n) > ").strip().lower()
                if ans == 'y':
                    writer.write_episode(episode_data, episode_idx)
                    episode_idx += 1
                    saved_count += 1
                    print(f"  💾  已保存，共保存 {saved_count} 条轨迹。")
                break

        if timestep >= config.horizon and not abort:
            print(f"\n  ⚠  已达最大步数 {config.horizon}。")
            ans = input("  是否保存本条轨迹？(y/n) > ").strip().lower()
            if ans == 'y':
                writer.write_episode(episode_data, episode_idx)
                episode_idx += 1
                saved_count += 1
                print(f"  💾  已保存，共保存 {saved_count} 条轨迹。")

    # ── 采集完成 ──
    env.close()
    cv2.destroyAllWindows()
    writer.finalize()

    print("\n" + "=" * 60)
    print("数据采集完成！")
    print("=" * 60)
    print(f"  共保存轨迹 : {saved_count} 条")
    print(f"  数据路径   : {config.output_dir}")
    print("=" * 60)


# ──────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="交互式手动末端位姿控制数据采集脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
输入格式示例:
  0.45 0.1 0.85                              仅改变位置，姿态和夹爪继承上次
  0.45 0.1 0.85 -1                           改变位置，夹爪张开，姿态继承上次
  0.45 0.1 0.85 1.0 0.0 0.0 0.0             改变位置和姿态（四元数），夹爪继承
  0.45 0.1 0.85 1.0 0.0 0.0 0.0 1           改变位置、姿态，夹爪闭合
""",
    )
    parser.add_argument("--config", type=str, required=True,
                        help="JSON 配置文件路径（与 gen.py 使用相同格式）")
    parser.add_argument("--render", action="store_true",
                        help="启用实时渲染（覆盖配置文件中的设置）")
    parser.add_argument("--output", type=str, default=None,
                        help="数据保存目录（覆盖配置文件中的 output_dir）")
    parser.add_argument("--num-episodes", type=int, default=10,
                        help="需要采集的轨迹条数（默认 10）")

    args = parser.parse_args()
    collect_manual(args)
