# PouringWater 环境使用指南

## 环境说明

**PouringWater** 是一个自定义的 robosuite 环境，模拟机器人倒水任务。

### 任务目标
从黄色杯子倒水到黑色杯子中。

### 成功标准
1. ✓ 黄色杯子被抬起（高度 > 10cm）
2. ✓ 两个杯子在 XY 平面对齐（偏航角差 < 10°）
3. ✓ 黄色杯子朝黑色杯子倾斜（倾斜角 > 10°）
4. ✓ 黄色杯子靠近黑色杯子（XY 距离 < 20cm）

### 环境组成
- **黄色杯子**（可移动）：半径 3.5cm，高 12cm，黄色
- **黑色杯子**（目标容器）：半径 4cm，高 10cm，深灰色
- **机器人**：默认 Panda，可选 Sawyer、IIWA 等
- **桌面**：80cm × 80cm

## 文件结构

```
gen/
├── pouring_water_env.py    # PouringWater 环境定义（纯 robosuite，无 mimicgen 依赖）
├── robosuite_env.py        # 环境包装器（已集成自定义环境支持）
├── collect_pouring.py      # PouringWater 专用数据收集脚本
├── simple_collect.py       # 通用数据收集脚本（也支持 PouringWater）
└── test_pouring_env.sh     # 测试脚本
```

## 快速开始

### 方法 1：使用专用脚本

```bash
# 收集 10 个回合，带可视化
python collect_pouring.py --num_episodes 10 --render

# 收集 50 个回合，包含相机观测
python collect_pouring.py --num_episodes 50 --use_camera --image_size 256

# 收集 20 个回合，使用不同机器人
python collect_pouring.py --num_episodes 20 --robot Sawyer --render
```

### 方法 2：使用通用脚本

```bash
# 基本收集
python simple_collect.py --env_name PouringWater --num_episodes 10

# 带可视化
python simple_collect.py --env_name PouringWater --num_episodes 5 --render

# 带相机观测
python simple_collect.py --env_name PouringWater --num_episodes 20 --use_camera
```

## 命令行参数

### 环境参数
- `--robot`: 机器人模型（默认：Panda）
- `--horizon`: 最大步数（默认：500）
- `--control_freq`: 控制频率（默认：20Hz）

### 数据收集参数
- `--num_episodes`: 收集的回合数（默认：10）
- `--output_dir`: 输出目录（默认：data/pouring_water）
- `--format`: 数据格式（hdf5 或 pickle，默认：hdf5）

### 观测参数
- `--use_camera`: 启用相机观测
- `--camera_names`: 相机名称，逗号分隔（默认：agentview,robot0_eye_in_hand）
- `--image_size`: 图像尺寸（默认：256）

### 可视化参数
- `--render`: 实时渲染显示
- `--verbose`: 显示详细信息（仅 collect_pouring.py）

## 环境特性

### 1. 奖励塑形
环境支持奖励塑形（reward shaping）以改善学习：
- 抬升奖励：杯子抬起越高，奖励越大
- 接近奖励：杯子之间距离越近，奖励越大
- 成功奖励：完成任务获得 1.0

### 2. 失败诊断
环境自动记录失败原因：
```python
episode_data["failure_reasons"] = [
    "Not Lifted (h=0.82 < 0.90)",
    "XY Misaligned (diff=45.2° > 10°)",
    "Not Tilted (angle=2.3° <= 10°)",
    "Not Near (dist=0.35m > 0.2m)"
]
```

### 3. 随机初始化
- 黄色杯子：随机位置在 [0, 0.1] × [0, 0.1] 区域
- 黑色杯子：随机位置在 [-0.1, 0] × [-0.3, -0.2] 区域
- 黄色杯子：随机旋转
- 黑色杯子：固定朝向

## 数据格式

### HDF5 格式
```
PouringWater_data.hdf5
├── episode_0/
│   ├── actions              # (T, 7) - 机器人动作
│   ├── rewards              # (T,) - 奖励
│   ├── dones                # (T,) - 结束标志
│   └── observations/
│       ├── state            # (T, state_dim) - 状态
│       ├── agentview_image  # (T, 256, 256, 3) - 相机图像
│       └── ...
├── episode_1/
│   └── ...
└── [metadata]
```

## 与原始 pouring.py 的区别

| 特性 | 原始 pouring.py | 新 pouring_water_env.py |
|------|----------------|------------------------|
| 依赖 | mimicgen | 纯 robosuite |
| 基类 | SingleArmEnv_MG | SingleArmEnv |
| 对象 | 外部 XML 模型 | CylinderObject |
| 资源文件 | 需要 yellow_cup.xml | 无需外部文件 |
| 集成 | 需要 mimicgen | 直接可用 |

## 测试环境

```bash
# 运行测试脚本
bash test_pouring_env.sh

# 或手动测试
python -c "from pouring_water_env import PouringWater; print('✓ 导入成功')"
python -c "from robosuite_env import CUSTOM_ENVS; print(f'自定义环境: {list(CUSTOM_ENVS.keys())}')"
```

## 示例输出

```
============================================================
PouringWater Data Collection
============================================================
Environment: PouringWater (Custom)
Robot: Panda
Episodes: 10
Output: data/pouring_water
Format: hdf5
============================================================

Initializing PouringWater environment...
✓ Environment loaded successfully
  Action dimension: 7
  State keys: ['robot0_joint_pos', 'robot0_joint_vel', ...]
  Image keys: []

Collecting 10 episodes...
100%|████████████████████| 10/10 [00:45<00:00, 4.5s/it]

============================================================
Data Collection Complete!
============================================================
Total episodes: 10
Success rate: 0/10 (0.0%)
Average reward: 0.123 ± 0.045
Data saved to: data/pouring_water

Failure Analysis:
------------------------------------------------------------
  Not Lifted (h=0.82 < 0.90): 8 (80.0%)
  XY Misaligned (diff=45.2° > 10°): 9 (90.0%)
  Not Tilted (angle=2.3° <= 10°): 10 (100.0%)
  Not Near (dist=0.35m > 0.2m): 7 (70.0%)
============================================================
```

## 故障排除

### 1. 导入错误
```bash
# 确保安装了所有依赖
pip install robosuite numpy h5py
```

### 2. 渲染问题
```bash
# 设置环境变量
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
```

### 3. MuJoCo 相关
确保 MuJoCo 2.1+ 已正确安装。

## 扩展建议

### 添加更多成功条件
在 `_check_success()` 方法中修改：
```python
# 添加倾倒时间检查
pouring_duration = timestep > 100  # 需要持续倾倒
success = lifted and xy_aligned and is_plane_tilted and is_near and pouring_duration
```

### 修改对象属性
在 `_load_model()` 中调整：
```python
self.yellow_cup = CylinderObject(
    size_min=[0.04, 0.08],  # 更大的杯子
    rgba=[0.8, 0.2, 0.2, 1.0],  # 红色
)
```

### 添加更多对象
```python
self.obstacles = [...]  # 添加障碍物
self.objects = [self.yellow_cup, self.black_cup] + self.obstacles
```

## 参考资料

- [robosuite 官方文档](https://robosuite.ai/docs/overview.html)
- [robosuite API](https://robosuite.ai/docs/simulation/environment.html)
- [自定义环境教程](https://robosuite.ai/docs/tutorials/add_environment.html)
