# Robosuite Data Collection

基于 robosuite 1.5 的数据收集脚本，用于生成机器人操作演示数据。

## 特性

- ✅ 支持 robosuite 1.5 的所有环境
- ✅ 支持多种机器人模型（Panda, Sawyer, IIWA等）
- ✅ 支持相机观测和状态观测
- ✅ HDF5 和 Pickle 数据格式
- ✅ 可扩展的策略接口（随机、脚本化、人类演示等）
- ✅ 实时可视化支持

## 安装

### 1. 安装 MuJoCo

参考官方文档：https://mujoco.org/

### 2. 安装依赖

```bash
cd /home/ljc/Git/Gen_VLA_Adapter/gen
pip install -r requirements.txt
```

### 3. 安装 robosuite

```bash
pip install robosuite
# 或从源码安装最新版本
git clone https://github.com/ARISE-Initiative/robosuite.git
cd robosuite
pip install -e .
```

## 快速开始

### 基本使用

收集 10 个随机策略演示：

```bash
python simple_collect.py --env_name PickPlaceCan --num_episodes 10
```

### 带可视化

```bash
python simple_collect.py --env_name Stack --num_episodes 5 --render
```

### 使用相机观测

```bash
python simple_collect.py \
    --env_name Door \
    --num_episodes 20 \
    --use_camera \
    --camera_names agentview,robot0_eye_in_hand \
    --image_size 224
```

### 更多环境示例

```bash
# 堆叠任务
python simple_collect.py --env_name Stack --robot Sawyer --num_episodes 50

# 开门任务
python simple_collect.py --env_name Door --robot Panda --num_episodes 30

# 擦拭任务
python simple_collect.py --env_name Wipe --robot IIWA --num_episodes 40

# 工具悬挂任务
python simple_collect.py --env_name ToolHang --num_episodes 25
```

## 支持的环境

robosuite 1.5 支持的主要环境：

- **PickPlaceCan**: 抓取和放置罐子
- **Stack**: 堆叠方块
- **Door**: 开门
- **Wipe**: 擦拭桌面
- **ToolHang**: 工具悬挂
- **NutAssembly**: 螺母装配
- **TwoArmPegInHole**: 双臂钉子插孔
- 更多环境请参考：https://robosuite.ai/docs/modules/environments.html

## 支持的机器人

- **Panda** (Franka Emika)
- **Sawyer** (Rethink Robotics)
- **IIWA** (KUKA)
- **UR5e** (Universal Robots)
- **Kinova3** (Kinova Gen3)
- **Jaco** (Kinova Jaco)

## 项目结构

```
gen/
├── config.py              # 配置类定义
├── robosuite_env.py       # 环境包装器
├── data_writer.py         # 数据写入工具
├── simple_collect.py      # 简单数据收集脚本
├── requirements.txt       # 依赖包
└── README.md             # 本文档
```

## 配置说明

主要配置参数在 `config.py` 中的 `DataCollectionConfig` 类：

```python
config = DataCollectionConfig(
    env_name="PickPlaceCan",      # 环境名称
    robots="Panda",                # 机器人模型
    num_episodes=100,              # 收集的回合数
    horizon=500,                   # 最大步数
    control_freq=20,               # 控制频率 (Hz)
    use_camera_obs=True,           # 使用相机观测
    camera_names=["agentview"],    # 相机名称列表
    camera_heights=224,            # 图像高度
    camera_widths=224,             # 图像宽度
    output_dir="data/output",      # 输出目录
    save_format="hdf5",            # 保存格式
)
```

## 数据格式

### HDF5 格式

```
data.hdf5
├── episode_0/
│   ├── actions          # (T, action_dim)
│   ├── rewards          # (T,)
│   ├── dones            # (T,)
│   └── observations/
│       ├── state        # (T, state_dim)
│       └── agentview_image  # (T, H, W, 3)
├── episode_1/
│   └── ...
└── [metadata]
```

### Pickle 格式

每个回合保存为单独的 `.pkl` 文件：

```
data/
├── PickPlaceCan_ep0000.pkl
├── PickPlaceCan_ep0001.pkl
└── metadata.json
```

## 扩展自定义策略

可以通过继承 `RandomPolicy` 类来实现自定义策略：

```python
class MyCustomPolicy:
    def __init__(self, action_dim: int):
        self.action_dim = action_dim
        # 初始化策略
        
    def get_action(self, observation: dict) -> np.ndarray:
        # 根据观测生成动作
        action = ...  # 你的策略逻辑
        return action
    
    def reset(self):
        # 重置策略状态
        pass
```

## 命令行参数

```bash
python simple_collect.py --help

参数：
  --env_name         环境名称 (默认: PickPlaceCan)
  --robot            机器人模型 (默认: Panda)
  --num_episodes     收集回合数 (默认: 10)
  --horizon          最大步数 (默认: 500)
  --control_freq     控制频率 (默认: 20)
  --output_dir       输出目录 (默认: data/robosuite_random)
  --format           数据格式 (hdf5 或 pickle)
  --use_camera       启用相机观测
  --camera_names     相机名称 (逗号分隔)
  --image_size       图像大小 (默认: 224)
  --render           可视化渲染
```

## 常见问题

### 1. MuJoCo 许可证问题

MuJoCo 2.1+ 是免费的，无需许可证。确保安装了正确版本。

### 2. 渲染问题

如果遇到渲染问题，设置环境变量：
```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
```

### 3. 内存不足

收集大量数据时，考虑：
- 减小图像分辨率
- 使用 pickle 格式而不是 HDF5
- 分批收集数据

## 参考资料

- [robosuite 官方文档](https://robosuite.ai/docs/overview.html)
- [robosuite GitHub](https://github.com/ARISE-Initiative/robosuite)
- [MuJoCo 文档](https://mujoco.readthedocs.io/)

## 许可证

本项目遵循与 robosuite 相同的许可证。
