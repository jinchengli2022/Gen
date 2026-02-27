# Gen — 仿真数据自动生成框架

> **定位**：从一条人类源示范 (source demo) 出发，在 robosuite 仿真中批量生成多样化的操作轨迹数据，输出 RLDS/TFDS 格式，可直接被上层 VLA-Adapter 训练管线读取。

---

## 1. 整体流程

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Gen/ 数据生成管线                                 │
│                                                                         │
│  source demo (HDF5)                                                     │
│       │                                                                 │
│       ▼                                                                 │
│  SourceDemoLoader  ──►  SourceDemoPreprocessor  ──►  processed demo     │
│                           (限距插值预处理)                                │
│       │                                                                 │
│       ▼                                                                 │
│  ┌──────── gen.py 主循环（N episodes）────────┐                          │
│  │  env.reset()   → 随机化物体位姿              │                        │
│  │  TrajectoryGenerator.transform_demo_to_new_scene()                   │
│  │    ├─ Approach 段: 速度匹配 + 线性/Slerp 插值  │                      │
│  │    ├─ Grasp 段:   刚体变换                     │                      │
│  │    └─ Move 段:    XY缩放 + Z保高 + Slerp旋转   │                      │
│  │  WaypointPolicy  → collect_episode()          │                      │
│  │  _check_success() → 只保存成功 episode         │                      │
│  └───────────────────────────────────────────────┘                       │
│       │                                                                 │
│       ▼                                                                 │
│  RLDSDataWriter  ──►  TFRecord (RLDS 格式)                              │
│                       + 检查用 HDF5 / 视频                               │
└─────────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   VLA-Adapter 训练管线                                    │
│                                                                         │
│  finetune.py  ←  RLDSDataset  ←  tfds.builder_from_directory()         │
│    │                                                                    │
│    ├─ libero_dataset_transform (标准化)                                  │
│    ├─ Q99 归一化                                                        │
│    ├─ Action Chunking + Image Augmentation                              │
│    └─ LoRA 微调 Qwen2.5-0.5B VLM                                       │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 目录结构

```
Gen/
├── configs/
│   ├── config.py                          # DataCollectionConfig 数据类
│   └── examples/
│       └── pouring_water_trajgen.json     # PouringWater 生成配置
├── env/
│   ├── pouring_water_env.py               # PouringWater 自定义环境（ManipulationEnv 子类）
│   └── assets/                            # MuJoCo XML 模型（杯子等）
├── env_interfaces/
│   └── robosuite_env.py                   # RoboSuiteDataCollector 环境包装器
├── scripts/
│   ├── gen.py                             # ★ 主入口：轨迹生成 + 数据收集
│   ├── generate_demo.py                   # 源示范录制脚本
│   ├── manual_collect.py                  # 手动遥操作收集
│   └── simple_collect.py                  # 简单随机策略收集
├── utils/
│   ├── source_loader.py                   # SourceDemoLoader — 读取 HDF5 源示范
│   ├── source_preprocessor.py             # SourceDemoPreprocessor — 限距插值预处理
│   ├── trajectory_generator.py            # TrajectoryGenerator — 三段式轨迹变换
│   ├── trajectory_visualizer.py           # 轨迹可视化工具
│   └── data_writer.py                     # HDF5 / Pickle / RLDS 数据写入器
├── tfds_builders/
│   └── pouringwater_generated/            # TFDS DatasetBuilder（定义数据 schema）
├── source/                                # 源示范文件（gen_demo.hdf5 等）
├── data/                                  # 生成数据输出目录
├── docs/                                  # 设计文档
└── tests/                                 # 测试脚本
```

---

## 3. 核心模块说明

### 3.1 数据生成入口 — `scripts/gen.py`

主脚本，完整执行一次数据生成流程：

1. 加载 JSON 配置 → `DataCollectionConfig`
2. 初始化 robosuite 环境 → `RoboSuiteDataCollector`
3. 加载并预处理源示范 → `SourceDemoLoader` + `SourceDemoPreprocessor`
4. 循环 N 个 episode：
   - `env.reset()` 随机化场景（物体位置 / 朝向）
   - `TrajectoryGenerator` 将源轨迹变换到新场景
   - `WaypointPolicy` 逐步执行，环境仿真
   - 成功判定 → `RLDSDataWriter` 保存
5. 输出统计日志

### 3.2 轨迹变换 — `utils/trajectory_generator.py`

核心算法，将一条源示范轨迹适配到任意新场景。分三段独立变换：

| 段 | 策略 | 说明 |
|---|------|------|
| **Approach** | 速度匹配 + 位姿插值 | 从当前 EEF 位置到新物体上方抓取点，保持源 demo 的移动速度，自动调整步数 |
| **Grasp** | 刚体变换 | 根据新旧物体位姿的变换矩阵 `T_new @ T_old⁻¹`，将源 demo 的抓取段做整体刚体变换 |
| **Move** | XY 缩放 + Z 保高 + Slerp | XY 平面按新旧物体距离缩放并旋转，Z 轴保持源 demo 的运动弧线高度，旋转通过 Slerp 插值 |

### 3.3 限距预处理 — `utils/source_preprocessor.py`

检测源轨迹中相邻帧的位移/旋转增量是否超过控制器物理上限（`output_max`），若超出则在超距段内插入中间帧。保证变换后的轨迹在每一步都不触发控制器 clipping。

### 3.4 环境包装 — `env_interfaces/robosuite_env.py`

`RoboSuiteDataCollector` 封装了：
- 控制器配置构建（支持 `OSC_POSE` / `OSC_POSITION` / `IK_POSE`）
- 统一的 `get_robot_eef_pose()` / `get_object_pose()` 接口
- 多相机渲染

### 3.5 自定义环境 — `env/pouring_water_env.py`

`PouringWater(ManipulationEnv)` —— 倒水任务：

- **物体**：yellow_cup（操作对象）+ black_cup（目标容器）
- **成功条件**：lifted + XY aligned + tilted + near target
- **Placement Sampler**：随机化两杯位置和 Z 轴朝向

### 3.6 数据写入 — `utils/data_writer.py`

`RLDSDataWriter` 将 episode 数据转为 RLDS 格式：

```
observation/image        : (256, 256, 3) uint8   — agentview 相机
observation/wrist_image  : (256, 256, 3) uint8   — 腕部相机
observation/state        : (8,) float32           — eef_pos(3) + eef_axisangle(3) + gripper_qpos(2)
action                   : (7,) float32           — delta_pos(3) + delta_axisangle(3) + gripper(1)
language_instruction     : string                 — 任务描述
```

数据格式与 LIBERO 完全一致，可直接复用 `libero_dataset_transform`。

---

## 4. 快速使用

### 4.1 数据生成

```bash
cd /home/ljc/Git/Gen_VLA_Adapter/Gen

# 基本生成（使用 JSON 配置）
CUDA_VISIBLE_DEVICES=6 python scripts/gen.py \
    --config configs/examples/pouring_water_trajgen.json

# 带实时渲染（调试用）
CUDA_VISIBLE_DEVICES=6 python scripts/gen.py \
    --config configs/examples/pouring_water_trajgen.json --render

# 带 debug 模式（保存轨迹可视化 + CSV）
CUDA_VISIBLE_DEVICES=6 python scripts/gen.py \
    --config configs/examples/pouring_water_trajgen.json --debug
```

生成结果保存在 `data/pouring_water_generated/PouringWater/` 目录下，包括：
- `1.0.0/` — RLDS TFRecord 数据（可直接用于训练）
- `*.log` — 生成日志（每个 episode 的详细信息）
- `videos/` — 每个成功 episode 的渲染视频

### 4.2 VLA-Adapter 训练

```bash
cd /home/ljc/Git/Gen_VLA_Adapter

data_name=pouringwater_generated
current_time=$(date +%Y%m%d_%H%M%S)

CUDA_VISIBLE_DEVICES=6 torchrun --standalone --nnodes 1 --nproc-per-node 1 \
    vla-scripts/finetune.py \
    --vlm_path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
    --config_file_path pretrained_models/configs \
    --data_root_dir data/gen/pouring_water_generated/PouringWater \
    --dataset_name $data_name \
    --run_root_dir outputs \
    --use_film False \
    --num_images_in_input 2 \
    --use_proprio True \
    --use_lora True \
    --use_fz False \
    --use_minivlm True \
    --image_aug True \
    --num_steps_before_decay 400000 \
    --max_steps 400005 \
    --save_freq 5000 \
    --save_latest_checkpoint_only False \
    --merge_lora_during_training True \
    --batch_size 4 \
    --grad_accumulation_steps 8 \
    --learning_rate 2e-4 \
    --lora_rank 64 \
    --use_pro_version True \
    --wandb_entity "jincheng-li2022-xidian-university" \
    --wandb_project "$data_name" \
    --run_id_note "VLA-Adapter--${data_name}--${current_time}" \
    > logs/VLA-Adapter--${data_name}--${current_time}.log 2>&1 &
```

---

## 5. 配置说明

JSON 配置文件 (`configs/examples/pouring_water_trajgen.json`) 主要字段：

| 字段 | 值 | 说明 |
|------|-----|------|
| `env_name` | `"PouringWater"` | 自定义环境名 |
| `robots` | `"UR5e"` | 机器人型号 |
| `controller_type` | `"OSC_POSE"` | 控制器类型 |
| `num_episodes` | `500` | 尝试生成的 episode 数（仅成功的会保存） |
| `save_format` | `"rlds"` | 输出格式（可选 `hdf5` / `pickle` / `rlds`） |
| `source_demo_path` | `"source/gen_demo.hdf5"` | 源示范文件路径 |
| `limit_dpos` / `limit_drot` | `1.0` | 归一化限距阈值 (0~1) |
| `language_instruction` | `"pour the water..."` | 任务语言指令（写入 RLDS） |
| `horizon` | `1000` | 单 episode 最大步数 |
| `camera_names` | `["agentview", "robot0_eye_in_hand"]` | 采集的相机视角 |

---

## 6. VLA-Adapter 侧适配

为让生成数据被训练管线识别，需在以下文件中注册（已完成）：

| 文件 | 修改 |
|------|------|
| `prismatic/vla/datasets/rlds/dataset.py` | `tfds.builder()` fallback 到 `tfds.builder_from_directory()` |
| `prismatic/vla/datasets/rlds/oxe/configs.py` | 注册 `pouringwater_generated` 数据集配置 |
| `prismatic/vla/datasets/rlds/oxe/transforms.py` | 注册使用 `libero_dataset_transform` |
| `prismatic/vla/datasets/rlds/oxe/mixtures.py` | 注册 mixture |

详细说明见 [docs/RLDS_DATA_PIPELINE.md](docs/RLDS_DATA_PIPELINE.md) 和 [docs/VLA_ADAPTER数据格式.md](docs/VLA_ADAPTER数据格式.md)。

---

## 7. 设计文档索引

| 文档 | 内容 |
|------|------|
| [docs/trajectory_generation.md](docs/trajectory_generation.md) | 轨迹生成算法详解 |
| [docs/RLDS_DATA_PIPELINE.md](docs/RLDS_DATA_PIPELINE.md) | RLDS 数据写入管道全流程 |
| [docs/VLA_ADAPTER数据格式.md](docs/VLA_ADAPTER数据格式.md) | 训练数据格式与数据流 |
| [docs/POURING_README.md](docs/POURING_README.md) | PouringWater 环境说明 |
| [docs/REFACTOR_SPEED_LIMIT.md](docs/REFACTOR_SPEED_LIMIT.md) | 限距体系重构记录 |
| [docs/关节限位解决方案.md](docs/关节限位解决方案.md) | UR5e 关节限位问题分析与方案 |
