# VLA-Adapter 训练数据格式详解

本文档详细说明 VLA-Adapter 项目在执行 `finetune.py` 训练时所读取的数据格式、结构与完整的数据流转过程。

---

## 目录

1. [数据总览](#1-数据总览)
2. [磁盘上的数据格式（RLDS/TFDS）](#2-磁盘上的数据格式rldstfds)
3. [数据集配置（configs.py）](#3-数据集配置configspy)
4. [标准化变换（transforms.py）](#4-标准化变换transformspy)
5. [数据归一化](#5-数据归一化)
6. [轨迹变换与分帧](#6-轨迹变换与分帧)
7. [Batch 变换（RLDSBatchTransform）](#7-batch-变换rldsbatchtransform)
8. [最终送入模型的 Batch 格式](#8-最终送入模型的-batch-格式)
9. [关键常量](#9-关键常量)
10. [数据统计文件（dataset_statistics.json）](#10-数据统计文件dataset_statisticsjson)
11. [如何制作自定义数据集](#11-如何制作自定义数据集)

---

## 1. 数据总览

VLA-Adapter 使用 **RLDS（Reinforcement Learning Datasets）** 格式存储训练数据，底层基于 **TensorFlow Datasets (TFDS)** 框架。数据存储在 TFRecord 文件中。

**完整数据流水线**：

```
磁盘 TFRecord 文件
    ↓  tfds.builder() + dl.DLataset.from_rlds()
原始轨迹 (trajectory dict)
    ↓  standardize_fn（如 libero_dataset_transform）
标准化轨迹 {observation, action, language_instruction}
    ↓  restructure() — 提取 image/proprio/task
规范轨迹 {observation: {image_primary, proprio, ...}, action, task, dataset_name}
    ↓  normalize_action_and_proprio() — Q99 归一化
归一化轨迹
    ↓  chunk_act_obs() — 时间窗口切分 + action chunking
分帧数据 (每个 frame 含 window_size 个观测 + NUM_ACTIONS_CHUNK 个动作)
    ↓  apply_frame_transforms() — 图像解码、resize、数据增强
    ↓  RLDSBatchTransform.__call__() — 构造 prompt、tokenize、提取 pixel_values
单样本 dict
    ↓  PaddedCollatorForActionPrediction — padding、堆叠
最终 Batch (送入模型)
```

---

## 2. 磁盘上的数据格式（RLDS/TFDS）

### 2.1 目录结构

```
data/
└── libero/                              # data_root_dir
    └── libero_spatial_no_noops/         # dataset_name
        └── 1.0.0/                       # 版本号
            ├── dataset_info.json        # 数据集元信息
            ├── features.json            # 特征描述（数据 schema）
            ├── libero_spatial_no_noops-train.tfrecord-00000-of-00016
            ├── libero_spatial_no_noops-train.tfrecord-00001-of-00016
            ├── ...                      # TFRecord 分片文件
            └── dataset_statistics_XXXX.json  # 自动计算的统计缓存
```

### 2.2 RLDS 数据 Schema

每个 TFRecord 文件存储多条**轨迹 (trajectory)**，每条轨迹包含多个**步骤 (step)**。RLDS 标准格式如下：

```python
# 一条轨迹 (trajectory) 的结构
{
    "observation": {
        "image":        (T, H, W, 3),    # uint8, 主相机 RGB 图像
        "wrist_image":  (T, H, W, 3),    # uint8, 腕部相机 RGB 图像（可选）
        "state":        (T, D_state),    # float32, 机器人本体感知状态
    },
    "action":           (T, 7),          # float32, 机器人动作
    "language_instruction": string,      # 自然语言任务指令
    # ... 其他可选字段
}
```

### 2.3 LIBERO 数据集具体字段

对于 LIBERO 数据集（如 `libero_spatial_no_noops`）：

| 字段 | 形状 | 类型 | 说明 |
|------|------|------|------|
| `observation/image` | `(T, 256, 256, 3)` | uint8 | 第三人称相机 RGB |
| `observation/wrist_image` | `(T, 256, 256, 3)` | uint8 | 腕部相机 RGB |
| `observation/state` | `(T, 8)` | float32 | 本体感知状态（见下文） |
| `action` | `(T, 7)` | float32 | 末端执行器增量动作（见下文） |
| `language_instruction` | scalar | string | 任务语言指令 |

---

## 3. 数据集配置（configs.py）

每个数据集在 `prismatic/vla/datasets/rlds/oxe/configs.py` 中注册配置，定义了如何从原始轨迹中提取各字段。

### LIBERO 配置

```python
"libero_spatial_no_noops": {
    "image_obs_keys": {
        "primary": "image",           # 主相机 → 映射为 image_primary
        "secondary": None,            # 无次要相机
        "wrist": "wrist_image"        # 腕部相机 → 映射为 image_wrist
    },
    "depth_obs_keys": {
        "primary": None,              # 不使用深度图
        "secondary": None,
        "wrist": None
    },
    "state_obs_keys": [
        "EEF_state",                  # 末端执行器状态（来自 transform 后）
        "gripper_state"               # 夹爪状态（来自 transform 后）
    ],
    "state_encoding": StateEncoding.POS_EULER,   # 状态编码方式
    "action_encoding": ActionEncoding.EEF_POS,   # 动作编码方式
}
```

### state_encoding 枚举

| 类型 | 说明 | 维度组成 |
|------|------|----------|
| `POS_EULER` | 末端位姿（欧拉角） | XYZ(3) + RPY(3) + PAD(1) + Gripper(1) = **8D** |
| `POS_QUAT` | 末端位姿（四元数） | XYZ(3) + Quat(4) + Gripper(1) = **8D** |
| `JOINT` | 关节角 | Joints(7) + Gripper(1) = **8D** |

### action_encoding 枚举

| 类型 | 说明 | 维度组成 |
|------|------|----------|
| `EEF_POS` | 末端增量 | ΔXYZ(3) + ΔRPY(3) + Gripper(1) = **7D** |
| `JOINT_POS` | 关节增量 | ΔJoints(7) + Gripper(1) = **8D** |

---

## 4. 标准化变换（transforms.py）

原始轨迹经过 `standardize_fn` 转为统一格式。LIBERO 使用 `libero_dataset_transform`：

```python
def libero_dataset_transform(trajectory):
    # 1. 夹爪动作处理：[-1, 1] → clip到[0, 1] → 翻转 → 1=张开, 0=闭合
    gripper_action = trajectory["action"][:, -1:]
    gripper_action = invert_gripper_actions(tf.clip_by_value(gripper_action, 0, 1))

    # 2. 重组动作向量：前6维(delta_pos + delta_rot) + 处理后的gripper
    trajectory["action"] = tf.concat([
        trajectory["action"][:, :6],      # delta_xyz(3) + delta_rpy(3)
        gripper_action,                    # 0=close, 1=open
    ], axis=1)

    # 3. 提取本体感知状态
    trajectory["observation"]["EEF_state"] = trajectory["observation"]["state"][:, :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -2:]
    
    return trajectory
```

**经过 transform 后**，`restructure()` 函数会进一步规范化：

```python
# restructure() 的输出格式
{
    "observation": {
        "image_primary":   (T, H, W, 3),     # 主相机图像（编码为 bytes）
        "image_wrist":     (T, H, W, 3),     # 腕部图像（编码为 bytes）
        "proprio":         (T, 8),            # EEF_state(6) + gripper_state(2) 拼接
        "timestep":        (T,),              # 时间步索引
    },
    "task": {
        "language_instruction": string,       # 语言指令
    },
    "action":              (T, 7),            # delta_xyz(3) + delta_rpy(3) + gripper(1)
    "dataset_name":        (T,),              # 数据集名称（重复字符串）
}
```

---

## 5. 数据归一化

### 5.1 归一化方式

LIBERO 使用 **`BOUNDS_Q99`** 归一化（基于 1% 和 99% 分位数）：

```python
# 对于 action 和 proprio 分别:
normalized = clip(2 * (x - q01) / (q99 - q01 + 1e-8) - 1,  -1,  1)
```

归一化后，action 和 proprio 的值域被限制在 **[-1, 1]**。

### 5.2 归一化掩码

```python
# EEF_POS 动作编码的归一化掩码
action_normalization_mask = [True, True, True, True, True, True, False]
#                           Δx    Δy    Δz    Δrx   Δry   Δrz   gripper
# 前6维归一化，gripper维度不归一化（因为已经是0或1）
```

### 5.3 绝对/相对动作掩码

```python
# EEF_POS 动作编码的绝对动作掩码
absolute_action_mask = [False, False, False, False, False, False, True]
#                       Δx     Δy     Δz     Δrx    Δry    Δrz   gripper
# 前6维是相对动作（delta），gripper是绝对动作
# 用于在 action chunking 时正确填充超出轨迹末尾的动作
```

---

## 6. 轨迹变换与分帧

### 6.1 Action Chunking（动作分块）

通过 `chunk_act_obs()` 将轨迹切分为重叠的时间窗口：

```python
# 关键参数
window_size = 1                              # 观测窗口大小
future_action_window_size = NUM_ACTIONS_CHUNK - 1  # = 7（未来动作窗口）
```

**切分后的每个 frame**：

```python
{
    "observation": {
        "image_primary":  (window_size, H, W, 3),     # 过去+当前观测
        "proprio":        (window_size, 8),
        "pad_mask":       (window_size,),              # 填充标志
        ...
    },
    "action":             (window_size + future_action_window_size, 7),  # (8, 7) = 8步action chunk
    "task": { "language_instruction": string },
    "dataset_name": (1,),
}
```

即每个训练样本包含 **当前动作 + 7个未来动作 = 8步 action chunk**。

### 6.2 图像变换

```python
# 1. 图像 resize
resize_size = (224, 224)     # 由 VLM 配置决定

# 2. 图像增强（仅训练时）
image_augment_kwargs = {
    "random_resized_crop": {"scale": [0.9, 0.9], "ratio": [1.0, 1.0]},
    "random_brightness": [0.2],
    "random_contrast": [0.8, 1.2],
    "random_saturation": [0.8, 1.2],
    "random_hue": [0.05],
}
```

---

## 7. Batch 变换（RLDSBatchTransform）

`RLDSBatchTransform.__call__()` 将每个 RLDS frame 转换为模型可用的格式。

### 7.1 输入（来自 RLDS pipeline 的一个 frame）

```python
rlds_batch = {
    "dataset_name": "libero_spatial_no_noops",
    "observation": {
        "image_primary": (1, 224, 224, 3),    # uint8 图像
        "image_wrist":   (1, 224, 224, 3),    # uint8 腕部图像
        "proprio":       (1, 8),              # float32 本体感知
    },
    "action":            (8, 7),              # float32, 8步动作块（已归一化到[-1,1]）
    "task": {
        "language_instruction": b"pick up the red cup..."
    },
}
```

### 7.2 处理流程（use_minivlm=True 时，即当前命令使用的路径）

```python
# 1. 提取图像：主相机第一帧（注意：wrist 图像单独处理，走 pixel_values_wrist）
img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])

# 2. 提取语言指令
lang = rlds_batch["task"]["language_instruction"].decode().lower()

# 3. 提取动作
current_action = rlds_batch["action"][0]      # 当前动作 (7,)
future_actions = rlds_batch["action"][1:]     # 未来动作 shape=(7, 7)

# 4. 动作 tokenize（use_minivlm=True 时，返回 token ID 列表而非字符串）
#    每个动作维度 → 256个bin → 映射到词表末尾 256 个 token
#    current_action (7D)   → 7个 token ID
#    future_actions (7步×7D) → 7×7 = 49个 token ID（每步7个，共7步）
#    总共 = 7 + 49 = 56 个 action token（flattened_action_chunk_string）
#
#    ⚠️ 关键逻辑：固定为 NUM_TOKENS=64 个 token
#      - 若 56 < 64：随机从已有 56 个中重采样 8 个补充到 64
#      - 若 56 > 64（不可能，因 8步×7D=56 < 64）：截断取前 64

# 5. 构造 prompt（GPT 回答部分留空，action token 直接追加到 input_ids 末尾）
#    "[问] What action should the robot take to {lang}?\n[答] "
#    然后从 prompt 的 input_ids 中删去末尾 3 个 token（特殊符号），
#    再拼接上 64 个 action token ID

# 6. 设置 labels：与 input_ids 完全一致，
#    再将末尾 (NUM_TOKENS+1) 之前的所有位置设为 -100（不计 loss）
#    即只有最后 65 个位置（64个action token + 1个stop token）参与 loss 计算
```

### 7.3 输出（单个样本 dict）

```python
{
    "pixel_values":        Tensor(C, 224, 224),   # 主相机经过 image_transform 的图像，C=3
    "pixel_values_wrist":  Tensor(C, 224, 224),   # 腕部相机图像（use_wrist_image=True 时）
                                                   # ⚠️ 此时两张图是分开存放的！
                                                   # Collator 中会 torch.cat 到 pixel_values 上
    "input_ids":      Tensor(seq_len,),      # prompt token + 64个action token
    "labels":         Tensor(seq_len,),      # 仅末尾 65 位有效（64+stop），其余为 -100
    "dataset_name":   str,                   # 数据集名称
    "actions":        ndarray(8, 7),         # 原始 action chunk（未经 tokenize，用于 L1 loss）
    "proprio":        ndarray(1, 8),         # 本体感知状态（use_proprio=True 时才有）
}
```

---

## 8. 最终送入模型的 Batch 格式

`PaddedCollatorForActionPrediction` 将多个样本组成一个 batch：

```python
batch = {
    # 视觉输入（两张图 cat 后）
    "pixel_values":    Tensor(B, 6, 224, 224),
    # ⚠️ 实现细节：
    #   - Collator 中：pixel_values = torch.cat([主相机stack, 腕部stack], dim=1)
    #   - 即在 channel 维度（dim=1）拼接，而非新增图像维度
    #   - 结果：(B, 3, 224, 224) cat (B, 3, 224, 224) → (B, 6, 224, 224)

    # 文本输入
    "input_ids":       Tensor(B, max_seq_len),  # 右侧 pad 到 batch 内最长
    "attention_mask":  Tensor(B, max_seq_len),  # pad 位置为 False
    "labels":          Tensor(B, max_seq_len),  # pad 位置为 -100

    # 动作标签（L1 regression loss 的 ground truth，已归一化到 [-1,1]）
    "actions":         Tensor(B, 8, 7),
    #   actions[:, 0, :]  = 当前步动作 (B, 7)
    #   actions[:, 1:, :] = 未来7步动作 (B, 7, 7)

    # 本体感知（use_proprio=True 时）
    "proprio":         Tensor(B, 8),
    # ⚠️ 注意：经过 np.squeeze(np.stack(proprio))，shape 从 (B, 1, 8) → (B, 8)
}
```

### 8.1 各字段详细维度

| 字段 | 形状 | 说明 |
|------|------|------|
| `pixel_values` | `(B, 6, 224, 224)` | 2张图像各3通道=6通道；若1张图则3通道 |
| `input_ids` | `(B, L)` | L 为 batch 内最长序列长度 |
| `attention_mask` | `(B, L)` | 非 padding 位置为 1 |
| `labels` | `(B, L)` | 仅 action token 位置有有效值，其余为 -100 |
| `actions` | `(B, 8, 7)` | **这是 L1 loss 的 ground truth** |
| `proprio` | `(B, 8)` | EEF 状态 6D + 夹爪 2D |

---

## 9. 关键常量

定义在 `prismatic/vla/constants.py`，会根据命令行参数自动检测平台：

### LIBERO 常量

```python
NUM_ACTIONS_CHUNK = 8       # Action chunk 大小（当前+7个未来）
ACTION_DIM = 7              # 动作维度: delta_xyz(3) + delta_rpy(3) + gripper(1)
PROPRIO_DIM = 8             # 本体感知维度: EEF_state(6) + gripper_state(2)
ACTION_PROPRIO_NORMALIZATION_TYPE = "bounds_q99"  # Q1/Q99 分位数归一化
NUM_TOKENS = 64             # Action token 序列固定长度（不足则填充）
```

### 动作维度详解

```
action (7D):
  [0] Δx        末端执行器 x 方向位移增量
  [1] Δy        末端执行器 y 方向位移增量
  [2] Δz        末端执行器 z 方向位移增量
  [3] Δroll     末端执行器 roll 旋转增量
  [4] Δpitch    末端执行器 pitch 旋转增量
  [5] Δyaw      末端执行器 yaw 旋转增量
  [6] gripper   夹爪状态: 1.0 = 张开, 0.0 = 闭合
```

### 本体感知维度详解

```
proprio (8D):
  [0:6] EEF_state       末端执行器状态 (来自 observation["state"][:, :6])
                         = EEF xyz (3) + EEF euler_angles (3)
  [6:8] gripper_state   夹爪状态 (来自 observation["state"][:, -2:])
                         = gripper_qpos (1) + gripper_qvel (1)
```

---

## 10. 数据统计文件（dataset_statistics.json）

训练时自动计算并缓存，推理时用于反归一化。格式：

```json
{
  "libero_spatial_no_noops": {
    "action": {
      "mean": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5],
      "std":  [0.01, 0.01, 0.01, 0.05, 0.05, 0.05, 0.5],
      "max":  [0.05, 0.05, 0.05, 0.3, 0.3, 0.3, 1.0],
      "min":  [-0.05, -0.05, -0.05, -0.3, -0.3, -0.3, 0.0],
      "q01":  [-0.02, -0.02, -0.02, -0.1, -0.1, -0.1, 0.0],
      "q99":  [0.02, 0.02, 0.02, 0.1, 0.1, 0.1, 1.0]
    },
    "proprio": {
      "mean": [...],
      "std":  [...],
      "max":  [...],
      "min":  [...],
      "q01":  [...],
      "q99":  [...]
    },
    "num_transitions": 50000,
    "num_trajectories": 500
  }
}
```

**重要**：推理时需要 `q01` 和 `q99` 来反归一化模型输出：

```python
# 反归一化公式
raw_action = (normalized_action + 1) / 2 * (q99 - q01) + q01
```

---

## 11. 如何制作自定义数据集

如果你想让 VLA-Adapter 训练自己的数据，需要完成以下步骤：

### 11.1 方式一：转换为 RLDS/TFDS 格式（推荐）

需要将数据转为 TFDS 格式的 TFRecord。每条轨迹必须包含：

```python
# 必须字段
trajectory = {
    "observation": {
        "image":       np.ndarray (T, H, W, 3) uint8,    # 主相机 RGB
        "state":       np.ndarray (T, 8) float32,        # 本体感知状态
        # 可选：
        "wrist_image": np.ndarray (T, H, W, 3) uint8,    # 腕部相机 RGB
    },
    "action":          np.ndarray (T, 7) float32,         # 7D 动作
    "language_instruction": str,                           # 任务指令
}
```

**动作格式要求**：
- 维度 [0:3]：末端执行器位置增量 (delta xyz)
- 维度 [3:6]：末端执行器旋转增量 (delta roll-pitch-yaw)
- 维度 [6]：夹爪动作（归一化前：-1=张开, 1=闭合；transform 后转为 0=闭合, 1=张开）

**状态格式要求**：
- 维度 [0:6]：末端执行器位姿 (xyz + euler)
- 维度 [-2:]：夹爪状态 (qpos + qvel)

### 11.2 方式二：使用 PyTorch Dataset

参考 `DummyDataset` 类，实现自定义 `__getitem__`：

```python
class MyDataset(Dataset):
    def __getitem__(self, idx):
        image = ...          # PIL Image (224×224)
        action = ...         # np.ndarray (7,), 范围 [-1, 1]
        instruction = ...    # str

        # 构造与 RLDSBatchTransform 输出一致的 dict
        return dict(
            pixel_values=image_transform(image),
            input_ids=input_ids,
            labels=labels,
            dataset_name="my_dataset",
            actions=actions_chunk,      # (8, 7)
            proprio=proprio,            # (8,) 如果 use_proprio
        )
```

### 11.3 注册新数据集

1. 在 `configs.py` 中添加数据集配置
2. 在 `transforms.py` 中添加 `standardize_fn` 和注册到 `OXE_STANDARDIZATION_TRANSFORMS`
3. 将 TFDS 格式数据放入 `data_root_dir/<dataset_name>/` 目录

---

## 附录：完整训练命令参数与数据的对应关系

```bash
CUDA_VISIBLE_DEVICES=0 torchrun ... vla-scripts/finetune.py \
  --data_root_dir data/libero \               # TFDS 数据根目录
  --dataset_name libero_spatial_no_noops \    # 数据集名称（对应 configs.py 中的 key）
  --num_images_in_input 2 \                   # 使用 2 张图（主相机 + 腕部）
  --use_proprio True \                        # 包含本体感知输入 (8D)
  --batch_size 1 \                            # 每 batch 1 条样本
  --use_minivlm True \                        # 使用 Qwen2.5-0.5B 架构
  --image_aug True \                          # 开启图像数据增强
  --use_lora True \                           # 使用 LoRA 微调
  --use_pro_version True                      # 使用 Pro 版本 action head
  # ↓ 以下为默认值，命令中未显式指定
  # --use_l1_regression True（default=True）  # ⚠️ 默认开启！使用连续动作头 L1 loss
  #                                           # 而非 next-token prediction loss
  # --use_film False（default=False）
  # --use_fz False（default=False）
```

对应的数据维度：

| 维度 | 说明 |
|------|------|
| 输入图像 | `(B, 6, 224, 224)` — 主相机(3) + 腕部(3) 在 channel 维度拼接 |
| 输入文本 | `"What action should the robot take to {instruction}?"` → tokenized |
| 输入本体感知 | `(B, 8)` — EEF 位姿(6D) + 夹爪状态(2D) |
| Action Ground Truth | `(B, 8, 7)` — 8步 action chunk，每步 7D，用于 L1 loss |
| Token Supervision | `labels` 末尾 65 位有效 — 64个 action token + 1个 stop token |
| Loss 函数 | **L1 loss**（`use_l1_regression=True`），作用于 action head 的连续输出与 `actions` 之间 |
