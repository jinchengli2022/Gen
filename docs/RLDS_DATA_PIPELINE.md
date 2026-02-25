# RLDS 数据写入管道

## 概述

本文档记录了 RLDS/TFDS 格式数据写入管道的完整实现，使 Gen/ 管道生成的仿真数据能够直接被 VLA-Adapter 按照 LIBERO 原有框架读取并进入训练流程。

### 设计目标

1. **格式兼容**：生成的数据与 LIBERO RLDS 格式完全一致（state/action/image 结构相同）
2. **无缝接入**：复用 VLA-Adapter 中已有的 `libero_dataset_transform`，无需编写新的 transform
3. **可视化检查**：同时保存渲染视频 + 检查用 HDF5，方便人工验证数据质量
4. **TFDS 标准**：通过 `tfds.core.GeneratorBasedBuilder` 写入标准 TFRecord，确保 `tfds.builder_from_directory()` 可直接加载

---

## 修改文件清单

| # | 文件路径 | 操作 | 作用 |
|---|---------|------|------|
| 1 | `Gen/utils/data_writer.py` | **修改** | 新增 `RLDSDataWriter` 类 + 更新 `create_data_writer` 工厂函数 |
| 2 | `Gen/configs/config.py` | **修改** | `DataCollectionConfig` 添加 `language_instruction` 字段 |
| 3 | `Gen/configs/examples/pouring_water_trajgen.json` | **修改** | `save_format` → `"rlds"` + 添加 `language_instruction` |
| 4 | `Gen/scripts/gen.py` | **修改** | 当 `save_format=="rlds"` 时传递额外参数给 data writer |
| 5 | `Gen/tfds_builders/pouringwater_generated/__init__.py` | **新建** | TFDS builder 包入口 |
| 6 | `Gen/tfds_builders/pouringwater_generated/pouringwater_generated_dataset_builder.py` | **新建** | TFDS `GeneratorBasedBuilder` 子类，定义数据集 schema |
| 7 | `prismatic/vla/datasets/rlds/dataset.py` | **修改** | `tfds.builder()` fallback 到 `tfds.builder_from_directory()` |
| 8 | `prismatic/vla/datasets/rlds/oxe/configs.py` | **修改** | 注册 `pouringwater_generated` 数据集配置 |
| 9 | `prismatic/vla/datasets/rlds/oxe/transforms.py` | **修改** | 注册 `pouringwater_generated` 使用 `libero_dataset_transform` |
| 10 | `prismatic/vla/datasets/rlds/oxe/mixtures.py` | **修改** | 注册 `pouringwater_generated` mixture |

---

## 一、Gen 侧修改（数据生成端）

### 1.1 `Gen/utils/data_writer.py` — RLDSDataWriter 类

**修改内容**：在已有的 `HDF5DataWriter`、`PickleDataWriter` 之后新增 `RLDSDataWriter` 类（约 360 行新代码），并更新 `create_data_writer` 工厂函数支持 `format="rlds"`。

#### RLDSDataWriter 类结构

```
RLDSDataWriter(DataWriter)
├── __init__()           — 初始化参数、创建视频目录
├── _extract_rlds_data() — 从 episode_data 提取 RLDS 格式数据（核心转换逻辑）
├── _save_video()        — 保存图像序列为 mp4
├── write_episode()      — 提取数据 + 暂存内存 + 保存视频
├── _save_check_hdf5()   — 保存单个 episode 为检查用 HDF5
├── _write_tfrecords()   — 通过 TFDS builder API 写入标准 TFRecord
└── finalize()           — 随机选 episode 保存检查 HDF5 + 写入 TFRecord
```

#### 关键数据转换逻辑 (`_extract_rlds_data`)

```
gen.py collect_episode() 的输出
├── observations[0:T]  (取前 T 个，丢弃最后一个 next_obs)
│   └── raw_obs
│       ├── agentview_image        → np.flipud() → observation/image      (uint8, 256×256×3)
│       ├── robot0_eye_in_hand_image → np.flipud() → observation/wrist_image (uint8, 256×256×3)
│       ├── robot0_eef_pos (3)     ─┐
│       ├── robot0_eef_quat (4)    ─┤→ quat2axisangle → observation/state  (float32, 8D)
│       └── robot0_gripper_qpos (2)─┘
└── actions[0:T]                   → action                                (float32, 7D)
```

**obs/action 对齐**：`observations` 有 T+1 个（初始 obs + T 个 next_obs），`actions` 有 T 个。RLDS 的 step_t = (obs_t, action_t)，因此只使用 `observations[0:T]`。

**图像翻转**：robosuite 使用 OpenGL 渲染，返回的图像是上下颠倒的。LIBERO RLDS 数据中图像已旋转 180°（参考 `regenerate_libero_dataset.py` 第 8-9 行注释）。这里使用 `np.flipud()` 翻转。

**State 构造**：
| 索引 | 字段 | 来源 | 说明 |
|------|------|------|------|
| [0:3] | eef_pos | `robot0_eef_pos` | 末端执行器位置 |
| [3:6] | eef_axisangle | `quat2axisangle(robot0_eef_quat)` | 末端执行器旋转（axis-angle 表示） |
| [6:8] | gripper_qpos | `robot0_gripper_qpos` | 夹爪关节位置（2D） |

> 注意：与 LIBERO 完全一致。`libero_dataset_transform` 中 `state[:, :6]` 取 EEF_state，`state[:, -2:]` 取 gripper_state。

**Action**：gen.py 的 `WaypointPolicy.get_action()` 返回 `[delta_pos/max_dpos, delta_rot_axisangle/max_drot, gripper]` 7D，这是归一化后的控制器输入，与 LIBERO 的 action 格式一致。

#### TFRecord 写入 (`_write_tfrecords`)

使用内联的 `tfds.core.GeneratorBasedBuilder` 子类，通过 `builder.download_and_prepare()` 自动完成 TFRecord 序列化 + 元数据（`dataset_info.json`、`features.json`）生成。确保输出格式与 TFDS 标准完全兼容。

#### 检查 HDF5 (`_save_check_hdf5`)

随机挑选一个 episode，保存为 HDF5，结构为：

```
check_episode_X.hdf5
├── observation/
│   ├── image          (T, 256, 256, 3)  uint8
│   ├── wrist_image    (T, 256, 256, 3)  uint8
│   └── state          (T, 8)            float32
├── action             (T, 7)            float32
└── attrs:
    ├── language_instruction  string
    ├── num_steps             int
    ├── state_format          "eef_pos(3) + eef_axisangle(3) + gripper_qpos(2)"
    └── action_format         "delta_pos(3) + delta_axisangle(3) + gripper(1)"
```

#### 视频保存 (`_save_video`)

每个 episode 保存两个视角的 mp4 视频：
- `ep{idx:04d}_agentview.mp4` — 第三人称相机
- `ep{idx:04d}_eye_in_hand.mp4` — 腕部相机

#### `create_data_writer` 工厂函数更新

```python
# 原先
def create_data_writer(output_dir, env_name, format="hdf5") -> DataWriter:
    # 只支持 "hdf5", "pickle"

# 修改后
def create_data_writer(output_dir, env_name, format="hdf5", **kwargs) -> DataWriter:
    # 新增 "rlds" 选项，kwargs 传递给 RLDSDataWriter
    # kwargs 包括: language_instruction, image_key_primary, image_key_wrist,
    #              dataset_name, save_video, video_fps
```

---

### 1.2 `Gen/configs/config.py` — 添加 language_instruction 字段

```python
# 新增字段
language_instruction: str = ""  # Task language instruction (required for rlds format)

# save_format 注释更新
save_format: str = "hdf5"  # "hdf5", "pickle", or "rlds"
```

---

### 1.3 `Gen/configs/examples/pouring_water_trajgen.json` — 示例配置更新

```diff
-    "save_format": "hdf5",
+    "save_format": "rlds",
+    "language_instruction": "pour the water from the cup into the bowl",
```

---

### 1.4 `Gen/scripts/gen.py` — 传递 RLDS 参数

在 `create_data_writer` 调用之前，根据 `save_format` 构造额外参数：

```python
# 新增代码
writer_kwargs = {}
if config.save_format == "rlds":
    writer_kwargs["language_instruction"] = config.language_instruction
    writer_kwargs["image_key_primary"] = "agentview_image"
    writer_kwargs["image_key_wrist"] = "robot0_eye_in_hand_image"

writer = create_data_writer(
    output_dir=config.output_dir,
    env_name=config.env_name,
    format=config.save_format,
    **writer_kwargs            # ← 新增
)
```

---

### 1.5 `Gen/tfds_builders/pouringwater_generated/` — TFDS Builder 包

新建的 TFDS builder 定义了数据集 schema，用作备用注册方式。包含：

- `__init__.py` — 导入 builder class
- `pouringwater_generated_dataset_builder.py` — `PouringwaterGenerated(tfds.core.GeneratorBasedBuilder)` 子类

schema 定义（与 `RLDSDataWriter._write_tfrecords` 中的内联 builder 一致）：

```python
features = FeaturesDict({
    "steps": Dataset({
        "observation": FeaturesDict({
            "image":       Image(shape=(256, 256, 3), dtype=uint8, encoding="png"),
            "wrist_image": Image(shape=(256, 256, 3), dtype=uint8, encoding="png"),
            "state":       Tensor(shape=(8,), dtype=float32),
        }),
        "action":                Tensor(shape=(7,), dtype=float32),
        "reward":                Scalar(dtype=float32),
        "discount":              Scalar(dtype=float32),
        "is_first":              bool,
        "is_last":               bool,
        "is_terminal":           bool,
        "language_instruction":  Text(),
    }),
})
```

---

## 二、VLA-Adapter 侧修改（训练读取端）

### 2.1 `prismatic/vla/datasets/rlds/dataset.py` — Builder 加载 fallback

**修改位置**：`make_dataset_from_rlds()` 函数中的 `tfds.builder()` 调用。

```python
# 原先
builder = tfds.builder(name, data_dir=data_dir)

# 修改后：增加 fallback，支持本地生成的数据集
try:
    builder = tfds.builder(name, data_dir=data_dir)
except Exception:
    import os
    local_dataset_dir = os.path.join(data_dir, name)
    if os.path.isdir(local_dataset_dir):
        builder = tfds.builder_from_directory(local_dataset_dir)
    else:
        raise
```

**原因**：`tfds.builder(name, ...)` 要求 builder 已在 TFDS 注册中心注册（通过 pip install）。对于本地通过 `RLDSDataWriter` 生成的数据集，标准目录结构已包含完整的 `dataset_info.json` + `features.json`，可以用 `tfds.builder_from_directory()` 直接从目录加载，无需安装 builder 包。

---

### 2.2 `prismatic/vla/datasets/rlds/oxe/configs.py` — 数据集配置注册

在 `OXE_DATASET_CONFIGS` 字典中新增条目：

```python
### Generated datasets (from Gen/ pipeline, same format as LIBERO)
"pouringwater_generated": {
    "image_obs_keys": {"primary": "image", "secondary": None, "wrist": "wrist_image"},
    "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
    "state_obs_keys": ["EEF_state", "gripper_state"],
    "state_encoding": StateEncoding.POS_EULER,
    "action_encoding": ActionEncoding.EEF_POS,
},
```

与 LIBERO 各数据集配置完全相同，因为我们的 state/action/image 字段结构一致。

---

### 2.3 `prismatic/vla/datasets/rlds/oxe/transforms.py` — Transform 注册

在 `OXE_STANDARDIZATION_TRANSFORMS` 字典中新增条目：

```python
### Generated datasets (from Gen/ pipeline, same format as LIBERO)
"pouringwater_generated": libero_dataset_transform,
```

直接复用 `libero_dataset_transform`，该函数执行：
1. gripper action: `[-1, 1]` → `clip [0, 1]` → `invert` → `1=open, 0=close`
2. `EEF_state = state[:, :6]`（eef_pos + eef_axisangle）
3. `gripper_state = state[:, -2:]`（gripper_qpos 2D）

---

### 2.4 `prismatic/vla/datasets/rlds/oxe/mixtures.py` — Mixture 注册

```python
# === Generated Datasets (from Gen/ pipeline) ===
"pouringwater_generated": [
    ("pouringwater_generated", 1.0),
],
```

---

## 三、输出目录结构

运行数据生成后，`output_dir` 下的文件结构：

```
data/pouring_water_generated/
├── pouringwater_generated/              # TFDS 数据集目录
│   └── 1.0.0/
│       ├── dataset_info.json            # TFDS 元数据
│       ├── features.json                # 特征定义
│       └── pouringwater_generated-train.tfrecord-*  # TFRecord 数据文件
├── videos/                              # 渲染视频
│   ├── ep0000_agentview.mp4
│   ├── ep0000_eye_in_hand.mp4
│   ├── ep0001_agentview.mp4
│   ├── ep0001_eye_in_hand.mp4
│   └── ...
├── check/                               # 检查用 HDF5
│   └── check_episode_X.hdf5
└── gen_log_*.log                        # 生成日志
```

---

## 四、完整数据流

```
                        Gen 侧（数据生成）
┌──────────────────────────────────────────────────────┐
│                                                      │
│  gen.py                                              │
│  ├── collect_episode()                               │
│  │   └── observations[T+1], actions[T], rewards[T]   │
│  │                                                   │
│  └── writer.write_episode(episode_data, idx)         │
│      └── RLDSDataWriter._extract_rlds_data()         │
│          │                                           │
│          ├── 图像: raw_obs[agentview_image]           │
│          │         → np.flipud()                     │
│          │         → observation/image (uint8)       │
│          │                                           │
│          ├── 腕部: raw_obs[robot0_eye_in_hand_image]  │
│          │         → np.flipud()                     │
│          │         → observation/wrist_image (uint8)  │
│          │                                           │
│          ├── 状态: eef_pos(3)                         │
│          │       + quat2axisangle(eef_quat)(3)       │
│          │       + gripper_qpos(2)                   │
│          │         → observation/state (float32, 8D)  │
│          │                                           │
│          └── 动作: [delta_pos/max_dpos,               │
│                     delta_rot/max_drot,              │
│                     gripper]                         │
│                    → action (float32, 7D)            │
│                                                      │
│  writer.finalize()                                   │
│  ├── _save_check_hdf5()  → check/check_episode_X.hdf5│
│  └── _write_tfrecords()  → TFDS builder API          │
│      └── download_and_prepare()                      │
│          → pouringwater_generated/1.0.0/*.tfrecord   │
│                                                      │
└──────────────────────────────────────────────────────┘
                           │
                           │ TFRecord + metadata
                           ▼
              VLA-Adapter 侧（训练读取）
┌──────────────────────────────────────────────────────┐
│                                                      │
│  dataset.py :: make_dataset_from_rlds()              │
│  │                                                   │
│  ├── tfds.builder("pouringwater_generated", data_dir)│
│  │   └── fallback: tfds.builder_from_directory()     │
│  │                                                   │
│  ├── dl.DLataset.from_rlds(builder, ...)             │
│  │                                                   │
│  ├── libero_dataset_transform()     [transforms.py]  │
│  │   ├── gripper: clip[0,1] → invert (1=open)       │
│  │   ├── EEF_state = state[:, :6]                    │
│  │   └── gripper_state = state[:, -2:]               │
│  │                                                   │
│  ├── restructure()                  [dataset.py]     │
│  │   └── 合并 state_obs_keys → proprio               │
│  │                                                   │
│  ├── normalize_action_and_proprio() [data_utils.py]  │
│  │                                                   │
│  ├── chunk_act_obs()                [data_utils.py]  │
│  │                                                   │
│  └── → RLDSBatchTransform → PaddedCollator → Model   │
│                                                      │
└──────────────────────────────────────────────────────┘
```

---

## 五、训练使用方式

### 5.1 数据生成

```bash
cd Gen/
python scripts/gen.py --config configs/examples/pouring_water_trajgen.json
```

### 5.2 VLA-Adapter 训练

将 `data_dir` 指向 `output_dir`，数据集名称为 `pouringwater_generated`：

```bash
torchrun --nproc_per_node=1 vla-scripts/finetune.py \
    --data_root_dir /path/to/data/pouring_water_generated \
    --dataset_name pouringwater_generated \
    ...
```

### 5.3 数据检查

```python
import h5py

f = h5py.File("data/pouring_water_generated/check/check_episode_X.hdf5", "r")
print(f.attrs["language_instruction"])   # "pour the water from the cup into the bowl"
print(f.attrs["state_format"])           # "eef_pos(3) + eef_axisangle(3) + gripper_qpos(2)"
print(f.attrs["action_format"])          # "delta_pos(3) + delta_axisangle(3) + gripper(1)"
print(f["observation/image"].shape)      # (T, 256, 256, 3)
print(f["observation/state"].shape)      # (T, 8)
print(f["action"].shape)                 # (T, 7)
f.close()
```

---

## 六、扩展新数据集

如果要用 Gen/ 管道生成新环境的数据集，需要：

1. **Gen 侧**：在 JSON 配置中设 `"save_format": "rlds"` 和 `"language_instruction": "..."`，`dataset_name` 默认为 `{env_name.lower()}_generated`

2. **VLA-Adapter 侧**（3 个文件，各加 1 条）：

   ```python
   # configs.py — OXE_DATASET_CONFIGS
   "new_env_generated": {
       "image_obs_keys": {"primary": "image", "secondary": None, "wrist": "wrist_image"},
       "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
       "state_obs_keys": ["EEF_state", "gripper_state"],
       "state_encoding": StateEncoding.POS_EULER,
       "action_encoding": ActionEncoding.EEF_POS,
   },

   # transforms.py — OXE_STANDARDIZATION_TRANSFORMS
   "new_env_generated": libero_dataset_transform,

   # mixtures.py — OXE_DATASET_MIXTURES
   "new_env_generated": [
       ("new_env_generated", 1.0),
   ],
   ```

   只要 state/action 格式与 LIBERO 一致（8D state + 7D action），就可以直接复用 `libero_dataset_transform`，无需编写新的 transform 函数。
