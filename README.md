# Robosuite Data Generation Framework

A modular and extensible framework for collecting robotic manipulation demonstration data from robosuite environments.

## 📁 Project Structure

```
gen/
├── scripts/               # 可执行脚本
│   ├── simple_collect.py      # 通用数据收集脚本
│   └── collect_pouring.py     # PouringWater专用收集脚本
├── configs/               # 配置文件
│   └── config.py             # 数据收集配置类
├── env_interfaces/        # 环境接口/包装器
│   └── robosuite_env.py      # Robosuite环境包装器
├── env/                   # 环境实现
│   └── pouring_water_env.py  # PouringWater自定义环境
├── utils/                 # 工具函数
│   └── data_writer.py        # 数据写入工具(HDF5/Pickle)
├── docs/                  # 文档
│   ├── README.md             # 主文档
│   └── POURING_README.md     # PouringWater环境文档
├── tests/                 # 测试脚本
│   └── test_pouring_env.sh   # 环境测试脚本
├── examples/              # 示例代码(待添加)
├── data/                  # 生成的数据(自动创建)
├── requirements.txt       # Python依赖
└── setup.py              # 项目安装配置
```

## 🚀 Quick Start

### Installation

```bash
cd /home/ljc/Git/Gen_VLA_Adapter/gen
pip install -r requirements.txt
```

### Basic Usage

```bash
# 通用环境数据收集
python scripts/simple_collect.py --env_name PickPlaceCan --num_episodes 10

# 带可视化
python scripts/simple_collect.py --env_name Stack --num_episodes 5 --render

# PouringWater环境
python scripts/collect_pouring.py --num_episodes 10 --render
```

## 📦 Module Overview

### configs/
配置管理模块，定义数据收集的所有参数。

### env_interfaces/
环境接口层，提供统一的环境交互接口，处理观测、动作等。

### env/
具体环境实现，包含自定义环境定义。

### utils/
工具函数库，数据写入、可视化等辅助功能。

### scripts/
可执行脚本，用于实际数据收集任务。

## 🎯 Supported Environments

### Standard Robosuite Environments
- PickPlaceCan, Stack, Door, Wipe, ToolHang
- NutAssembly, TwoArmLift, TwoArmPegInHole
- 更多环境见：https://robosuite.ai/docs/modules/environments.html

### Custom Environments
- **PouringWater**: 倒水任务环境（详见 [docs/POURING_README.md](docs/POURING_README.md)）

## 📖 Documentation

- [Main Documentation](docs/README.md) - 完整使用文档
- [PouringWater Environment](docs/POURING_README.md) - PouringWater环境说明

## 🔧 Development

### Adding New Environments

1. 在 `env/` 目录下创建环境文件
2. 在 `env_interfaces/robosuite_env.py` 中注册环境
3. (可选) 在 `scripts/` 创建专用收集脚本

### Running Tests

```bash
bash tests/test_pouring_env.sh
```

## 📄 License

See main project LICENSE file.
