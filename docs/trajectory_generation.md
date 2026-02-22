# Trajectory Generation Feature

This document explains the MimicGen-inspired trajectory generation capabilities added to the data collection framework.

## Overview

The trajectory generation system allows you to **generate large amounts of diverse demonstration data from a single source demonstration**. This is achieved through intelligent pose transformation algorithms that adapt trajectories to new object configurations.

## Core Components

### 1. Source Demo Loader (`utils/source_demo_loader.py`)

Loads and parses human demonstrations from HDF5 files:

```python
from utils.source_demo_loader import SourceDemoLoader

loader = SourceDemoLoader("data/source_demos/demo.hdf5")
demo = loader.get_demo(0)  # Get first demonstration
```

**Features:**
- Extracts EEF poses, gripper actions, object poses
- Supports multiple demonstrations per file
- Provides trajectory segmentation

### 2. Trajectory Generator (`utils/trajectory_generator.py`)

Implements pose transformation algorithms:

```python
from utils.trajectory_generator import TrajectoryGenerator

gen = TrajectoryGenerator(env_interface)

# Generate grasping trajectory
poses, gripper = gen.generate_grasp_trajectory(
    target_object_pose=object_pose,
    grasp_pose_in_object=grasp_offset
)

# Transform source trajectory to new scene
new_poses = gen.transform_trajectory_to_new_scene(
    src_eef_poses=source_poses,
    src_object_pose=old_object_pose,
    target_object_pose=new_object_pose,
    current_eef_pose=robot_pose
)
```

**Key Algorithms:**
- **Pose Interpolation**: Linear for position, Slerp for rotation
- **Vector Scaling**: Adapts XY plane motion to new distances/angles
- **Z-axis Correction**: Compensates for height differences
- **Relative Pose Preservation**: Maintains spatial relationships

### 3. Waypoint Policy (`scripts/gen.py`)

Executes pre-computed trajectories:

```python
from scripts.gen import WaypointPolicy

policy = WaypointPolicy(
    env_interface=env,
    waypoint_poses=target_poses,
    gripper_actions=gripper_commands
)

action = policy.get_action(observation)
```

## Usage

### Step 1: Collect Source Demonstration

First, collect a high-quality human demonstration:

```bash
# Use teleoperation or kinesthetic teaching
python scripts/gen.py --config configs/examples/pouring_water.json --render
```

This creates: `data/pouring_water/demo.hdf5`

### Step 2: Configure Trajectory Generation

Create a config file with trajectory generation enabled:

```json
{
    "env_name": "PouringWater",
    "use_trajectory_generation": true,
    "source_demo_path": "data/pouring_water/demo.hdf5",
    "num_episodes": 1000,
    ...
}
```

### Step 3: Generate Data

Run generation with the new config:

```bash
python scripts/gen.py --config configs/examples/pouring_water_trajgen.json
```

**Result:** 1000 diverse trajectories generated from the single source demo!

## Configuration Parameters

### New Config Fields

| Parameter | Type | Description |
|-----------|------|-------------|
| `use_trajectory_generation` | bool | Enable trajectory generation mode |
| `source_demo_path` | str | Path to source HDF5 demonstration |
| `policy_type` | str | Set to "trajectory_gen" |

### Example Config

```json
{
    "env_name": "PouringWater",
    "robots": "Panda",
    "num_episodes": 100,
    "use_trajectory_generation": true,
    "source_demo_path": "data/source_demos/pouring_demo.hdf5",
    "output_dir": "data/generated_data"
}
```

## How It Works

### Transformation Pipeline

```
1. env.reset() → New random scene

2. Load Source Demo
   ├─ EEF trajectory: src_poses (N, 4, 4)
   ├─ Gripper actions: src_grippers (N, 1)
   └─ Object poses: src_objects

3. Get Current Scene State
   ├─ Current robot pose
   ├─ Current object poses
   └─ Target object poses

4. Transform Trajectory
   ├─ Compute relative transformations
   ├─ Apply vector scaling (XY plane)
   ├─ Correct Z-axis heights
   └─ Interpolate rotations (Slerp)

5. Execute Transformed Trajectory
   ├─ WaypointPolicy follows poses
   ├─ Collect observations/actions
   └─ Save to HDF5
```

### Vector Scaling Algorithm

**Problem:** Objects move, distances change between episodes

**Solution:**
```python
# Compute scaling and rotation
src_vec = src_end[:2] - src_start[:2]
target_vec = target_end[:2] - target_start[:2]

scale = ||target_vec|| / ||src_vec||
angle_diff = atan2(target_vec) - atan2(src_vec)

# Transform each point
for point in src_trajectory:
    rel_pos = point - src_start
    
    # Scale and rotate XY
    rel_xy = R(angle_diff) @ rel_pos[:2] * scale
    
    # Correct Z with linear interpolation
    rel_z = rel_pos[2] + z_correction * progress
    
    new_point = target_start + [rel_xy, rel_z]
```

**Result:** Trajectory adapts to new scene while preserving motion shape

## Advanced: Subtask Segmentation

For complex tasks like PouringWater (grasp → pour), implement subtask logic:

```python
# In trajectory_generator.py
class SubtaskTrajectoryGenerator:
    def generate_full_task(self, env, src_demo):
        trajectories = []
        
        # Subtask 1: Grasp
        grasp_traj = self.generate_grasp_trajectory(...)
        trajectories.append(grasp_traj)
        
        # Subtask 2: Pour (use source demo transformation)
        pour_traj = self.transform_trajectory_to_new_scene(...)
        trajectories.append(pour_traj)
        
        return np.concatenate(trajectories)
```

## Troubleshooting

### Issue: "Demo file not found"

**Solution:** Check `source_demo_path` in config:
```bash
ls data/source_demos/  # Verify file exists
```

### Issue: "Action dimension mismatch"

**Solution:** Ensure source demo uses same robot:
```json
{
    "robots": "Panda",  # Must match source demo robot
    ...
}
```

### Issue: "Trajectory fails to reach target"

**Possible causes:**
1. Source demo too short → Increase `horizon`
2. Transformation error → Check object poses are correct
3. IK failures → Adjust controller gains

**Debug:** Enable rendering to visualize:
```bash
python scripts/gen.py --config ... --render
```

## Comparison: Random vs Trajectory Generation

| Metric | Random Policy | Trajectory Generation |
|--------|---------------|----------------------|
| Success Rate | ~0-5% | ~60-90% |
| Data Efficiency | Low (need many episodes) | High (1 demo → 1000s) |
| Quality | Inconsistent | Consistent, structured |
| Diversity | High (random) | Moderate (scene variations) |
| Human Effort | None | 1 demonstration |

## Future Enhancements

1. **Multi-source selection**: Select different source demos per subtask
2. **Noise injection**: Add Gaussian noise to actions for robustness
3. **Failure recovery**: Detect failures and retry with different parameters
4. **Online refinement**: Use RL to fine-tune generated trajectories

## References

- **MimicGen Paper**: [arxiv.org/abs/2310.17596](https://arxiv.org/abs/2310.17596)
- **Pose Interpolation**: Slerp (Spherical Linear Interpolation)
- **Vector Scaling**: Geometric transformation preserving motion structure
