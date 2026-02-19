# Configuration Files

This directory contains configuration files for data collection.

## Structure

- `config.py`: Defines the `DataCollectionConfig` dataclass
- `examples/`: Example configuration files for different tasks

## Configuration Format

All configuration files should be in JSON format with the following structure:

```json
{
    "env_name": "PickPlaceCan",
    "robots": "Panda",
    "controller_config": null,
    "has_renderer": false,
    "has_offscreen_renderer": true,
    "use_camera_obs": false,
    "use_object_obs": true,
    "reward_shaping": true,
    "control_freq": 20,
    "horizon": 500,
    "camera_names": ["agentview", "robot0_eye_in_hand"],
    "camera_heights": 224,
    "camera_widths": 224,
    "camera_depths": false,
    "num_episodes": 10,
    "output_dir": "data/demo",
    "save_format": "hdf5",
    "policy_type": "random",
    "use_scripted_policy": false,
    "ignore_done": false,
    "hard_reset": true
}
```

## Configuration Fields

### Environment Settings

- **env_name** (str): Name of the robosuite environment (e.g., "PickPlaceCan", "Stack", "PouringWater")
- **robots** (str): Robot model (e.g., "Panda", "Sawyer", "UR5e")
- **controller_config** (dict|null): Custom controller configuration (see below)
- **control_freq** (int): Control frequency in Hz (default: 20)
- **horizon** (int): Maximum episode length (default: 500)
- **reward_shaping** (bool): Enable reward shaping (default: true)

### Controller Configuration

For custom controllers (e.g., OSC_POSE), set `controller_config`:

```json
{
    "controller_config": {
        "type": "OSC_POSE",
        "input_max": 1,
        "input_min": -1,
        "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
        "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
        "kp": 150,
        "damping_ratio": 1,
        "impedance_mode": "fixed",
        "control_delta": true,
        "uncouple_pos_ori": true
    }
}
```

For default controller, set to `null`.

### Rendering Settings

- **has_renderer** (bool): Enable on-screen rendering (default: false)
- **has_offscreen_renderer** (bool): Enable offscreen rendering for camera obs (default: true)

### Observation Settings

- **use_camera_obs** (bool): Include camera observations (default: false)
- **use_object_obs** (bool): Include object state observations (default: true)
- **camera_names** (list): List of camera names (e.g., ["agentview", "robot0_eye_in_hand"])
- **camera_heights** (int): Image height in pixels (default: 224)
- **camera_widths** (int): Image width in pixels (default: 224)
- **camera_depths** (bool): Include depth images (default: false)

### Data Collection Settings

- **num_episodes** (int): Number of episodes to collect (default: 10)
- **output_dir** (str): Directory to save data (default: "data/demo")
- **save_format** (str): Data format - "hdf5" or "pickle" (default: "hdf5")
- **policy_type** (str): Policy type (default: "random")
- **use_scripted_policy** (bool): Use scripted policy if available (default: false)
- **ignore_done** (bool): Ignore done signal (default: false)
- **hard_reset** (bool): Hard reset between episodes (default: true)

## Example Configs

### 1. PickPlace Demo (State-only)

Simple pick-place task with state observations:

```bash
python scripts/gen.py --config configs/examples/pickplace_demo.json
```

Features:
- Panda robot
- State observations only
- 10 episodes
- HDF5 format

### 2. PouringWater (Custom Environment)

Custom pouring task with OSC_POSE controller:

```bash
python scripts/gen.py --config configs/examples/pouring_water.json
```

Features:
- UR5e robot
- OSC_POSE controller with custom parameters
- 100 episodes
- 1000 timesteps per episode

### 3. Stack with Vision

Stack task with camera observations:

```bash
python scripts/gen.py --config configs/examples/stack_with_vision.json
```

Features:
- Sawyer robot
- Camera observations (agentview + eye-in-hand)
- 256x256 images
- 50 episodes

## Using Config Files

### Basic Usage

```bash
python scripts/gen.py --config path/to/config.json
```

### With Real-time Rendering Override

The `--render` flag overrides the `has_renderer` setting in the config:

```bash
python scripts/gen.py --config configs/examples/pouring_water.json --render
```

This is useful for:
- Debugging environment behavior
- Visualizing data collection
- Testing without modifying config files

## Creating Custom Configs

1. Copy an example config:
   ```bash
   cp configs/examples/pickplace_demo.json configs/my_task.json
   ```

2. Edit the parameters:
   ```json
   {
       "env_name": "Door",
       "robots": "Panda",
       "num_episodes": 50,
       "output_dir": "data/door_opening",
       ...
   }
   ```

3. Run collection:
   ```bash
   python scripts/gen.py --config configs/my_task.json
   ```

## Tips

- Use `"has_renderer": false` for faster headless collection
- Set `use_camera_obs: true` only if needed (slower but required for vision models)
- Adjust `horizon` based on task complexity
- Use descriptive `output_dir` names for organization
- Start with fewer `num_episodes` for testing

## Troubleshooting

**Config file not found:**
```
✗ Error: Configuration file not found: path/to/config.json
```
→ Check the path is correct (relative to where you run the script)

**Invalid JSON:**
```
✗ Failed to load config: Expecting property name enclosed in double quotes
```
→ Validate JSON syntax (use `null` instead of `None`, lowercase `true/false`)

**Environment fails to load:**
```
✗ Failed to load environment: ...
```
→ Check `env_name` is correct and environment exists
→ For custom environments (like PouringWater), ensure they're registered in `env_interfaces/robosuite_env.py`
