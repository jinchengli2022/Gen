# Custom Objects Assets

This directory contains XML files for custom objects used in robosuite environments.

## Structure

```
assets/
└── objects/
    ├── yellow_cup.xml
    ├── black_cup.xml
    └── README.md (this file)
```

## XML Format

Objects are defined using MuJoCo XML format. Each object file should contain:

1. **Assets** (optional): Textures, materials
2. **Body**: Main object body with geoms and sites

### Example Structure

```xml
<mujoco model="object_name">
    <asset>
        <texture name="obj_tex" type="cube" builtin="flat" rgb1="1.0 0.0 0.0"/>
        <material name="obj_mat" texture="obj_tex"/>
    </asset>
    
    <worldbody>
        <body name="object_name" pos="0 0 0">
            <geom name="obj_geom" type="cylinder" size="0.04 0.05" mass="0.1"/>
            <site name="obj_top" pos="0 0 0.05" size="0.01"/>
        </body>
    </worldbody>
</mujoco>
```

## Yellow Cup (`yellow_cup.xml`)

- **Color**: Yellow (RGB: 1.0, 0.9, 0.2)
- **Shape**: Cylinder with handle
- **Dimensions**: Radius 0.035m, Height 0.12m (0.06m half-height)
- **Mass**: ~0.065kg total
- **Purpose**: Cup to pour from in pouring task

## Black Cup (`black_cup.xml`)

- **Color**: Dark gray/black (RGB: 0.2, 0.2, 0.2)
- **Shape**: Cylinder with handle
- **Dimensions**: Radius 0.04m, Height 0.10m (0.05m half-height)
- **Mass**: ~0.13kg total (heavier for stability)
- **Purpose**: Target cup in pouring task

## Usage in Environment

The PouringWater environment loads these models using `MujocoXMLObject`:

```python
from robosuite.models.objects import MujocoXMLObject

yellow_cup = MujocoXMLObject(
    fname="path/to/yellow_cup.xml",
    name="yellow_cup",
    joints=[dict(type="free", damping="0.0005")],
    obj_type="all",
    duplicate_collision_geoms=True,
)
```

### Specifying Asset Path

In your configuration file (`pouring_water.json`):

```json
{
    "env_name": "PouringWater",
    "base_path": "/path/to/gen/env",
    ...
}
```

Or use `null` to default to the environment directory:

```json
{
    "base_path": null
}
```

## Customization

To create your own objects:

1. **Copy existing XML**: Use `yellow_cup.xml` or `black_cup.xml` as template
2. **Modify geometry**: Change `size`, `mass`, `rgba` attributes
3. **Add/remove geoms**: Modify the body structure
4. **Update materials**: Change textures and colors
5. **Save as new file**: e.g., `custom_object.xml`
6. **Update environment**: Modify `_load_model()` in environment class

### Key Attributes

- **type**: Geom shape (`cylinder`, `box`, `sphere`, `capsule`, etc.)
- **size**: Dimensions (varies by type)
  - Cylinder: `[radius, half_height]`
  - Box: `[half_x, half_y, half_z]`
  - Sphere: `[radius]`
- **mass**: Object mass in kg
- **rgba**: Color `[red, green, blue, alpha]` (0-1 range)
- **pos**: Position relative to parent `[x, y, z]`

## Sites

Sites are reference points on objects used for:
- Grasping targets
- Orientation tracking
- Visual debugging

Example:
```xml
<site name="cup_top" pos="0 0 0.06" size="0.01"/>
```

## Tips

1. **Mass distribution**: Use realistic masses for stable simulation
2. **Collision geoms**: Keep simple for performance
3. **Visual vs collision**: Can separate for better visuals
4. **Joint damping**: Prevents jittery motion (0.0005 works well)
5. **Test in viewer**: Use robosuite viewer to check appearance

## References

- [MuJoCo XML Reference](http://www.mujoco.org/book/XMLreference.html)
- [robosuite Objects](https://robosuite.ai/docs/modules/objects.html)
