# IRN G01 - ArUco Marker Display in Gazebo Ignition

This package provides functionality to display ArUco markers in Gazebo Ignition simulation environment. It supports multiple formats (PNG, SVG, PDF) and allows dynamic world modification.

## Features

- **ArUco Marker Generation**: Generate ArUco markers with custom IDs and sizes
- **Format Conversion**: Convert between PNG, JPG, BMP, and SVG formats
- **World Management**: Add, remove, and update ArUco markers in Gazebo worlds
- **Dynamic Modification**: Modify world configuration at runtime
- **Launch Support**: Ready-to-use launch files for Gazebo Ignition

## Installation

### Prerequisites

- ROS 2 (Humble or newer)
- Gazebo Ignition (tested with Garden)
- Python 3.8+

### Optional Dependencies

For full functionality with format conversion, install:
```bash
pip install opencv-python pillow
```

### Building

```bash
cd ~/ros2_ws/src
git clone <repository-url> irn_g01
cd ~/ros2_ws
colcon build --symlink-install --packages-select irn_g01
source install/setup.bash
```

## Usage

### 1. Generating ArUco Markers

Generate a PNG ArUco marker with ID 0:

```bash
python3 irn_g01/scripts/generate_aruco.py generate --id 0 --output marker_0.png
```

Convert to SVG format:

```bash
python3 irn_g01/scripts/generate_aruco.py convert marker_0.png --format svg --output marker_0.svg
```

### 2. Managing Worlds

List all ArUco markers in a world:

```bash
python3 irn_g01/scripts/manage_world.py list irn_g01/worlds/aruco_world.world
```

Add a new marker to a world:

```bash
python3 irn_g01/scripts/manage_world.py add irn_g01/worlds/aruco_world.world \
  --id 1 --name my_marker --x 1.0 --y 2.0 --z 0
```

Update a marker's position:

```bash
python3 irn_g01/scripts/manage_world.py update irn_g01/worlds/aruco_world.world \
  --name my_marker --x 0.5 --y 0.5 --z 0
```

Remove a marker:

```bash
python3 irn_g01/scripts/manage_world.py remove irn_g01/worlds/aruco_world.world --name my_marker
```

### 3. Launching Gazebo Ignition

Launch Gazebo Ignition with the default ArUco world:

```bash
ros2 launch irn_g01 gazebo_aruco.launch.py
```

Launch with a custom world file:

```bash
ros2 launch irn_g01 gazebo_aruco.launch.py world:=/path/to/custom.world
```

## Python API

### Generate ArUco Marker

```python
from irn_g01 import generate_aruco_marker

# Generate marker with ID 0
marker = generate_aruco_marker(
    aruco_id=0,
    marker_size=200,
    output_path='marker.png'
)
```

### Convert Format

```python
from irn_g01 import convert_aruco_format

# Convert PNG to SVG
success = convert_aruco_format(
    input_path='marker.png',
    output_format='svg',
    output_path='marker.svg'
)
```

### Manage World

```python
from irn_g01 import WorldManager

# Load world
manager = WorldManager('worlds/aruco_world.world')

# Add marker
manager.add_aruco_marker(
    marker_id=1,
    position=(1.0, 0.0, 0.0),
    rotation=(0, 0, 0),
    marker_name='test_marker'
)

# Update marker
manager.update_marker_pose(
    marker_name='test_marker',
    position=(2.0, 1.0, 0.0)
)

# List markers
markers = manager.list_markers()
print(f"Markers: {markers}")

# Save changes
manager.save_world('worlds/aruco_world_modified.world')
```

## World File Format

The world files are in SDF 1.10 format. ArUco markers are defined as models with the following structure:

```xml
<model name="aruco_marker_0">
  <pose>0 0 0 0 0 0</pose>
  <include>
    <uri>model://aruco_marker</uri>
  </include>
</model>
```

## Troubleshooting

### Issue: "When trying to modify the world it doesn't work"

**Solution**: Make sure to:
1. Use the provided `manage_world.py` script to modify worlds
2. Always call `save_world()` after making changes
3. Ensure the world file is properly formatted XML
4. Check file permissions (world file must be writable)

### Issue: ArUco marker not displaying in Gazebo

**Solution**:
1. Verify the marker model exists in the Gazebo model path
2. Check that the world file syntax is correct
3. Ensure Gazebo can find the models directory:
   ```bash
   export IGN_GAZEBO_MODEL_PATH=$IGN_GAZEBO_MODEL_PATH:$(ros2 pkg prefix irn_g01)/share/irn_g01/models
   ```

### Issue: Import errors with OpenCV or Pillow

**Solution**: Install optional dependencies:
```bash
pip install opencv-python pillow
```

## File Structure

```
irn_g01/
├── irn_g01/
│   ├── __init__.py                 # Package initialization
│   ├── aruco_utils.py             # ArUco generation and conversion utilities
│   ├── world_manager.py           # World file management
│   ├── models/
│   │   └── aruco_marker/
│   │       └── model.sdf          # ArUco marker SDF model
│   ├── worlds/
│   │   └── aruco_world.world      # Default world with ArUco marker
│   ├── launch/
│   │   └── gazebo_aruco.launch.py # Gazebo launch file
│   └── scripts/
│       ├── generate_aruco.py      # CLI tool for generating markers
│       └── manage_world.py        # CLI tool for managing worlds
├── test/                          # Unit tests
├── package.xml                    # ROS 2 package manifest
├── setup.py                       # Python setup configuration
└── readme.md                      # This file
```

## Contributing

Contributions are welcome! Please follow the ROS 2 coding standards and include tests for new features.

## License

Apache License 2.0

## Contact

Maintainer: dacu (daddek9@gmail.com)