#!/usr/bin/env python3
"""Script to manage ArUco markers in Gazebo worlds."""

import argparse
import logging
import sys
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from world_manager import WorldManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Manage ArUco markers in Gazebo worlds'
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Add marker command
    add_parser = subparsers.add_parser('add', help='Add an ArUco marker to the world')
    add_parser.add_argument('world', type=str, help='Path to the world file')
    add_parser.add_argument('--id', type=int, required=True, help='Marker ID')
    add_parser.add_argument('--name', type=str, help='Marker name')
    add_parser.add_argument('--x', type=float, default=0, help='X position')
    add_parser.add_argument('--y', type=float, default=0, help='Y position')
    add_parser.add_argument('--z', type=float, default=0, help='Z position')
    add_parser.add_argument('--roll', type=float, default=0, help='Roll rotation')
    add_parser.add_argument('--pitch', type=float, default=0, help='Pitch rotation')
    add_parser.add_argument('--yaw', type=float, default=0, help='Yaw rotation')
    add_parser.add_argument('--output', type=str, help='Output world file')

    # Remove marker command
    remove_parser = subparsers.add_parser('remove', help='Remove an ArUco marker')
    remove_parser.add_argument('world', type=str, help='Path to the world file')
    remove_parser.add_argument('--name', type=str, required=True, help='Marker name')
    remove_parser.add_argument('--output', type=str, help='Output world file')

    # List markers command
    list_parser = subparsers.add_parser('list', help='List all markers in the world')
    list_parser.add_argument('world', type=str, help='Path to the world file')

    # Update marker command
    update_parser = subparsers.add_parser('update', help='Update marker pose')
    update_parser.add_argument('world', type=str, help='Path to the world file')
    update_parser.add_argument('--name', type=str, required=True, help='Marker name')
    update_parser.add_argument('--x', type=float, help='X position')
    update_parser.add_argument('--y', type=float, help='Y position')
    update_parser.add_argument('--z', type=float, help='Z position')
    update_parser.add_argument('--roll', type=float, help='Roll rotation')
    update_parser.add_argument('--pitch', type=float, help='Pitch rotation')
    update_parser.add_argument('--yaw', type=float, help='Yaw rotation')
    update_parser.add_argument('--output', type=str, help='Output world file')

    args = parser.parse_args()

    if args.command == 'add':
        manager = WorldManager(args.world)
        success = manager.add_aruco_marker(
            marker_id=args.id,
            position=(args.x, args.y, args.z),
            rotation=(args.roll, args.pitch, args.yaw),
            marker_name=args.name
        )
        if success:
            output = args.output or args.world
            manager.save_world(output)
            logger.info(f"Marker added and world saved to {output}")
            return 0
        else:
            logger.error("Failed to add marker")
            return 1

    elif args.command == 'remove':
        manager = WorldManager(args.world)
        success = manager.remove_marker(args.name)
        if success:
            output = args.output or args.world
            manager.save_world(output)
            logger.info(f"Marker removed and world saved to {output}")
            return 0
        else:
            logger.error("Failed to remove marker")
            return 1

    elif args.command == 'list':
        manager = WorldManager(args.world)
        markers = manager.list_markers()
        if markers:
            logger.info("Markers in the world:")
            for marker in markers:
                logger.info(f"  - {marker}")
            return 0
        else:
            logger.info("No markers found in the world")
            return 0

    elif args.command == 'update':
        manager = WorldManager(args.world)
        position = (args.x, args.y, args.z) if any([args.x is not None, args.y is not None, args.z is not None]) else None
        rotation = (args.roll, args.pitch, args.yaw) if any([args.roll is not None, args.pitch is not None, args.yaw is not None]) else None

        success = manager.update_marker_pose(args.name, position, rotation)
        if success:
            output = args.output or args.world
            manager.save_world(output)
            logger.info(f"Marker pose updated and world saved to {output}")
            return 0
        else:
            logger.error("Failed to update marker pose")
            return 1

    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
