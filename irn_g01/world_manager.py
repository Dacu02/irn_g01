"""World manager for loading and modifying Gazebo Ignition worlds."""

import os
import logging
from typing import Optional, Dict, Any, List
from xml.etree import ElementTree as ET
from pathlib import Path

logger = logging.getLogger(__name__)


class WorldManager:
    """Manage Gazebo world files and modifications."""

    def __init__(self, world_file_path: str):
        """
        Initialize the world manager.

        Args:
            world_file_path: Path to the world file
        """
        self.world_file_path = world_file_path
        self.tree = None
        self.root = None
        self._load_world()

    def _load_world(self) -> bool:
        """Load the world file from disk."""
        if not os.path.exists(self.world_file_path):
            logger.error(f"World file not found: {self.world_file_path}")
            return False

        try:
            self.tree = ET.parse(self.world_file_path)
            self.root = self.tree.getroot()
            logger.info(f"World file loaded: {self.world_file_path}")
            return True
        except ET.ParseError as e:
            logger.error(f"Failed to parse world file: {e}")
            return False

    def add_aruco_marker(
        self,
        marker_id: int,
        position: tuple = (0, 0, 0),
        rotation: tuple = (0, 0, 0),
        marker_name: Optional[str] = None,
        uri: str = "model://aruco_marker"
    ) -> bool:
        """
        Add an ArUco marker to the world.

        Args:
            marker_id: Unique ID for the marker
            position: (x, y, z) position tuple
            rotation: (roll, pitch, yaw) rotation tuple
            marker_name: Optional custom name for the marker
            uri: URI of the marker model

        Returns:
            True if marker was added successfully, False otherwise
        """
        if self.root is None:
            logger.error("World file not loaded")
            return False

        try:
            # Find the world element
            world = self.root.find('world')
            if world is None:
                logger.error("No world element found in the SDF file")
                return False

            # Create marker name
            if marker_name is None:
                marker_name = f"aruco_marker_{marker_id}"

            # Check if marker already exists
            existing_marker = world.find(f".//model[@name='{marker_name}']")
            if existing_marker is not None:
                logger.warning(f"Marker {marker_name} already exists, replacing it")
                world.remove(existing_marker)

            # Create the marker model element
            marker_element = ET.Element('model', name=marker_name)
            pose_str = f"{position[0]} {position[1]} {position[2]} {rotation[0]} {rotation[1]} {rotation[2]}"
            pose_element = ET.SubElement(marker_element, 'pose')
            pose_element.text = pose_str

            # Add include element
            include_element = ET.SubElement(marker_element, 'include')
            uri_element = ET.SubElement(include_element, 'uri')
            uri_element.text = uri

            # Add the marker to the world
            world.append(marker_element)

            logger.info(f"Added ArUco marker: {marker_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to add ArUco marker: {e}")
            return False

    def remove_marker(self, marker_name: str) -> bool:
        """
        Remove a marker from the world.

        Args:
            marker_name: Name of the marker to remove

        Returns:
            True if marker was removed, False otherwise
        """
        if self.root is None:
            logger.error("World file not loaded")
            return False

        try:
            world = self.root.find('world')
            if world is None:
                logger.error("No world element found")
                return False

            marker = world.find(f".//model[@name='{marker_name}']")
            if marker is not None:
                world.remove(marker)
                logger.info(f"Removed marker: {marker_name}")
                return True
            else:
                logger.warning(f"Marker not found: {marker_name}")
                return False

        except Exception as e:
            logger.error(f"Failed to remove marker: {e}")
            return False

    def list_markers(self) -> List[str]:
        """
        List all markers in the world.

        Returns:
            List of marker names
        """
        if self.root is None:
            logger.error("World file not loaded")
            return []

        try:
            world = self.root.find('world')
            if world is None:
                logger.error("No world element found")
                return []

            markers = []
            for model in world.findall(".//model"):
                marker_name = model.get('name')
                if marker_name and 'aruco' in marker_name.lower():
                    markers.append(marker_name)

            return markers

        except Exception as e:
            logger.error(f"Failed to list markers: {e}")
            return []

    def update_marker_pose(
        self,
        marker_name: str,
        position: Optional[tuple] = None,
        rotation: Optional[tuple] = None
    ) -> bool:
        """
        Update the pose of a marker.

        Args:
            marker_name: Name of the marker
            position: Optional (x, y, z) position tuple
            rotation: Optional (roll, pitch, yaw) rotation tuple

        Returns:
            True if pose was updated, False otherwise
        """
        if self.root is None:
            logger.error("World file not loaded")
            return False

        try:
            world = self.root.find('world')
            if world is None:
                logger.error("No world element found")
                return False

            marker = world.find(f".//model[@name='{marker_name}']")
            if marker is None:
                logger.error(f"Marker not found: {marker_name}")
                return False

            pose_element = marker.find('pose')
            if pose_element is None:
                pose_element = ET.SubElement(marker, 'pose')

            # Get current pose
            if position is None or rotation is None:
                current_pose = pose_element.text.split() if pose_element.text else [0]*6
                if position is None:
                    position = tuple(map(float, current_pose[:3]))
                if rotation is None:
                    rotation = tuple(map(float, current_pose[3:6]))

            pose_str = f"{position[0]} {position[1]} {position[2]} {rotation[0]} {rotation[1]} {rotation[2]}"
            pose_element.text = pose_str

            logger.info(f"Updated pose for marker: {marker_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to update marker pose: {e}")
            return False

    def save_world(self, output_path: Optional[str] = None) -> bool:
        """
        Save the world to a file.

        Args:
            output_path: Optional path to save the world. If not provided, uses the original path.

        Returns:
            True if save was successful, False otherwise
        """
        if self.tree is None:
            logger.error("No world file loaded")
            return False

        output_path = output_path or self.world_file_path

        try:
            # Register namespaces to preserve formatting
            ET.register_namespace('', 'http://sdformat.org/schemas/sdformat-1.10.xsd')

            # Write the tree
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            self.tree.write(
                output_path,
                encoding='UTF-8',
                xml_declaration=True
            )

            logger.info(f"World saved to: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save world: {e}")
            return False

    def get_world_content(self) -> Optional[str]:
        """Get the current world content as a string."""
        try:
            return ET.tostring(self.root, encoding='unicode')
        except Exception as e:
            logger.error(f"Failed to get world content: {e}")
            return None
