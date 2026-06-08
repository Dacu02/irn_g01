"""Utility functions for ArUco marker handling."""

import os
import logging
from pathlib import Path
from typing import Optional, Tuple
from contextlib import suppress

logger = logging.getLogger(__name__)

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logger.warning("NumPy not available. Some ArUco functions may not work.")

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    logger.warning("OpenCV not available. Some ArUco functions may not work.")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("Pillow not available. PNG conversion may not work.")


def generate_aruco_marker(
    aruco_id: int = 0,
    marker_size: int = 200,
    format_type: str = "png",
    output_path: Optional[str] = None
) -> Optional:
    """
    Generate an ArUco marker image.

    Args:
        aruco_id: The ArUco marker ID (0-249 for 4x4 grid, 0-999 for 5x5 grid, etc.)
        marker_size: Size of the marker in pixels
        format_type: Output format ('png', 'jpg', 'bmp')
        output_path: Optional path to save the marker image

    Returns:
        The marker image as a numpy array, or None if generation failed
    """
    if not OPENCV_AVAILABLE or not NUMPY_AVAILABLE:
        logger.error("OpenCV and NumPy are required to generate ArUco markers")
        return None

    try:
        # Get the ArUco dictionary (5x5 grid with 250 markers)
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)

        # Generate the marker image
        marker_image = cv2.aruco.generateImageMarker(
            dictionary=aruco_dict,
            markerSize=aruco_id,
            sidePixels=marker_size,
            borderBits=1
        )

        # Save to file if output_path is provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, marker_image)
            logger.info(f"ArUco marker saved to {output_path}")

        return marker_image

    except Exception as e:
        logger.error(f"Failed to generate ArUco marker: {e}")
        return None


def convert_aruco_format(
    input_path: str,
    output_format: str = "png",
    output_path: Optional[str] = None
) -> bool:
    """
    Convert ArUco marker between different formats.

    Args:
        input_path: Path to the input ArUco marker image
        output_format: Target format ('png', 'jpg', 'bmp', 'svg')
        output_path: Path to save the converted image

    Returns:
        True if conversion was successful, False otherwise
    """
    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        return False

    try:
        if output_format == "svg":
            return _convert_to_svg(input_path, output_path)
        elif output_format in ["png", "jpg", "jpeg", "bmp"]:
            return _convert_image_format(input_path, output_format, output_path)
        else:
            logger.error(f"Unsupported output format: {output_format}")
            return False

    except Exception as e:
        logger.error(f"Failed to convert ArUco marker: {e}")
        return False


def _convert_image_format(
    input_path: str,
    output_format: str,
    output_path: Optional[str]
) -> bool:
    """Convert image between standard formats using PIL."""
    if not PIL_AVAILABLE:
        logger.error("Pillow is required for image format conversion")
        return False

    try:
        img = Image.open(input_path)
        if output_path is None:
            base_path = os.path.splitext(input_path)[0]
            output_path = f"{base_path}.{output_format}"

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img.save(output_path)
        logger.info(f"Image converted to {output_format}: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Image format conversion failed: {e}")
        return False


def _convert_to_svg(
    input_path: str,
    output_path: Optional[str]
) -> bool:
    """Convert ArUco marker to SVG format."""
    if not PIL_AVAILABLE or not OPENCV_AVAILABLE or not NUMPY_AVAILABLE:
        logger.error("PIL, OpenCV, and NumPy are required for SVG conversion")
        return False

    try:
        # Read the image
        img = Image.open(input_path)
        img_array = np.array(img)

        if output_path is None:
            base_path = os.path.splitext(input_path)[0]
            output_path = f"{base_path}.svg"

        # Convert image to binary (black and white)
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array

        # Threshold to create binary image
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # Create SVG
        height, width = binary.shape
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w') as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">\n')
            f.write(f'<rect width="{width}" height="{height}" fill="white"/>\n')

            # Convert pixels to rectangles
            for y in range(height):
                for x in range(width):
                    if binary[y, x] == 0:  # Black pixel
                        f.write(f'<rect x="{x}" y="{y}" width="1" height="1" fill="black"/>\n')

            f.write('</svg>\n')

        logger.info(f"ArUco marker converted to SVG: {output_path}")
        return True

    except Exception as e:
        logger.error(f"SVG conversion failed: {e}")
        return False


def get_package_share_directory() -> str:
    """Get the share directory for the irn_g01 package."""
    try:
        from ament_index_python.packages import get_package_share_directory
        return get_package_share_directory('irn_g01')
    except ImportError:
        # Fallback to relative path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(os.path.dirname(current_dir), 'share', 'irn_g01')
