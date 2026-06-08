#!/usr/bin/env python3
"""Script to generate and convert ArUco markers."""

import argparse
import logging
import sys
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from aruco_utils import generate_aruco_marker, convert_aruco_format

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Generate and convert ArUco markers'
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Generate command
    generate_parser = subparsers.add_parser('generate', help='Generate an ArUco marker')
    generate_parser.add_argument(
        '--id',
        type=int,
        default=0,
        help='ArUco marker ID (default: 0)'
    )
    generate_parser.add_argument(
        '--size',
        type=int,
        default=200,
        help='Marker size in pixels (default: 200)'
    )
    generate_parser.add_argument(
        '--format',
        type=str,
        default='png',
        choices=['png', 'jpg', 'bmp'],
        help='Output format (default: png)'
    )
    generate_parser.add_argument(
        '--output',
        type=str,
        help='Output file path'
    )

    # Convert command
    convert_parser = subparsers.add_parser('convert', help='Convert ArUco marker format')
    convert_parser.add_argument(
        'input',
        type=str,
        help='Input file path'
    )
    convert_parser.add_argument(
        '--format',
        type=str,
        default='png',
        choices=['png', 'jpg', 'bmp', 'svg'],
        help='Output format (default: png)'
    )
    convert_parser.add_argument(
        '--output',
        type=str,
        help='Output file path'
    )

    args = parser.parse_args()

    if args.command == 'generate':
        logger.info(f"Generating ArUco marker ID {args.id}")
        result = generate_aruco_marker(
            aruco_id=args.id,
            marker_size=args.size,
            format_type=args.format,
            output_path=args.output
        )
        if result is not None:
            logger.info("ArUco marker generated successfully")
            return 0
        else:
            logger.error("Failed to generate ArUco marker")
            return 1

    elif args.command == 'convert':
        logger.info(f"Converting ArUco marker: {args.input} -> {args.format}")
        success = convert_aruco_format(
            input_path=args.input,
            output_format=args.format,
            output_path=args.output
        )
        if success:
            logger.info("ArUco marker converted successfully")
            return 0
        else:
            logger.error("Failed to convert ArUco marker")
            return 1

    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
