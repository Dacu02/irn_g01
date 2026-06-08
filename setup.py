from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'irn_g01'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'worlds'),
            glob(os.path.join(package_name, 'worlds', '*.world'))),
        (os.path.join('share', package_name, 'models', 'aruco_marker'),
            glob(os.path.join(package_name, 'models', 'aruco_marker', 'model.sdf'))),
        (os.path.join('share', package_name, 'launch'),
            glob(os.path.join(package_name, 'launch', '*.py'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='dacu',
    maintainer_email='daddek9@gmail.com',
    description='ArUco marker display package for Gazebo Ignition',
    license='Apache License 2.0',
    extras_require={
        'test': [
            'pytest',
        ],
        'dev': [
            'opencv-python',
            'Pillow',
        ],
    },
    entry_points={
        'console_scripts': [
        ],
    },
)
