from setuptools import find_packages, setup

package_name = 'irn_g01'
share_dir = 'share/' + package_name

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        (share_dir, ['package.xml']),
        (share_dir + '/worlds', ['worlds/aruco_demo.sdf']),
        (share_dir + '/models/aruco_marker', ['models/aruco_marker/model.config', 'models/aruco_marker/model.sdf']),
        (share_dir + '/models/aruco_marker/materials/textures', ['models/aruco_marker/materials/textures/aruco_4x4_00.png']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='dacu',
    maintainer_email='daddek9@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
        ],
    },
)
