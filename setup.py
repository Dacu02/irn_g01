from setuptools import find_packages, setup

package_name = 'irn_g01'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/worlds', ['worlds/aruco_demo.sdf']),
        ('share/' + package_name + '/models/aruco_marker', ['models/aruco_marker/model.config', 'models/aruco_marker/model.sdf']),
        ('share/' + package_name + '/models/aruco_marker/materials/textures', ['models/aruco_marker/materials/textures/aruco_4x4_00.png']),
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
