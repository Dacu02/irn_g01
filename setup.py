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
            'aruco_reader = irn_g01.aruco_reader:main',
            'predictive_follower_FSM = irn_g01.predictive_follower_FSM:main',
            'compute_pose = irn_g01.compute_pose:main',
        ],
    },
)
