from setuptools import find_packages, setup

package_name = 'traversability_pkg'

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
    maintainer='emilknarreborg',
    maintainer_email='emil.knarreborg@gmail.com',
    description='Traversability costmap nodes: LiDAR slope/roughness, semantic projection, and fused costmap.',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'semantic_lidar_debug_node = traversability_pkg.semantics_lidar_debug_node:main',
            'semantic_cost_node = traversability_pkg.semantic_cost_node:main',
            'ground_projection_node = traversability_pkg.ground_projection_node:main',
            'lidar_cost_node = traversability_pkg.lidar_cost_node:main',
            'fused_costmap_node = traversability_pkg.fused_costmap_node:main',
        ],
    },
)
