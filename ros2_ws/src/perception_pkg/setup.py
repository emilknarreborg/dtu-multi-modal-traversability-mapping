from setuptools import find_packages, setup

package_name = 'perception_pkg'

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
    description='GSCNN semantic segmentation inference node and LiDAR-camera extrinsic calibration tool.',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'image_preprocessing_node = perception_pkg.image_preprocessing_node:main',
            'semantic_segmentation_node = perception_pkg.semantic_segmentation_node:main',
            'lidar_camera_calibration_node = perception_pkg.lidar_camera_calibration_node:main',
        ],
    },
)
