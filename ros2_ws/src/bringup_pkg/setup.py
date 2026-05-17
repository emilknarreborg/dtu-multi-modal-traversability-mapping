from setuptools import find_packages, setup
from glob import glob

package_name = 'bringup_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/rviz', glob('rviz/*.rviz')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/config/sensors', glob('config/sensors/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='emilknarreborg',
    maintainer_email='emil.knarreborg@gmail.com',
    description='Launch files, sensor configurations, and RViz config for the traversability pipeline.',
    license='Apache-2.0',
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
