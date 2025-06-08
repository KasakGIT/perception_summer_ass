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
        ('share/' + package_name + '/launch', ['launch/perception.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kasak',
    maintainer_email='kasak14112005@gmai.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'cone_detector = perception_pkg.cone_detector:main',
        ],
    },
)
