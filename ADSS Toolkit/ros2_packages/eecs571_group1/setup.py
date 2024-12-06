from setuptools import setup

package_name = 'eecs571_group1'
submodule1 = 'eecs571_group1/localmap_racing'
submodule2 = 'eecs571_group1/utils'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name,submodule1,submodule2],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='f1tenth',
    maintainer_email='f1tenth@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        	'control_node = eecs571_group1.control_node:main',
        ],
    },
)
