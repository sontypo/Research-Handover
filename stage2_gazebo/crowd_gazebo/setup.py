from glob import glob
from setuptools import setup

package_name = "crowd_gazebo"
setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/worlds", glob("worlds/*.world")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    description="Gazebo crowd worlds and ground-truth people tracker",
    license="MIT",
    entry_points={"console_scripts": [
        "actor_tracker = crowd_gazebo.actor_tracker:main",
        "generate_worlds = crowd_gazebo.generate_worlds:main",
    ]},
)
