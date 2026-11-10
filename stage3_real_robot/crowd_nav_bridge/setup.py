from glob import glob
from setuptools import setup

package_name = "crowd_nav_bridge"
setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    description="socialnav_core controllers on ROS 2 robots",
    license="MIT",
    entry_points={"console_scripts": [
        "nav_controller = crowd_nav_bridge.nav_controller:main",
        "people_adapter = crowd_nav_bridge.people_adapter:main",
        "trial_logger = crowd_nav_bridge.trial_logger:main",
    ]},
)
