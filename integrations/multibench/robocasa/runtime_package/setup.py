"""Editable metadata for the unmodified pinned simulator source.

Upstream's MANIFEST recursively enumerates every mesh and texture even for an
editable install. This runtime-only packaging avoids copying/enumerating those
assets (kept at their official source paths) and excludes unused dataset tooling.
"""
from pathlib import Path
from setuptools import find_packages, setup

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "third_party/robocasa365"
setup(
    name="robocasa",
    version="1.0.1",
    description="Pinned unmodified RoboCasa365 simulator runtime (local editable packaging)",
    packages=find_packages(where=str(SOURCE), include=["robocasa", "robocasa.*"]),
    package_dir={"": str(SOURCE)},
    include_package_data=False,
    install_requires=["robosuite==1.5.2", "numpy==2.2.5", "numba==0.61.2",
                      "scipy==1.15.3", "mujoco==3.3.1", "pygame", "Pillow",
                      "opencv-python", "pyyaml", "pynput", "tqdm", "termcolor",
                      "imageio", "h5py", "lxml", "hidapi", "gymnasium"],
)
