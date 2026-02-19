"""
setup.py

Installation configuration for the robosuite data generation framework.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="robosuite-data-gen",
    version="0.1.0",
    author="Gen VLA Team",
    description="A modular framework for collecting robotic manipulation data from robosuite",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(include=[
        "configs",
        "env_interfaces",
        "env",
        "utils",
        "scripts",
    ]),
    python_requires=">=3.8",
    install_requires=[
        "robosuite>=1.5.0",
        "numpy>=1.21.0",
        "h5py>=3.7.0",
        "tqdm>=4.64.0",
        "opencv-python>=4.6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
        "viz": [
            "matplotlib>=3.5.0",
            "pillow>=9.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "collect-data=scripts.simple_collect:main",
            "collect-pouring=scripts.collect_pouring:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    keywords="robotics manipulation data-collection robosuite reinforcement-learning",
)
