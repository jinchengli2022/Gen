"""
Robosuite Data Generation Framework

A modular framework for collecting robotic manipulation data from robosuite environments.
"""

__version__ = "0.1.0"
__author__ = "Gen VLA Team"

from configs.config import DataCollectionConfig
from env_interfaces.robosuite_env import RoboSuiteDataCollector
from utils.data_writer import create_data_writer, HDF5DataWriter, PickleDataWriter

__all__ = [
    "DataCollectionConfig",
    "RoboSuiteDataCollector",
    "create_data_writer",
    "HDF5DataWriter",
    "PickleDataWriter",
]
