"""
data_writer.py

Utilities for writing collected data to disk in various formats.
"""

import os
import h5py
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import json


class DataWriter:
    """Base class for data writers."""
    
    def __init__(self, output_dir: str, env_name: str):
        """
        Initialize data writer.
        
        Args:
            output_dir: Directory to save data
            env_name: Environment name
        """
        self.output_dir = Path(output_dir)
        self.env_name = env_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def write_episode(self, episode_data: Dict[str, Any], episode_idx: int):
        """Write a single episode to disk."""
        raise NotImplementedError
    
    def finalize(self):
        """Finalize writing (e.g., close files)."""
        pass


class HDF5DataWriter(DataWriter):
    """Writer for HDF5 format."""
    
    def __init__(self, output_dir: str, env_name: str):
        super().__init__(output_dir, env_name)
        self.hdf5_path = self.output_dir / f"{env_name}_data.hdf5"
        self.hdf5_file = h5py.File(self.hdf5_path, "w")
        self.episodes_written = 0
        
    def write_episode(self, episode_data: Dict[str, Any], episode_idx: int):
        """
        Write a single episode to HDF5 file.
        
        Args:
            episode_data: Dictionary containing episode data with keys:
                - observations: List of observation dicts
                - actions: List of action arrays
                - rewards: List of reward scalars
                - dones: List of done flags
                - info: Episode info dict
            episode_idx: Episode index
        """
        grp = self.hdf5_file.create_group(f"episode_{episode_idx}")
        
        # Write actions
        actions = np.array(episode_data["actions"])
        grp.create_dataset("actions", data=actions, compression="gzip")
        
        # Write rewards
        rewards = np.array(episode_data["rewards"])
        grp.create_dataset("rewards", data=rewards, compression="gzip")
        
        # Write dones
        dones = np.array(episode_data["dones"], dtype=bool)
        grp.create_dataset("dones", data=dones, compression="gzip")
        
        # Write observations
        obs_grp = grp.create_group("observations")
        
        # Get first observation to determine structure
        first_obs = episode_data["observations"][0]
        
        # Handle state observations
        if "state" in first_obs:
            states = np.array([obs["state"] for obs in episode_data["observations"]])
            obs_grp.create_dataset("state", data=states, compression="gzip")
        
        # Handle image observations
        for key in first_obs.keys():
            if key.endswith("_image"):
                images = np.array([obs[key] for obs in episode_data["observations"]])
                obs_grp.create_dataset(key, data=images, compression="gzip")
            elif key.endswith("_depth"):
                depths = np.array([obs[key] for obs in episode_data["observations"]])
                obs_grp.create_dataset(key, data=depths, compression="gzip")
        
        # Write metadata
        grp.attrs["episode_length"] = len(episode_data["actions"])
        grp.attrs["total_reward"] = np.sum(rewards)
        grp.attrs["success"] = episode_data.get("success", False)
        
        self.episodes_written += 1
        
    def finalize(self):
        """Close HDF5 file."""
        # Write global metadata
        self.hdf5_file.attrs["num_episodes"] = self.episodes_written
        self.hdf5_file.attrs["env_name"] = self.env_name
        
        self.hdf5_file.close()
        print(f"Saved {self.episodes_written} episodes to {self.hdf5_path}")


class PickleDataWriter(DataWriter):
    """Writer for pickle format (separate file per episode)."""
    
    def write_episode(self, episode_data: Dict[str, Any], episode_idx: int):
        """
        Write a single episode to pickle file.
        
        Args:
            episode_data: Episode data dictionary
            episode_idx: Episode index
        """
        episode_path = self.output_dir / f"{self.env_name}_ep{episode_idx:04d}.pkl"
        
        with open(episode_path, "wb") as f:
            pickle.dump(episode_data, f)
    
    def finalize(self):
        """Write metadata file."""
        metadata_path = self.output_dir / "metadata.json"
        metadata = {
            "env_name": self.env_name,
            "format": "pickle",
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)


def create_data_writer(output_dir: str, env_name: str, format: str = "hdf5") -> DataWriter:
    """
    Factory function to create appropriate data writer.
    
    Args:
        output_dir: Directory to save data
        env_name: Environment name
        format: Data format ("hdf5" or "pickle")
        
    Returns:
        DataWriter instance
    """
    if format == "hdf5":
        return HDF5DataWriter(output_dir, env_name)
    elif format == "pickle":
        return PickleDataWriter(output_dir, env_name)
    else:
        raise ValueError(f"Unknown format: {format}")
