"""
source_demo_loader.py

Load and parse source demonstration data from HDF5 files.
Compatible with MimicGen datagen_info format.
"""

import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
import robosuite.utils.transform_utils as T


class SourceDemoLoader:
    """Loads source demonstrations from HDF5 file."""
    
    def __init__(self, demo_path):
        """
        Args:
            demo_path: Path to HDF5 demonstration file
        """
        self.demo_path = Path(demo_path)
        if not self.demo_path.exists():
            raise FileNotFoundError(f"Demo file not found: {demo_path}")
        
        self.demos = []
        self._load_demos()
    
    def _load_demos(self):
        """Load all demonstrations from HDF5 file."""
        with h5py.File(self.demo_path, 'r') as f:
            # Get all demo keys (e.g., "demo_0", "demo_1", ...)
            demo_keys = [k for k in f['data'].keys() if k.startswith('demo_')]
            
            print(f"Loading {len(demo_keys)} demonstrations from {self.demo_path}")
            
            for demo_key in tqdm(sorted(demo_keys), desc="Loading demos"):
                demo_group = f[f'data/{demo_key}']
                
                demo_data = {}
                
                # Check if this is MimicGen format (has datagen_info)
                if 'datagen_info' in demo_group:
                    # MimicGen format - read from datagen_info
                    datagen_info = demo_group['datagen_info']
                    
                    # Extract EEF poses (already in matrix or pose format)
                    demo_data['eef_poses'] = datagen_info['eef_pose'][:]
                    
                    # Extract target poses (used for trajectory following)
                    demo_data['target_poses'] = datagen_info['target_pose'][:]
                    
                    # Extract gripper actions
                    demo_data['gripper_actions'] = datagen_info['gripper_action'][:]
                    
                    # Extract object poses (dictionary of poses per object)
                    demo_data['object_poses'] = {}
                    if 'object_poses' in datagen_info:
                        for obj_name in datagen_info['object_poses'].keys():
                            demo_data['object_poses'][obj_name] = datagen_info['object_poses'][obj_name][:]
                    
                    # Extract subtask termination signals if available
                    demo_data['subtask_term_signals'] = {}
                    if 'subtask_term_signals' in datagen_info:
                        for signal_name in datagen_info['subtask_term_signals'].keys():
                            demo_data['subtask_term_signals'][signal_name] = datagen_info['subtask_term_signals'][signal_name][:]
                    
                    # Also load actions, rewards, dones if available
                    if 'actions' in demo_group:
                        demo_data['actions'] = demo_group['actions'][:]
                    if 'rewards' in demo_group:
                        demo_data['rewards'] = demo_group['rewards'][:]
                    if 'dones' in demo_group:
                        demo_data['dones'] = demo_group['dones'][:]
                    
                else:
                    # Standard robomimic format - read from obs and actions
                    demo_data['actions'] = demo_group['actions'][:]
                    demo_data['rewards'] = demo_group['rewards'][:] if 'rewards' in demo_group else None
                    demo_data['dones'] = demo_group['dones'][:] if 'dones' in demo_group else None
                    
                    # Extract observations
                    obs_group = demo_group['obs']
                    demo_data['observations'] = {}
                    
                    for obs_key in obs_group.keys():
                        demo_data['observations'][obs_key] = obs_group[obs_key][:]
                    
                    # Reconstruct EEF poses from observations
                    if 'robot0_eef_pos' in demo_data['observations']:
                        eef_pos = demo_data['observations']['robot0_eef_pos']
                        eef_quat = demo_data['observations'].get('robot0_eef_quat')
                        
                        if eef_quat is not None:
                            # Combine into 7D pose (x,y,z, qw,qx,qy,qz)
                            demo_data['eef_poses'] = np.concatenate([eef_pos, eef_quat], axis=1)
                        else:
                            demo_data['eef_poses'] = eef_pos
                    
                    # Extract object poses from observations
                    demo_data['object_poses'] = {}
                    for key in demo_data['observations'].keys():
                        if '_pos' in key and 'robot' not in key and 'gripper' not in key:
                            obj_name = key.replace('_pos', '')
                            obj_pos = demo_data['observations'][key]
                            obj_quat = demo_data['observations'].get(f'{obj_name}_quat')
                            
                            if obj_quat is not None:
                                # 7D pose format
                                demo_data['object_poses'][obj_name] = np.concatenate(
                                    [obj_pos, obj_quat], axis=1
                                )
                    
                    # Extract gripper actions (assume last action dimension)
                    demo_data['gripper_actions'] = demo_data['actions'][:, -1:]
                
                self.demos.append(demo_data)
            
            print(f"✓ Loaded {len(self.demos)} demonstrations")
            if len(self.demos) > 0:
                demo = self.demos[0]
                if 'actions' in demo and demo['actions'] is not None:
                    print(f"  Demo length: {len(demo['actions'])} steps")
                    print(f"  Action dim: {demo['actions'].shape[1]}")
                if 'eef_poses' in demo and demo['eef_poses'] is not None:
                    print(f"  EEF poses shape: {demo['eef_poses'].shape}")
                if 'target_poses' in demo and demo['target_poses'] is not None:
                    print(f"  Target poses shape: {demo['target_poses'].shape}")
                print(f"  Object poses: {list(demo['object_poses'].keys())}")
                if 'subtask_term_signals' in demo:
                    print(f"  Subtask signals: {list(demo['subtask_term_signals'].keys())}")
    
    def get_demo(self, index=0):
        """
        Get demonstration by index.
        
        Args:
            index: Demo index (default: 0)
            
        Returns:
            Dictionary containing demo data
        """
        if index >= len(self.demos):
            index = 0
        return self.demos[index]
    
    def get_trajectory_segment(self, demo_index=0, start_step=None, end_step=None):
        """
        Extract a segment of trajectory from demonstration.
        
        Args:
            demo_index: Index of demonstration
            start_step: Starting step (default: 0)
            end_step: Ending step (default: end of demo)
            
        Returns:
            Dictionary with trajectory segment data
        """
        demo = self.get_demo(demo_index)
        
        if start_step is None:
            start_step = 0
        if end_step is None:
            # Determine length from available data
            if 'actions' in demo and demo['actions'] is not None:
                end_step = len(demo['actions'])
            elif 'eef_poses' in demo and demo['eef_poses'] is not None:
                end_step = len(demo['eef_poses'])
            elif 'target_poses' in demo and demo['target_poses'] is not None:
                end_step = len(demo['target_poses'])
            else:
                end_step = 0
        
        segment = {
            'eef_poses': None,
            'target_poses': None,
            'gripper_actions': None,
            'object_poses': {},
            'subtask_term_signals': {}
        }
        
        # Extract EEF poses
        if 'eef_poses' in demo and demo['eef_poses'] is not None:
            segment['eef_poses'] = demo['eef_poses'][start_step:end_step]
        
        # Extract target poses (MimicGen format)
        if 'target_poses' in demo and demo['target_poses'] is not None:
            segment['target_poses'] = demo['target_poses'][start_step:end_step]
        
        # Extract gripper actions
        if 'gripper_actions' in demo and demo['gripper_actions'] is not None:
            segment['gripper_actions'] = demo['gripper_actions'][start_step:end_step]
        
        # Extract object poses
        if 'object_poses' in demo:
            for obj_name, poses in demo['object_poses'].items():
                segment['object_poses'][obj_name] = poses[start_step:end_step]
        
        # Extract subtask termination signals
        if 'subtask_term_signals' in demo:
            for signal_name, signals in demo['subtask_term_signals'].items():
                segment['subtask_term_signals'][signal_name] = signals[start_step:end_step]
        
        return segment
    
    @property
    def num_demos(self):
        """Number of loaded demonstrations."""
        return len(self.demos)
    
    def get_all_demo_keys(self):
        """
        Get list of all demo keys (compatible with MimicGen API).
        
        Returns:
            List of demo keys like ['demo_0', 'demo_1', ...]
        """
        return [f"demo_{i}" for i in range(len(self.demos))]
    
    def is_mimicgen_format(self, demo_index=0):
        """
        Check if the loaded demo is in MimicGen format (has datagen_info).
        
        Args:
            demo_index: Demo index to check
            
        Returns:
            True if MimicGen format, False otherwise
        """
        demo = self.get_demo(demo_index)
        # MimicGen format has target_poses and subtask_term_signals
        return 'target_poses' in demo and 'subtask_term_signals' in demo
