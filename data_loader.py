"""
OpenKBP Data Loader
Created: 2025-02-22 15:40:24 UTC
Author: aagneye-syam
"""

import os
import pathlib
import numpy as np
import pandas as pd
from config import MODEL_CONFIG, ROI_CONFIG, MEMORY_CONFIG

def normalize_dicom(img):
    """
    Normalize DICOM image using 12-bit format (0-4095) as specified in dataset.
    Clips CT values between 0 and 4095 to convert mixed 12/16-bit to standard 12-bit.
    """
    img = img.copy()
    img = np.clip(img, MODEL_CONFIG['HOUNSFIELD_MIN'], MODEL_CONFIG['HOUNSFIELD_MAX'])
    img = img / MODEL_CONFIG['HOUNSFIELD_RANGE']
    return img.astype(np.float32)

def get_paths(directory_path, ext=''):
    """Get file paths from directory"""
    if not os.path.isdir(directory_path):
        print(f"Directory not found: {directory_path}")
        return []

    all_paths = []
    if ext == '':
        dir_list = os.listdir(directory_path)
        for path in dir_list:
            if '.' != path[0]:  # Ignore hidden files
                all_paths.append(os.path.join(directory_path, str(path)))
    else:
        data_root = pathlib.Path(directory_path)
        for path in data_root.glob(f'*.{ext}'):
            all_paths.append(str(path))

    return sorted(all_paths)

class DataLoader:
    """Data loader for OpenKBP dataset - handles sparse matrix format"""
    
    def __init__(self, file_paths_list, batch_size=1, patient_shape=(128, 128, 128),
                 shuffle=True, mode_name='dose_prediction'):
        """
        Initialize DataLoader
        Args:
            file_paths_list: List of paths to patient directories
            batch_size: Number of patients to load at once
            patient_shape: Shape of patient tensor (128x128x128)
            shuffle: Whether to shuffle data between epochs
            mode_name: Type of data loading (dose_prediction, training_model, evaluation)
        """
        self.rois = ROI_CONFIG
        self.batch_size = batch_size
        self.patient_shape = patient_shape
        self.indices = np.arange(len(file_paths_list))
        self.file_paths_list = file_paths_list
        self.shuffle = shuffle
        self.full_roi_list = sum(map(list, self.rois.values()), [])
        self.num_rois = len(self.full_roi_list)
        
        # Create patient ID list
        self.patient_id_list = []
        for path in self.file_paths_list:
            try:
                if 'test-pats' in path:
                    patient_id = os.path.basename(path)
                    self.patient_id_list.append(patient_id)
            except Exception as e:
                print(f"Warning: Could not parse patient ID from path: {path}")

        # Set data loading mode
        self.mode_name = mode_name
        self.set_mode(self.mode_name)

    def set_mode(self, mode_name):
        """Set data loading mode and required file shapes"""
        self.mode_name = mode_name

        if mode_name == 'dose_prediction':
            self.required_files = {
                'ct': (self.patient_shape + (1,)),
                'structure_masks': (self.patient_shape + (self.num_rois,)),
                'possible_dose_mask': (self.patient_shape + (1,)),
                'voxel_dimensions': (3,)
            }
        elif mode_name == 'training_model':
            self.required_files = {
                'dose': (self.patient_shape + (1,)),
                'ct': (self.patient_shape + (1,)),
                'structure_masks': (self.patient_shape + (self.num_rois,)),
                'possible_dose_mask': (self.patient_shape + (1,)),
                'voxel_dimensions': (3,)
            }
        elif mode_name == 'evaluation':
            self.required_files = {
                'dose': (self.patient_shape + (1,)),
                'structure_masks': (self.patient_shape + (self.num_rois,)),
                'possible_dose_mask': (self.patient_shape + (1,)),
                'voxel_dimensions': (3,)
            }
        else:
            raise ValueError(f"Unsupported mode: {mode_name}")

    def load_file(self, file_name):
        """Load sparse matrix format CSV file"""
        try:
            if not os.path.exists(file_name):
                print(f"File not found: {file_name}")
                return None
                
            # Load CSV file
            loaded_file_df = pd.read_csv(file_name, index_col=0)
            if loaded_file_df.empty:
                print(f"Empty file: {file_name}")
                return None
                
            # Handle different file types
            if 'voxel_dimensions.csv' in file_name:
                return np.array(loaded_file_df.index, dtype=np.float32)
            elif loaded_file_df.isnull().values.any():
                return np.array(loaded_file_df.index, dtype=np.int32).squeeze()
            else:
                return {
                    'indices': np.array(loaded_file_df.index, dtype=np.int32).squeeze(),
                    'data': np.array(loaded_file_df['data'], dtype=np.float32).squeeze()
                }
                
        except Exception as e:
            print(f"Error loading file {file_name}: {str(e)}")
            return None

    def load_and_shape_data(self, path_to_load):
        """Load sparse data and reshape to dense 128x128x128 tensors"""
        try:
            loaded_file = {}
            files_to_load = get_paths(path_to_load, ext='csv')
            
            if not files_to_load:
                print(f"No CSV files found in: {path_to_load}")
                return None

            # Initialize tensors with float32
            total_voxels = np.prod(self.patient_shape)
            shaped_data = {}.fromkeys(self.required_files)
            for key in shaped_data:
                shaped_data[key] = np.zeros(self.required_files[key], dtype=np.float32)

            # Load all files
            for f in files_to_load:
                f_name = os.path.basename(f).split('.')[0]
                if f_name in self.required_files or f_name in self.full_roi_list:
                    loaded_file[f_name] = self.load_file(f)

            # Process each type of data
            for key in shaped_data:
                try:
                    if key == 'structure_masks':
                        # Process in chunks to save memory
                        chunk_size = MEMORY_CONFIG['CHUNK_SIZE']
                        for roi_idx, roi in enumerate(self.full_roi_list):
                            if roi in loaded_file:
                                indices = loaded_file[roi]
                                if isinstance(indices, np.ndarray):
                                    for i in range(0, len(indices), chunk_size):
                                        chunk_indices = indices[i:i+chunk_size]
                                        valid_indices = chunk_indices[chunk_indices < total_voxels]
                                        if len(valid_indices) > 0:
                                            shaped_data[key].reshape(-1, self.num_rois)[valid_indices, roi_idx] = 1
                        shaped_data[key] = shaped_data[key].reshape(*self.patient_shape, self.num_rois)

                    elif key == 'possible_dose_mask':
                        if key in loaded_file:
                            indices = loaded_file[key]
                            if isinstance(indices, np.ndarray):
                                for i in range(0, len(indices), MEMORY_CONFIG['CHUNK_SIZE']):
                                    chunk_indices = indices[i:i+MEMORY_CONFIG['CHUNK_SIZE']]
                                    valid_indices = chunk_indices[chunk_indices < total_voxels]
                                    if len(valid_indices) > 0:
                                        shaped_data[key].ravel()[valid_indices] = 1
                        shaped_data[key] = shaped_data[key].reshape(*self.patient_shape, 1)

                    elif key == 'voxel_dimensions':
                        if key in loaded_file:
                            shaped_data[key][:] = loaded_file[key].astype(np.float32)
                        else:
                            shaped_data[key][:] = np.array([3.5, 3.5, 2.0], dtype=np.float32)

                    elif key == 'ct':
                        if key in loaded_file and isinstance(loaded_file[key], dict):
                            indices = loaded_file[key]['indices']
                            data = loaded_file[key]['data']
                            for i in range(0, len(indices), MEMORY_CONFIG['CHUNK_SIZE']):
                                chunk_indices = indices[i:i+MEMORY_CONFIG['CHUNK_SIZE']]
                                chunk_data = data[i:i+MEMORY_CONFIG['CHUNK_SIZE']]
                                valid_mask = chunk_indices < total_voxels
                                if np.any(valid_mask):
                                    shaped_data[key].ravel()[chunk_indices[valid_mask]] = chunk_data[valid_mask]
                        shaped_data[key] = shaped_data[key].reshape(*self.patient_shape, 1)
                        shaped_data[key] = np.clip(shaped_data[key], 
                                                 MODEL_CONFIG['HOUNSFIELD_MIN'],
                                                 MODEL_CONFIG['HOUNSFIELD_MAX'])

                    print(f"Processed {key} - Shape: {shaped_data[key].shape}, "
                          f"Range: [{shaped_data[key].min():.2f}, {shaped_data[key].max():.2f}], "
                          f"Memory: {shaped_data[key].nbytes / (1024**2):.1f} MB")
                
                except Exception as e:
                    print(f"Error processing key {key}: {str(e)}")
                    continue

            return shaped_data

        except Exception as e:
            print(f"Error in load_and_shape_data: {str(e)}")
            return None

    def get_batch(self, index=None, patient_list=None):
        """Get a batch of patient data"""
        try:
            if patient_list is None:
                indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
            else:
                indices = self.patient_to_index(patient_list)

            file_paths_to_load = [self.file_paths_list[k] for k in indices]
            return self.load_data(file_paths_to_load)
        except Exception as e:
            print(f"Error in get_batch: {str(e)}")
            return None

    def load_data(self, file_paths_to_load):
        """Load data for multiple patients"""
        try:
            tf_data = {}.fromkeys(self.required_files)
            patient_list = []
            patient_path_list = []

            # Initialize arrays with float32
            for key in tf_data:
                shape = (self.batch_size,) + self.required_files[key]
                tf_data[key] = np.zeros(shape, dtype=np.float32)

            for i, pat_path in enumerate(file_paths_to_load):
                try:
                    patient_path_list.append(pat_path)
                    pat_id = os.path.basename(pat_path)
                    patient_list.append(pat_id)

                    loaded_data = self.load_and_shape_data(pat_path)
                    if loaded_data is not None:
                        for key in tf_data:
                            if key in loaded_data:
                                tf_data[key][i] = loaded_data[key]
                except Exception as e:
                    print(f"Error loading patient {pat_id}: {str(e)}")
                    continue

            tf_data['patient_list'] = patient_list
            tf_data['patient_path_list'] = patient_path_list
            return tf_data

        except Exception as e:
            print(f"Error in load_data: {str(e)}")
            return None

    def number_of_batches(self):
        """Get number of batches in dataset"""
        return int(np.floor(len(self.file_paths_list) / self.batch_size))

    def patient_to_index(self, patient_list):
        """Convert patient IDs to indices"""
        try:
            un_shuffled_indices = [self.patient_id_list.index(k) for k in patient_list]
            shuffled_indices = [self.indices[k] for k in un_shuffled_indices]
            return shuffled_indices
        except Exception as e:
            print(f"Error in patient_to_index: {str(e)}")
            return []

    def on_epoch_end(self):
        """Shuffle indices at the end of epoch if enabled"""
        if self.shuffle:
            np.random.shuffle(self.indices)