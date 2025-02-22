import os
import pathlib
import numpy as np
import pandas as pd
import tempfile
from config import MODEL_CONFIG, ROI_CONFIG

def normalize_dicom(img):
    """Normalize DICOM image using Hounsfield units"""
    try:
        HOUNSFIELD_MIN = MODEL_CONFIG['HOUNSFIELD_MIN']
        HOUNSFIELD_MAX = MODEL_CONFIG['HOUNSFIELD_MAX']
        HOUNSFIELD_RANGE = MODEL_CONFIG['HOUNSFIELD_RANGE']

        img = np.clip(img, HOUNSFIELD_MIN, HOUNSFIELD_MAX)
        img = img.astype(np.float32) / HOUNSFIELD_RANGE
        return img
    except Exception as e:
        print(f"Error in normalize_dicom: {str(e)}")
        return None

def get_paths(directory_path, ext=''):
    """Get paths of files in directory"""
    try:
        if not os.path.isdir(directory_path):
            print(f"Directory not found: {directory_path}")
            return []

        all_image_paths = []
        if ext == '':
            dir_list = os.listdir(directory_path)
            for iPath in dir_list:
                if '.' != iPath[0]:  # Ignore hidden files
                    all_image_paths.append(os.path.join(directory_path, str(iPath)))
        else:
            data_root = pathlib.Path(directory_path)
            for iPath in data_root.glob(f'*.{ext}'):
                all_image_paths.append(str(iPath))

        return all_image_paths
    except Exception as e:
        print(f"Error in get_paths: {str(e)}")
        return []

def load_file(file_name):
    """Load file in OpenKBP dataset format"""
    try:
        loaded_file_df = pd.read_csv(file_name, index_col=0)

        if 'voxel_dimensions.csv' in file_name:
            loaded_file = np.loadtxt(file_name, dtype=np.float32)
        elif loaded_file_df.isnull().values.any():
            loaded_file = np.array(loaded_file_df.index, dtype=np.int32).squeeze()
        else:
            loaded_file = {
                'indices': np.array(loaded_file_df.index, dtype=np.int32).squeeze(),
                'data': np.array(loaded_file_df['data'], dtype=np.float32).squeeze()
            }

        return loaded_file
    except Exception as e:
        print(f"Error loading file {file_name}: {str(e)}")
        return None

class DataLoader:
    """Memory-efficient data loader for OpenKBP dataset"""
    
    def __init__(self, file_paths_list, batch_size=1, patient_shape=(128, 128, 128),
                 shuffle=True, mode_name='dose_prediction', use_memmap=True,
                 chunk_size=1000000):
        """
        Initialize the DataLoader
        
        Args:
            file_paths_list: List of file paths to load
            batch_size: Number of patients per batch
            patient_shape: Shape of patient data (default: (128,128,128))
            shuffle: Whether to shuffle data
            mode_name: Mode for data loading
            use_memmap: Whether to use memory mapping
            chunk_size: Size of chunks for large file loading
        """
        # Memory management settings
        self.use_memmap = use_memmap
        self.temp_dir = tempfile.mkdtemp() if use_memmap else None
        self.chunk_size = chunk_size

        # Update paths if needed
        if any('provided-data' in path for path in file_paths_list):
            base_dir = os.path.dirname(file_paths_list[0])
            test_pats_dir = os.path.join(base_dir, 'provided-data', 'test-pats')
            if os.path.exists(test_pats_dir):
                print(f"Using test patients directory: {test_pats_dir}")
                file_paths_list = get_paths(test_pats_dir)

        # Basic configuration
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
                else:
                    print(f"Skipping non-patient directory: {path}")
            except Exception as e:
                print(f"Warning: Could not parse patient ID from path: {path}")

        # Set mode
        self.mode_name = mode_name
        self.set_mode(self.mode_name)

    def create_memmap_array(self, shape, dtype=np.float32):
        """Create a memory-mapped array"""
        try:
            if self.use_memmap:
                filename = os.path.join(self.temp_dir, f'temp_{np.random.randint(0, 1000000)}.npy')
                return np.memmap(filename, dtype=dtype, mode='w+', shape=shape)
            return np.zeros(shape, dtype=dtype)
        except Exception as e:
            print(f"Error creating memmap array: {str(e)}")
            return np.zeros(shape, dtype=dtype)

    def set_mode(self, mode_name):
        """Set the mode for data loading"""
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

    def get_batch(self, index=None, patient_list=None):
        """Get a batch of data"""
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
        """Load data for given file paths"""
        try:
            tf_data = {}.fromkeys(self.required_files)
            patient_list = []
            patient_path_list = []

            # Initialize arrays
            for key in tf_data:
                shape = (self.batch_size, *self.required_files[key])
                tf_data[key] = self.create_memmap_array(shape)

            # Load data for each patient
            for i, pat_path in enumerate(file_paths_to_load):
                try:
                    patient_path_list.append(pat_path)
                    pat_id = pat_path.split('/')[-1].split('.')[0]
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

    def load_and_shape_data(self, path_to_load):
        """Load and shape data from files"""
        try:
            loaded_file = {}
            files_to_load = get_paths(path_to_load, ext='')

            print(f"Loading files from: {path_to_load}")
            print(f"Files found: {files_to_load}")

            for f in files_to_load:
                f_name = f.split('/')[-1].split('.')[0]
                if f_name in self.required_files or f_name in self.full_roi_list:
                    loaded_data = self.load_file_in_chunks(f) if os.path.getsize(f) > self.chunk_size else load_file(f)
                    if loaded_data is not None:
                        loaded_file[f_name] = loaded_data

            shaped_data = {}.fromkeys(self.required_files)
            for key in shaped_data:
                shaped_data[key] = self.create_memmap_array(self.required_files[key])

            for key in shaped_data:
                try:
                    if key == 'structure_masks':
                        for roi_idx, roi in enumerate(self.full_roi_list):
                            if roi in loaded_file:
                                np.put(shaped_data[key], self.num_rois * loaded_file[roi] + roi_idx, 1)
                    elif key == 'possible_dose_mask':
                        if key in loaded_file:
                            np.put(shaped_data[key], loaded_file[key], 1)
                    elif key == 'voxel_dimensions':
                        if key in loaded_file:
                            shaped_data[key][:] = loaded_file[key]
                        else:
                            shaped_data[key][:] = np.array([3.906, 3.906, 2.5], dtype=np.float32)
                    else:
                        if key in loaded_file:
                            np.put(shaped_data[key], loaded_file[key]['indices'], loaded_file[key]['data'])
                except Exception as e:
                    print(f"Error processing key {key}: {str(e)}")
                    continue

            return shaped_data

        except Exception as e:
            print(f"Error in load_and_shape_data: {str(e)}")
            return None

    def load_file_in_chunks(self, filename):
        """Load large files in chunks"""
        try:
            chunks = pd.read_csv(filename, index_col=0, chunksize=self.chunk_size)
            return pd.concat(chunks)
        except Exception as e:
            print(f"Error loading file in chunks {filename}: {str(e)}")
            return None

    def number_of_batches(self):
        """Get number of batches in epoch"""
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

    def cleanup(self):
        """Clean up temporary memory-mapped files"""
        if self.use_memmap and self.temp_dir:
            try:
                import shutil
                shutil.rmtree(self.temp_dir)
            except Exception as e:
                print(f"Error cleaning up temporary files: {str(e)}")

    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()