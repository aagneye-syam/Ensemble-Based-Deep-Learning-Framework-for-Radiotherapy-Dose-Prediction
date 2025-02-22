import os
import pathlib
import numpy as np
import pandas as pd
from config import MODEL_CONFIG, ROI_CONFIG

def normalize_dicom(img):
    """Normalize DICOM image using Hounsfield units"""
    HOUNSFIELD_MIN = MODEL_CONFIG['HOUNSFIELD_MIN']
    HOUNSFIELD_MAX = MODEL_CONFIG['HOUNSFIELD_MAX']
    HOUNSFIELD_RANGE = MODEL_CONFIG['HOUNSFIELD_RANGE']

    img = np.clip(img, HOUNSFIELD_MIN, HOUNSFIELD_MAX)
    img = img / HOUNSFIELD_RANGE
    return img

def get_paths(directory_path, ext=''):
    """Get paths of files in directory"""
    if not os.path.isdir(directory_path):
        return []

    all_image_paths = []
    if ext == '':
        dir_list = os.listdir(directory_path)
        for iPath in dir_list:
            if '.' != iPath[0]:  # Ignore hidden files
                all_image_paths.append(f'{directory_path}/{str(iPath)}')
    else:
        data_root = pathlib.Path(directory_path)
        for iPath in data_root.glob(f'*.{ext}'):
            all_image_paths.append(str(iPath))

    return all_image_paths

def load_file(file_name):
    """Load file in OpenKBP dataset format"""
    loaded_file_df = pd.read_csv(file_name, index_col=0)

    if 'voxel_dimensions.csv' in file_name:
        loaded_file = np.loadtxt(file_name)
    elif loaded_file_df.isnull().values.any():
        loaded_file = np.array(loaded_file_df.index).squeeze()
    else:
        loaded_file = {
            'indices': np.array(loaded_file_df.index).squeeze(),
            'data': np.array(loaded_file_df['data']).squeeze()
        }

    return loaded_file

class DataLoader:
    """Data loader for OpenKBP dataset"""
    def __init__(self, file_paths_list, batch_size=1, patient_shape=(128, 128, 128),
                 shuffle=True, mode_name='dose_prediction'):
        # If the path is to the root directory, update it to point to test-pats
        if any('provided-data' in path for path in file_paths_list):
            base_dir = os.path.dirname(file_paths_list[0])
            test_pats_dir = os.path.join(base_dir, 'provided-data', 'test-pats')
            if os.path.exists(test_pats_dir):
                print(f"Using test patients directory: {test_pats_dir}")
                file_paths_list = get_paths(test_pats_dir)

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
                    # Extract patient ID from the directory name
                    patient_id = os.path.basename(path)
                    self.patient_id_list.append(patient_id)
                else:
                    print(f"Skipping non-patient directory: {path}")
            except Exception as e:
                print(f"Warning: Could not parse patient ID from path: {path}")

        self.mode_name = mode_name
        self.set_mode(self.mode_name)

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
        else:
            raise ValueError(f"Unsupported mode: {mode_name}")

    def get_batch(self, index=None, patient_list=None):
        """Get a batch of data"""
        if patient_list is None:
            indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        else:
            indices = self.patient_to_index(patient_list)

        file_paths_to_load = [self.file_paths_list[k] for k in indices]
        return self.load_data(file_paths_to_load)

    def load_data(self, file_paths_to_load):
        """Load data for given file paths"""
        tf_data = {}.fromkeys(self.required_files)
        patient_list = []
        patient_path_list = []
    
        for key in tf_data:
            tf_data[key] = np.zeros((self.batch_size, *self.required_files[key]))

        for i, pat_path in enumerate(file_paths_to_load):
            patient_path_list.append(pat_path)
            pat_id = pat_path.split('/')[-1].split('.')[0]
            patient_list.append(pat_id)

            loaded_data_dict = self.load_and_shape_data(pat_path)
            for key in tf_data:
                tf_data[key][i,] = loaded_data_dict[key]

        tf_data['patient_list'] = patient_list
        tf_data['patient_path_list'] = patient_path_list
        return tf_data

    def load_and_shape_data(self, path_to_load):
        """Load and shape data from files"""
        loaded_file = {}
        files_to_load = get_paths(path_to_load, ext='')

        # Debug print
        print(f"Loading files from: {path_to_load}")
        print(f"Files found: {files_to_load}")

        for f in files_to_load:
            f_name = f.split('/')[-1].split('.')[0]
            if f_name in self.required_files or f_name in self.full_roi_list:
                try:
                    loaded_file[f_name] = load_file(f)
                except Exception as e:
                    print(f"Error loading file {f}: {str(e)}")

        shaped_data = {}.fromkeys(self.required_files)
        for key in shaped_data:
            shaped_data[key] = np.zeros(self.required_files[key])

        for key in shaped_data:
            if key == 'structure_masks':
                for roi_idx, roi in enumerate(self.full_roi_list):
                    if roi in loaded_file:
                        np.put(shaped_data[key], self.num_rois * loaded_file[roi] + roi_idx, 1)
            elif key == 'possible_dose_mask':
                if key in loaded_file:
                    np.put(shaped_data[key], loaded_file[key], 1)
            elif key == 'voxel_dimensions':
                if key in loaded_file:
                    shaped_data[key] = loaded_file[key]
                else:
                    print(f"Warning: voxel_dimensions not found. Using default values.")
                    shaped_data[key] = np.array([3.906, 3.906, 2.5])  # Default voxel size in mm
            else:
                if key in loaded_file:
                    np.put(shaped_data[key], loaded_file[key]['indices'], loaded_file[key]['data'])
                else:
                    print(f"Warning: {key} not found in loaded files")

        return shaped_data

    def number_of_batches(self):
        """Get number of batches in epoch"""
        return int(np.floor(len(self.file_paths_list) / self.batch_size))

    def patient_to_index(self, patient_list):
        """Convert patient IDs to indices"""
        un_shuffled_indices = [self.patient_id_list.index(k) for k in patient_list]
        shuffled_indices = [self.indices[k] for k in un_shuffled_indices]
        return shuffled_indices