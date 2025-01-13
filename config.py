import os

# Model configuration
MODEL_CONFIG = {
    'INPUT_SHAPE': (128, 128, 128, 11),
    'BATCH_SIZE': 1,
    'HOUNSFIELD_MIN': -1024,
    'HOUNSFIELD_MAX': 1500,
    'HOUNSFIELD_RANGE': 1000
}

# Path configuration
PATH_CONFIG = {
    'MODEL_PATH': os.environ.get('MODEL_PATH', 'models/u_net_model/3D_UNet128_100epochs.h5'),
    'DATA_DIR': os.environ.get('DATA_DIR', 'open-kbp-master'),
    'OUTPUT_DIR': os.environ.get('OUTPUT_DIR', 'results/u_net_prediction'),
}

# ROI configuration
ROI_CONFIG = {
    'oars': ['Brainstem', 'SpinalCord', 'RightParotid', 'LeftParotid',
             'Esophagus', 'Larynx', 'Mandible'],
    'targets': ['PTV56', 'PTV63', 'PTV70']
}