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
    'U_NET_PATH': os.path.join('models', 'u_net', 'unet.h5'),
    'DENSE_U_NET_PATH': os.path.join('models', 'dense_u_net', 'denseunet.h5'),
    'GAN_PATH': os.path.join('models', 'gan', 'gan.h5'),
    'RES_U_NET_PATH': os.path.join('models', 'res_u_net', 'resunet.h5'),
    'DATA_DIR': 'open-kbp-master',
    'OUTPUT_DIR': 'results',
}

# ROI configuration
ROI_CONFIG = {
    'oars': ['Brainstem', 'SpinalCord', 'RightParotid', 'LeftParotid',
             'Esophagus', 'Larynx', 'Mandible'],
    'targets': ['PTV56', 'PTV63', 'PTV70']
}