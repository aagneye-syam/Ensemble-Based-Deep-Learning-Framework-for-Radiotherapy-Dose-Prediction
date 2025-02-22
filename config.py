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
    'U_NET_PATH': os.environ.get('U_NET_PATH', 'models/u_net/unet.h5'),
    'ATTENTION_U_NET_PATH': os.environ.get('ATTENTION_U_NET_PATH', 'models/attention_u_net/attnunet.h5'),
    'DENSE_U_NET_PATH': os.environ.get('DENSE_U_NET_PATH', 'models/dense_u_net/denseunet.h5'),
    'GAN_PATH': os.environ.get('GAN_PATH', 'models/gan/gan.h5'),
    'RES_U_NET_PATH': os.environ.get('RES_U_NET_PATH', 'models/res_u_net/resunet.h5'),
    'DATA_DIR': os.environ.get('DATA_DIR', 'open-kbp-master'),
    'OUTPUT_DIR': os.environ.get('OUTPUT_DIR', 'results'),
    'TRUE_DOSE_DIR': os.environ.get('TRUE_DOSE_DIR', 'open-kbp-master/provided-data/true-doses'),
}

# ROI configuration
ROI_CONFIG = {
    'oars': ['Brainstem', 'SpinalCord', 'RightParotid', 'LeftParotid',
             'Esophagus', 'Larynx', 'Mandible'],
    'targets': ['PTV56', 'PTV63', 'PTV70']
}