import os

# Model configuration
MODEL_CONFIG = {
    'INPUT_SHAPE': (128, 128, 128, 11),  # CT + 10 structure masks (7 OARs + 3 PTVs)
    'BATCH_SIZE': 1,
    'HOUNSFIELD_MIN': -1024,   # As per original code
    'HOUNSFIELD_MAX': 1500,    # As per original code
    'HOUNSFIELD_RANGE': 1000,  # As per original code
    'NUM_FRACTIONS': 35        # 35 fractions for dose delivery
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
    'oars': [
        'Brainstem',    # Critical OAR
        'SpinalCord',   # Critical OAR
        'RightParotid', # Salivary gland
        'LeftParotid',  # Salivary gland
        'Esophagus',    # Digestive tract
        'Larynx',       # Voice box
        'Mandible'      # Lower jaw bone
    ],
    'targets': [
        'PTV56',  # Elective target volumes (56 Gy)
        'PTV63',  # Intermediate-risk target volumes (63 Gy)
        'PTV70'   # Gross disease target (70 Gy)
    ]
}

# Treatment configuration
TREATMENT_CONFIG = {
    'PRESCRIBED_DOSES': {
        'PTV70': 70.0,  # Gy in 35 fractions
        'PTV63': 63.0,  # Gy in 35 fractions
        'PTV56': 56.0   # Gy in 35 fractions
    },
    'BEAM_CONFIG': {
        'NUM_BEAMS': 9,     # 9 equispaced coplanar fields
        'ANGLES': [0, 40, 80, 120, 160, 200, 240, 280, 320],  # Beam angles
        'ENERGY': '6MV',    # Beam energy
        'DELIVERY': 'IMRT'  # Delivery technique
    }
}

# Memory configuration
MEMORY_CONFIG = {
    'USE_FLOAT32': True,     # Use float32 instead of float64
    'CHUNK_SIZE': 32,        # Size of chunks for processing
    'MAX_MEMORY_GB': 1.0,    # Maximum memory usage in GB
    'CLEAR_SESSION': True    # Clear TensorFlow session between predictions
}