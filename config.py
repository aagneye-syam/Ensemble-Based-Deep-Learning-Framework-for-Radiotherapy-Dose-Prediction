import os

# Model configuration
MODEL_CONFIG = {
    'INPUT_SHAPE': (128, 128, 128, 11),
    'BATCH_SIZE': 1,
    'HOUNSFIELD_MIN': -1024,
    'HOUNSFIELD_MAX': 1500,
    'HOUNSFIELD_RANGE': 1000
}