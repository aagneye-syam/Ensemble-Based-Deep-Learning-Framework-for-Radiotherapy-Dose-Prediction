import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from data_loader import DataLoader, normalize_dicom, get_paths
from tensorflow.keras.layers import concatenate
import pandas as pd
import tqdm
from config import PATH_CONFIG
import logging
from datetime import datetime

# Set up logging
current_time = "2025-02-07 05:37:38"
current_user = "aagneye-syam"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - User: ' + current_user + ' - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Disable MKL optimizations to avoid conv operation errors
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        return inputs

    def compute_output_shape(self, input_shape):
        return input_shape