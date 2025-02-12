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
    
class MultiModelDosePredictionPipeline:
    def __init__(self, models_config, data_dir):
        self.models = {}
        self.data_dir = data_dir
        self.active_models = []

        for model_name, model_path in models_config.items():
            try:
                if not os.path.exists(model_path):
                    logging.error(f"Model file not found: {model_path}")
                    continue

                self.models[model_name] = self.load_model_with_custom_objects(model_path, model_name)
                self.active_models.append(model_name)
                
                output_dir = os.path.join('results', f'{model_name}_prediction')
                os.makedirs(output_dir, exist_ok=True)
                logging.info(f"Successfully loaded {model_name} and created output directory at {output_dir}")

            except Exception as e:
                logging.error(f"Error loading model {model_name}: {str(e)}")