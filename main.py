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


        def predict_single_case(self, model, patient_data, model_name):
        try:
            logging.info(f"Starting prediction for patient using {model_name}")
            
            if 'ct' not in patient_data or patient_data['ct'] is None:
                raise ValueError("CT data is missing or None")

            voxel_dimensions = patient_data.get('voxel_dimensions')
            if voxel_dimensions is None:
                logging.warning("Voxel dimensions not found, using default values")
                voxel_dimensions = np.array([1.0, 1.0, 1.0])
            else:
                logging.info(f"Using voxel dimensions: {voxel_dimensions}")

            ct_normalized = normalize_dicom(patient_data['ct'], voxel_dimensions)
            logging.info(f"{model_name} CT normalized shape: {ct_normalized.shape}")
            
            # Check CT data range after normalization
            ct_min = np.min(ct_normalized)
            ct_max = np.max(ct_normalized)
            logging.info(f"CT value range after normalization: min={ct_min:.4f}, max={ct_max:.4f}")

            input_data = concatenate([
                ct_normalized,
                patient_data['structure_masks']
            ], axis=-1)
            logging.info(f"{model_name} input data shape: {input_data.shape}")

            if input_data.shape[1:] != model.input_shape[1:]:
                raise ValueError(f"Input shape mismatch. Expected {model.input_shape}, got {input_data.shape}")

            dose_pred = model.predict(input_data, verbose=0)
            logging.info(f"{model_name} prediction shape: {dose_pred.shape}")
            
            pred_min = np.min(dose_pred)
            pred_max = np.max(dose_pred)
            logging.info(f"Raw prediction stats: min={pred_min:.4f}, max={pred_max:.4f}")

            if pred_max <= 0.0001:
                raise ValueError(f"Prediction values too low: max={pred_max:.4f}")

            if 'possible_dose_mask' in patient_data:
                dose_pred = dose_pred * patient_data['possible_dose_mask']
                logging.info(f"Applied dose mask. New range: min={np.min(dose_pred):.4f}, max={np.max(dose_pred):.4f}")

            return dose_pred.squeeze()

        except Exception as e:
            logging.error(f"Error in prediction with {model_name}: {str(e)}")
            return None