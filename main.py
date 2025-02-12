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
        
def save_prediction(self, dose_pred, patient_id, model_name):
        try:
            if dose_pred is None:
                logging.warning(f"Prediction for {model_name} and patient {patient_id} is None, skipping save.")
                return

            output_dir = os.path.join('results', f'{model_name}_prediction')
            
            # Add scaling for very low values
            max_dose = np.max(dose_pred)
            if 0 < max_dose <= 0.0001:
                scale_factor = 0.0001 / max_dose
                dose_pred = dose_pred * scale_factor
                logging.info(f"Scaled dose values by factor {scale_factor:.4f}")
            
            # Create sparse representation with lower threshold
            threshold = 0.00001  # Reduced threshold for non-zero values
            non_zero_mask = dose_pred > threshold
            if not np.any(non_zero_mask):
                raise ValueError(f"No dose values above threshold ({threshold})")

            dose_to_save = {
                'data': dose_pred[non_zero_mask],
                'indices': np.nonzero(dose_pred.flatten() > threshold)[0]
            }

            # Create DataFrame
            dose_df = pd.DataFrame(
                data=dose_to_save['data'],
                index=dose_to_save['indices'],
                columns=['data']
            )

            if dose_df.empty:
                raise ValueError("Created DataFrame is empty")

            # Save to CSV
            output_path = os.path.join(output_dir, f'{patient_id}.csv')
            dose_df.to_csv(output_path)

            # Verify saved file
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                if file_size == 0:
                    raise ValueError(f"Saved file is empty: {output_path}")

                saved_df = pd.read_csv(output_path, index_col=0)
                logging.info(f"Successfully saved prediction: {len(saved_df)} non-zero values")
                logging.info(f"Value range in saved file: min={saved_df['data'].min():.4f}, max={saved_df['data'].max():.4f}")
                return output_path
            else:
                raise FileNotFoundError(f"Failed to save prediction file: {output_path}")

        except Exception as e:
            logging.error(f"Error saving prediction for {model_name} and patient {patient_id}: {str(e)}")
            raise

def run_pipeline(self):
        if not self.active_models:
            logging.error("No models were successfully loaded. Pipeline cannot continue.")
            return

        try:
            # Validate data directory
            if not os.path.exists(self.data_dir):
                raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

            data_loader = DataLoader(
                get_paths(self.data_dir, ext=''),
                mode_name='dose_prediction'
            )

            number_of_batches = data_loader.number_of_batches()
            if number_of_batches == 0:
                logging.error(f"No files found in {self.data_dir}")
                return

            logging.info(f'Processing {number_of_batches} patients with {len(self.active_models)} models')

            for idx in tqdm.tqdm(range(number_of_batches)):
                try:
                    patient_batch = data_loader.get_batch(idx)
                    patient_id = patient_batch['patient_list'][0]
                    logging.info(f'Processing patient: {patient_id}')

                    for model_name in self.active_models:
                        try:
                            dose_pred = self.predict_single_case(self.models[model_name], patient_batch, model_name)
                            self.save_prediction(dose_pred, patient_id, model_name)
                        except Exception as e:
                            logging.error(f"Error processing patient {patient_id} with model {model_name}: {str(e)}")
                            continue

                except Exception as e:
                    logging.error(f"Error processing batch {idx}: {str(e)}")
                    continue

        except Exception as e:
            logging.error(f"Error in pipeline: {str(e)}")
