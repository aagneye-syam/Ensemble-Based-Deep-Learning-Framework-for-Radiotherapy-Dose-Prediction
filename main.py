"""
OpenKBP Dose Prediction Pipeline
Created: 2025-02-22 15:42:53 UTC
Author: aagneye-syam
"""

import os
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import concatenate
from tqdm import tqdm
import gc
from config import PATH_CONFIG, MODEL_CONFIG, TREATMENT_CONFIG, ROI_CONFIG, MEMORY_CONFIG
from data_loader import DataLoader, normalize_dicom, get_paths

# Record execution start time
START_TIME = datetime.datetime.strptime("2025-02-22 15:42:53", "%Y-%m-%d %H:%M:%S")
USER = "aagneye-syam"
print(f"Execution started at {START_TIME.strftime('%Y-%m-%d %H:%M:%S')} UTC by {USER}")

# Set environment variables for TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF logging

def sparse_vector_function(x):
    """
    Convert dense tensor to sparse format as specified in OpenKBP dataset.
    Returns indices and data for non-zero elements in C-order (row-major).
    Format: CSV with index column and 'data' column.
    """
    try:
        # Process in chunks to save memory
        chunk_size = MEMORY_CONFIG['CHUNK_SIZE']
        x_flat = x.flatten(order='C')
        non_zero_indices = []
        non_zero_data = []
        
        for i in range(0, len(x_flat), chunk_size):
            chunk = x_flat[i:i + chunk_size]
            mask = chunk > 0
            if np.any(mask):
                non_zero_indices.extend(i + np.nonzero(mask)[0])
                non_zero_data.extend(chunk[mask])
        
        return {
            'data': np.array(non_zero_data, dtype=np.float32),
            'indices': np.array(non_zero_indices, dtype=np.int32)
        }
    except Exception as e:
        print(f"Error in sparse vector conversion: {str(e)}")
        return None

class MultiModelDosePredictionPipeline:
    def __init__(self, models_config, data_dir, batch_size=1):
        """
        Initialize pipeline for OpenKBP dataset processing
        Dataset: 128x128x128 voxel tensors with:
        - CT images (12-bit format)
        - Structure masks for OARs and PTVs
        - Possible dose mask
        - Voxel dimensions (~3.5mm x 3.5mm x 2mm)
        """
        self.models = {}
        self.data_dir = data_dir
        self.active_models = []
        self.batch_size = batch_size
        self.patient_shape = (128, 128, 128)
        
        # Create results directory
        os.makedirs('results', exist_ok=True)

        # Initialize data loader
        self.data_loader = DataLoader(
            get_paths(self.data_dir, ext=''),
            mode_name='dose_prediction',
            batch_size=self.batch_size,
            patient_shape=self.patient_shape
        )

        # Load models
        for model_name, model_path in models_config.items():
            try:
                if not os.path.exists(model_path):
                    print(f"Model path not found: {model_path}")
                    continue

                print(f"Loading model: {model_name}")
                model = load_model(model_path, compile=False)
                if model is not None:
                    self.models[model_name] = model
                    self.active_models.append(model_name)
                    os.makedirs(f'results/{model_name}', exist_ok=True)
                    print(f"Successfully loaded model: {model_name}")

            except Exception as e:
                print(f"Error loading model {model_name}: {str(e)}")
                continue

    def validate_data(self, patient_data):
        """Validate data according to OpenKBP specifications"""
        try:
            # Check if data exists
            if not patient_data or not isinstance(patient_data, dict):
                print("Invalid patient data format")
                return False
                
            required_keys = ['ct', 'structure_masks', 'possible_dose_mask']
            for key in required_keys:
                if key not in patient_data:
                    print(f"Missing required key: {key}")
                    return False
                if patient_data[key] is None:
                    print(f"Data is None for key: {key}")
                    return False
                
            # Check CT data
            ct_data = patient_data['ct']
            if not isinstance(ct_data, np.ndarray):
                print("CT data is not a numpy array")
                return False
            
            # Check shapes
            expected_shape = (1,) + self.patient_shape
            if ct_data.shape[:-1] != expected_shape:
                print(f"CT data shape mismatch: {ct_data.shape}, expected: {expected_shape + (1,)}")
                return False
                
            return True
            
        except Exception as e:
            print(f"Error in data validation: {str(e)}")
            return False

    def predict_single_case(self, model, patient_data):
        """Memory-efficient prediction"""
        try:
            if not self.validate_data(patient_data):
                print("Data validation failed")
                return None

            # Process in smaller chunks
            chunk_size = MEMORY_CONFIG['CHUNK_SIZE']
            ct_normalized = normalize_dicom(patient_data['ct'])
            masks = patient_data['structure_masks']
            
            # Initialize output array
            output_shape = self.patient_shape + (1,)
            dose_pred = np.zeros(output_shape, dtype=np.float32)
            
            # Process in chunks
            for z in range(0, self.patient_shape[0], chunk_size):
                z_end = min(z + chunk_size, self.patient_shape[0])
                
                # Prepare chunk data
                ct_chunk = ct_normalized[0, z:z_end, :, :, :]
                masks_chunk = masks[0, z:z_end, :, :, :]
                
                # Create input tensor
                input_chunk = np.concatenate([ct_chunk, masks_chunk], axis=-1)
                input_chunk = np.expand_dims(input_chunk, axis=0)
                
                # Predict chunk
                with tf.device('/CPU:0'):
                    pred_chunk = model.predict(input_chunk, verbose=0)
                
                # Store prediction
                dose_pred[z:z_end, :, :, 0] = pred_chunk[0, :, :, :, 0]
            
            # Apply possible dose mask
            if patient_data['possible_dose_mask'] is not None:
                dose_pred = dose_pred * patient_data['possible_dose_mask'][0]
            
            # Scale predictions to prescribed doses
            dose_pred = self.scale_to_prescribed_doses(dose_pred, patient_data['structure_masks'][0])
            
            return dose_pred

        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return None

    def scale_to_prescribed_doses(self, dose_pred, structure_masks):
        """Scale dose predictions to match prescribed doses for PTVs"""
        try:
            prescribed_doses = TREATMENT_CONFIG['PRESCRIBED_DOSES']
            
            # Get PTV indices from ROI_CONFIG
            ptv_indices = {ptv: idx for idx, ptv in enumerate(ROI_CONFIG['targets'])}
            
            # Scale each PTV region to its prescribed dose
            for ptv, prescribed_dose in prescribed_doses.items():
                if ptv in ptv_indices:
                    ptv_mask = structure_masks[..., ptv_indices[ptv]]
                    ptv_dose = dose_pred * ptv_mask
                    if np.sum(ptv_mask) > 0:
                        current_mean = np.mean(ptv_dose[ptv_mask > 0])
                        if current_mean > 0:
                            scale_factor = prescribed_dose / current_mean
                            dose_pred = np.where(ptv_mask > 0, dose_pred * scale_factor, dose_pred)
            
            return dose_pred
            
        except Exception as e:
            print(f"Error in dose scaling: {str(e)}")
            return dose_pred

    def save_prediction(self, dose_pred, patient_id, model_name):
        """Save prediction in OpenKBP sparse matrix format"""
        try:
            # Extract patient ID
            patient_id = os.path.basename(patient_id)
            
            # Create output directory
            output_dir = os.path.join('results', model_name)
            os.makedirs(output_dir, exist_ok=True)
            
            # Convert to sparse format (C-order as required)
            dose_sparse = sparse_vector_function(dose_pred)
            if dose_sparse is None or len(dose_sparse['data']) == 0:
                print(f"Warning: No non-zero dose points found for {patient_id}")
                return None

            # Create DataFrame in required format (index, data)
            dose_df = pd.DataFrame({
                'data': dose_sparse['data']
            }, index=dose_sparse['indices'])
            
            # Save to CSV
            output_path = os.path.join(output_dir, f'{patient_id}_dose.csv')
            dose_df.to_csv(output_path)
            
            print(f"Saved prediction to {output_path}")
            print(f"Number of non-zero dose points: {len(dose_sparse['data'])}")
            print(f"Dose range: [{dose_sparse['data'].min():.4f}, {dose_sparse['data'].max():.4f}] Gy")
            
            return output_path
            
        except Exception as e:
            print(f"Error saving prediction: {str(e)}")
            return None

    def run_pipeline(self):
        """Run the complete prediction pipeline"""
        if not self.active_models:
            print("No active models found")
            return

        try:
            number_of_batches = self.data_loader.number_of_batches()
            if number_of_batches == 0:
                print("No batches to process")
                return

            print(f"\nStarting pipeline execution at {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
            
            for idx in tqdm(range(number_of_batches), desc="Processing patients"):
                try:
                    # Get patient batch
                    patient_batch = self.data_loader.get_batch(idx)
                    if patient_batch is None or 'patient_list' not in patient_batch:
                        print(f"Invalid batch data for index {idx}")
                        continue

                    patient_id = patient_batch['patient_list'][0]
                    print(f"\nProcessing patient: {patient_id}")
                    
                    # Process with each model
                    for model_name in self.active_models:
                        try:
                            print(f"Processing with model: {model_name}")
                            
                            # Clear session to free memory
                            if MEMORY_CONFIG['CLEAR_SESSION']:
                                tf.keras.backend.clear_session()
                            
                            model = self.models[model_name]
                            dose_pred = self.predict_single_case(model, patient_batch)
                            if dose_pred is not None:
                                self.save_prediction(dose_pred, patient_id, model_name)
                            
                        except Exception as e:
                            print(f"Error processing model {model_name}: {str(e)}")
                            continue

                    # Clear memory
                    gc.collect()

                except Exception as e:
                    print(f"Error processing batch {idx}: {str(e)}")
                    continue

        except Exception as e:
            print(f"Pipeline error: {str(e)}")
            raise

        finally:
            end_time = datetime.datetime.utcnow()
            duration = end_time - START_TIME
            print(f"\nPipeline completed at {end_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
            print(f"Total execution time: {duration}")

def main():
    """Main execution function"""
    try:
        print("\nInitializing dose prediction pipeline...")
        
        # Configure models
        models_config = {
            'u_net': PATH_CONFIG['U_NET_PATH'],
            'dense_u_net': PATH_CONFIG['DENSE_U_NET_PATH'],
            'gan': PATH_CONFIG['GAN_PATH'],
            'res_u_net': PATH_CONFIG['RES_U_NET_PATH']
        }

        # Configure data directory
        data_dir = PATH_CONFIG['DATA_DIR']
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory not found at {data_dir}")

        # Use test patients directory if available
        test_pats_dir = os.path.join(data_dir, 'provided-data', 'test-pats')
        if os.path.exists(test_pats_dir):
            data_dir = test_pats_dir
            print(f"Using test patients directory: {test_pats_dir}")

        # Initialize and run pipeline
        pipeline = MultiModelDosePredictionPipeline(
            models_config, 
            data_dir,
            batch_size=MODEL_CONFIG['BATCH_SIZE']
        )
        
        print("Starting pipeline execution...")
        pipeline.run_pipeline()

    except Exception as e:
        print(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()