"""
OpenKBP Dose Prediction Pipeline
Created: 2025-02-23 06:09:12 UTC
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
START_TIME = datetime.datetime.strptime("2025-02-23 06:09:12", "%Y-%m-%d %H:%M:%S")
USER = "aagneye-syam"
print(f"Execution started at {START_TIME.strftime('%Y-%m-%d %H:%M:%S')} UTC by {USER}")

# Set environment variables for TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def sparse_vector_function(x):
    """Convert dense tensor to sparse format"""
    try:
        x_flat = x.flatten(order='C')
        non_zero_mask = x_flat > 0
        return {
            'data': x_flat[non_zero_mask],
            'indices': np.nonzero(x_flat)[0]
        }
    except Exception as e:
        print(f"Error in sparse vector conversion: {str(e)}")
        return None

class MultiModelDosePredictionPipeline:
    def __init__(self, model_path, data_dir, batch_size=1):
        """Initialize pipeline for OpenKBP dataset"""
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.patient_shape = (128, 128, 128)
        
        # Create results directory
        os.makedirs('results/u_net', exist_ok=True)

        # Initialize data loader
        self.data_loader = DataLoader(
            get_paths(self.data_dir, ext=''),
            mode_name='dose_prediction',
            batch_size=self.batch_size,
            patient_shape=self.patient_shape
        )

        # Load U-Net model
        try:
            print("Loading U-Net model...")
            self.model = load_model(model_path, compile=False)
            print("Successfully loaded U-Net model")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            self.model = None

    def predict_single_case(self, patient_data):
        """Predict dose for single case"""
        try:
            # Prepare input data exactly as in training
            ct = normalize_dicom(patient_data['ct'])
            mask = patient_data['structure_masks']
            
            # Create input tensor (same as training)
            ct = ct.reshape(128, 128, 128, 1)
            mask = mask.reshape(128, 128, 128, 10)
            input_data = np.concatenate([ct, mask], axis=-1)
            input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension
            
            # Make prediction
            dose_pred = self.model.predict(input_data, verbose=0)
            
            # Apply possible dose mask
            dose_pred = dose_pred * patient_data['possible_dose_mask']
            
            # Scale to prescribed doses
            dose_pred = self.scale_to_prescribed_doses(dose_pred[0], patient_data['structure_masks'][0])
            
            return dose_pred

        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return None

    def scale_to_prescribed_doses(self, dose_pred, structure_masks):
        """Scale dose predictions to match prescribed doses"""
        try:
            prescribed_doses = TREATMENT_CONFIG['PRESCRIBED_DOSES']
            ptv_indices = {ptv: idx for idx, ptv in enumerate(ROI_CONFIG['targets'])}
            
            for ptv, prescribed_dose in prescribed_doses.items():
                if ptv in ptv_indices:
                    ptv_mask = structure_masks[..., ptv_indices[ptv]]
                    ptv_dose = dose_pred * ptv_mask
                    if np.sum(ptv_mask) > 0:
                        current_mean = np.mean(ptv_dose[ptv_mask > 0])
                        if current_mean > 0:
                            scale_factor = prescribed_dose / current_mean
                            dose_pred = np.where(ptv_mask > 0, dose_pred * scale_factor, dose_pred)
            
            # Clip unrealistic values
            dose_pred = np.clip(dose_pred, 0, 80)  # Max realistic dose is 80 Gy
            return dose_pred
            
        except Exception as e:
            print(f"Error in dose scaling: {str(e)}")
            return dose_pred

    def save_prediction(self, dose_pred, patient_id):
        """Save prediction in sparse matrix format"""
        try:
            patient_id = os.path.basename(patient_id)
            
            dose_sparse = sparse_vector_function(dose_pred)
            if dose_sparse is None or len(dose_sparse['data']) == 0:
                print(f"Warning: No non-zero dose points for {patient_id}")
                return None

            dose_df = pd.DataFrame({
                'data': dose_sparse['data']
            }, index=dose_sparse['indices'])
            
            output_path = os.path.join('results', 'u_net', f'{patient_id}_dose.csv')
            dose_df.to_csv(output_path)
            
            print(f"Saved prediction to {output_path}")
            print(f"Non-zero points: {len(dose_sparse['data'])}")
            print(f"Dose range: [{dose_sparse['data'].min():.4f}, {dose_sparse['data'].max():.4f}] Gy")
            
            return output_path
            
        except Exception as e:
            print(f"Error saving prediction: {str(e)}")
            return None

    def run_pipeline(self):
        """Run prediction pipeline"""
        if self.model is None:
            print("No model loaded")
            return

        try:
            number_of_batches = self.data_loader.number_of_batches()
            if number_of_batches == 0:
                print("No batches to process")
                return

            print(f"\nStarting pipeline at {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
            
            for idx in tqdm(range(number_of_batches), desc="Processing patients"):
                try:
                    # Get patient batch
                    patient_batch = self.data_loader.get_batch(idx)
                    if patient_batch is None or 'patient_list' not in patient_batch:
                        print(f"Invalid batch data for index {idx}")
                        continue

                    patient_id = patient_batch['patient_list'][0]
                    print(f"\nProcessing patient: {patient_id}")
                    
                    # Make prediction
                    dose_pred = self.predict_single_case(patient_batch)
                    if dose_pred is not None:
                        self.save_prediction(dose_pred, patient_id)
                    
                    # Clear memory
                    if MEMORY_CONFIG['CLEAR_SESSION']:
                        tf.keras.backend.clear_session()
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
        
        # Get U-Net model path
        model_path = PATH_CONFIG['U_NET_PATH']
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

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
            model_path, 
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