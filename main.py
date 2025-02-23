"""
OpenKBP Dose Prediction Pipeline
Created: 2025-02-23 12:04:34 UTC
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
from data_loader import DataLoader, normalize_dicom, get_paths, sparse_vector_function

# Record execution start time
START_TIME = datetime.datetime.strptime("2025-02-23 12:04:34", "%Y-%m-%d %H:%M:%S")
USER = "aagneye-syam"
print(f"Current Date and Time (UTC - YYYY-MM-DD HH:MM:SS formatted): {START_TIME.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Current User's Login: {USER}")

# Set environment variables for TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Define available models
AVAILABLE_MODELS = ['u_net', 'gan', 'dense_u_net', 'res_u_net']

class MultiModelDosePredictionPipeline:
    def __init__(self, data_dir, batch_size=1):
        """Initialize pipeline for OpenKBP dataset"""
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.patient_shape = MODEL_CONFIG['INPUT_SHAPE'][:3]  # Get (128, 128, 128)
        self.models = {}
        self.stats = {model: {'processed': 0, 'failed': 0} for model in AVAILABLE_MODELS}
        
        # Create results directories
        for model_name in AVAILABLE_MODELS:
            model_dir = os.path.join(PATH_CONFIG['OUTPUT_DIR'], model_name)
            os.makedirs(model_dir, exist_ok=True)
            print(f"Created/verified directory: {model_dir}")

        # Initialize data loader with correct dimensions
        self.data_loader = DataLoader(
            get_paths(self.data_dir, ext=''),
            mode_name='dose_prediction',
            batch_size=1,  # Force batch_size to 1
            patient_shape=self.patient_shape,
            shuffle=False  # Disable shuffling for consistent results
        )

        # Load models
        self.load_models()

    def load_models(self):
        """Load all available models"""
        model_paths = {
            'u_net': PATH_CONFIG['U_NET_PATH'],
            'gan': PATH_CONFIG['GAN_PATH'],
            'dense_u_net': PATH_CONFIG['DENSE_U_NET_PATH'],
            'res_u_net': PATH_CONFIG['RES_U_NET_PATH']
        }

        for model_name, model_path in model_paths.items():
            try:
                print(f"Loading {model_name} model...")
                if os.path.exists(model_path):
                    self.models[model_name] = load_model(model_path, compile=False)
                    print(f"Successfully loaded {model_name} model")
                else:
                    print(f"{model_name} model not found at {model_path}")
            except Exception as e:
                print(f"Error loading {model_name} model: {str(e)}")

        if not self.models:
            print("Warning: No models were successfully loaded")

    def predict_single_case(self, model_name, patient_data):
        """Predict dose for single case"""
        try:
            if model_name not in self.models:
                print(f"Model {model_name} not loaded")
                self.stats[model_name]['failed'] += 1
                return None

            # Get preprocessed data from patient_data
            ct = patient_data['ct']  # Shape: (128, 128, 128, 1)
            mask = patient_data['structure_masks']  # Shape: (128, 128, 128, 10)
            
            # Remove any extra dimensions and ensure proper shape
            ct = np.squeeze(ct)  # Remove extra dimensions if any
            mask = np.squeeze(mask)
            
            # Reshape to match model input requirements
            ct = ct.reshape(128, 128, 128, 1)
            mask = mask.reshape(128, 128, 128, 10)
            
            # Debug info
            print(f"CT shape after reshape: {ct.shape}")
            print(f"Mask shape after reshape: {mask.shape}")
            
            # Concatenate along the last axis
            input_data = np.concatenate([ct, mask], axis=-1)  # Shape: (128, 128, 128, 11)
            
            # Add batch dimension without extra dimensions
            input_data = np.expand_dims(input_data, axis=0)  # Shape: (1, 128, 128, 128, 11)
            
            print(f"Input tensor shape for {model_name}: {input_data.shape}")
            
            # Make prediction
            dose_pred = self.models[model_name].predict(input_data, verbose=0)
            print(f"Raw prediction shape: {dose_pred.shape}")
            
            # Remove any extra dimensions from prediction
            dose_pred = np.squeeze(dose_pred)
            
            # Apply possible dose mask
            dose_pred = dose_pred * np.squeeze(patient_data['possible_dose_mask'])
            
            # Scale to prescribed doses
            dose_pred = self.scale_to_prescribed_doses(dose_pred, np.squeeze(patient_data['structure_masks']))
            
            # Ensure correct shape before returning
            dose_pred = dose_pred.reshape(128, 128, 128)
            
            print(f"Final dose prediction shape: {dose_pred.shape}")
            return dose_pred

        except Exception as e:
            print(f"Error in prediction with {model_name}: {str(e)}")
            print(f"Debug info:")
            print(f"CT shape: {ct.shape if 'ct' in locals() else 'Not created'}")
            print(f"Mask shape: {mask.shape if 'mask' in locals() else 'Not created'}")
            print(f"Input shape: {input_data.shape if 'input_data' in locals() else 'Not created'}")
            self.stats[model_name]['failed'] += 1
            return None

    def scale_to_prescribed_doses(self, dose_pred, structure_masks):
        """Scale dose predictions to match prescribed doses"""
        try:
            # Ensure correct shapes
            dose_pred = np.squeeze(dose_pred)
            structure_masks = np.squeeze(structure_masks)
            
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
            
            # Use a default maximum dose of 80 Gy if MAX_DOSE is not in MODEL_CONFIG
            max_dose = MODEL_CONFIG.get('MAX_DOSE', 80)
            return np.clip(dose_pred, 0, max_dose)
            
        except Exception as e:
            print(f"Error in dose scaling: {str(e)}")
            print(f"Debug info:")
            print(f"Dose pred shape: {dose_pred.shape}")
            print(f"Structure masks shape: {structure_masks.shape}")
            return dose_pred

    def save_prediction(self, dose_pred, patient_id, model_name):
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
            
            output_path = os.path.join(PATH_CONFIG['OUTPUT_DIR'], 
                                     model_name, 
                                     f'{patient_id}_dose.csv')
            dose_df.to_csv(output_path)
            
            print(f"Saved {model_name} prediction to {output_path}")
            print(f"Non-zero points: {len(dose_sparse['data'])}")
            print(f"Dose range: [{dose_sparse['data'].min():.4f}, {dose_sparse['data'].max():.4f}] Gy")
            
            self.stats[model_name]['processed'] += 1
            return output_path
            
        except Exception as e:
            print(f"Error saving {model_name} prediction: {str(e)}")
            self.stats[model_name]['failed'] += 1
            return None

    def run_pipeline(self):
        """Run prediction pipeline"""
        if not self.models:
            print("No models loaded")
            return

        try:
            number_of_batches = self.data_loader.number_of_batches()
            if number_of_batches == 0:
                print("No batches to process")
                return

            print(f"\nStarting pipeline at {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
            
            # Process with all available models in order
            model_order = AVAILABLE_MODELS
            
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
                    for model_name in model_order:
                        if model_name in self.models:
                            print(f"Processing with model: {model_name}")
                            dose_pred = self.predict_single_case(model_name, patient_batch)
                            if dose_pred is not None:
                                self.save_prediction(dose_pred, patient_id, model_name)
                            
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
            # Print final statistics
            print("\nPrediction Statistics:")
            for model_name, stats in self.stats.items():
                print(f"\n{model_name}:")
                print(f"  Processed: {stats['processed']}")
                print(f"  Failed: {stats['failed']}")
                total = stats['processed'] + stats['failed']
                if total > 0:
                    success_rate = (stats['processed'] / total) * 100
                    print(f"  Success Rate: {success_rate:.2f}%")
                else:
                    print("  Success Rate: N/A (no predictions attempted)")
            
            end_time = datetime.datetime.utcnow()
            duration = end_time - START_TIME
            print(f"\nPipeline completed at {end_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
            print(f"Total execution time: {duration}")

def main():
    """Main execution function"""
    try:
        print("\nInitializing dose prediction pipeline...")

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