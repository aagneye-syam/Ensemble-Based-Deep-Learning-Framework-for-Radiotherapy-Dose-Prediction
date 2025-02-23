"""
OpenKBP Dose Prediction Pipeline
Created: 2025-02-23 06:45:13 UTC
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

# Set environment variables
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Configuration
MODEL_CONFIG = {
    'INPUT_SHAPE': (128, 128, 128, 11),
    'OUTPUT_SHAPE': (128, 128, 128, 1),
    'MAX_DOSE': 80.0,
    'BATCH_SIZE': 1
}

PATH_CONFIG = {
    'BASE_DIR': 'open-kbp-master',
    'DATA_DIR': os.path.join('open-kbp-master', 'provided-data', 'test-pats'),
    'RESULTS_DIR': 'results',
    'MODEL_PATHS': {
        'u_net': os.path.join('models', 'u_net.h5'),
        'gan': os.path.join('models', 'gan.h5'),
        'res_u_net': os.path.join('models', 'res_u_net.h5')
    }
}

ROI_LIST = [
    'Brainstem', 'SpinalCord', 'RightParotid', 'LeftParotid',
    'Esophagus', 'Larynx', 'Mandible', 'PTV56', 'PTV63', 'PTV70'
]

DOSE_PRESCRIPTION = {
    'PTV70': 70.0,
    'PTV63': 63.0,
    'PTV56': 56.0
}

def normalize_dicom(img):
    """Normalize CT images"""
    HOUNSFIELD_MIN = -1024
    HOUNSFIELD_MAX = 1500
    HOUNSFIELD_RANGE = 1000
    
    img = np.clip(img, HOUNSFIELD_MIN, HOUNSFIELD_MAX)
    img = img / HOUNSFIELD_RANGE
    return img.astype(np.float32)

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

class DosePredictionPipeline:
    def __init__(self, data_dir):
        """Initialize pipeline"""
        self.data_dir = data_dir
        self.models = {}
        self.stats = {model: {'processed': 0, 'failed': 0, 'skipped': 0} 
                     for model in PATH_CONFIG['MODEL_PATHS'].keys()}
        
        # Setup
        self.setup_directories()
        self.load_models()
        self.print_status()

    def setup_directories(self):
        """Create necessary directories"""
        try:
            for model_name in PATH_CONFIG['MODEL_PATHS'].keys():
                path = os.path.join(PATH_CONFIG['RESULTS_DIR'], model_name)
                os.makedirs(path, exist_ok=True)
                print(f"Created/verified directory: {path}")
        except Exception as e:
            print(f"Error creating directories: {str(e)}")
            raise

    def load_models(self):
        """Load all models"""
        for model_name, path in PATH_CONFIG['MODEL_PATHS'].items():
            try:
                print(f"Loading model: {model_name}")
                if os.path.exists(path):
                    model = load_model(path, compile=False)
                    self.models[model_name] = model
                    print(f"Successfully loaded model: {model_name}")
                else:
                    print(f"Model not found: {path}")
            except Exception as e:
                print(f"Error loading {model_name}: {str(e)}")

    def print_status(self):
        """Print pipeline status"""
        print("\nPipeline Status:")
        print(f"Data directory: {self.data_dir}")
        print("Loaded models:", list(self.models.keys()))
        print(f"Results directory: {PATH_CONFIG['RESULTS_DIR']}")

    def load_patient_data(self, patient_id):
        """Load patient data"""
        try:
            patient_path = os.path.join(self.data_dir, patient_id)
            
            # Load CT data
            ct_path = os.path.join(patient_path, 'ct.csv')
            ct_data = pd.read_csv(ct_path, index_col=0)
            ct = np.zeros(MODEL_CONFIG['INPUT_SHAPE'][:3] + (1,), dtype=np.float32)
            ct.ravel()[ct_data.index] = ct_data['data'].values
            
            # Load structure masks
            masks = np.zeros(MODEL_CONFIG['INPUT_SHAPE'][:3] + (10,), dtype=np.float32)
            for idx, roi in enumerate(ROI_LIST):
                roi_path = os.path.join(patient_path, f'{roi}.csv')
                if os.path.exists(roi_path):
                    roi_data = pd.read_csv(roi_path, index_col=0)
                    masks.reshape(-1, 10)[roi_data.index, idx] = 1
            
            # Load possible dose mask
            mask_path = os.path.join(patient_path, 'possible_dose_mask.csv')
            mask_data = pd.read_csv(mask_path, index_col=0)
            dose_mask = np.zeros(MODEL_CONFIG['OUTPUT_SHAPE'], dtype=np.float32)
            dose_mask.ravel()[mask_data.index] = 1
            
            return {
                'ct': ct,
                'structure_masks': masks,
                'possible_dose_mask': dose_mask
            }
            
        except Exception as e:
            print(f"Error loading data for patient {patient_id}: {str(e)}")
            return None

    def predict_dose(self, model_name, patient_data):
        """Predict dose using specified model"""
        try:
            if model_name not in self.models:
                print(f"Model {model_name} not loaded")
                self.stats[model_name]['skipped'] += 1
                return None

            # Prepare input
            ct = normalize_dicom(patient_data['ct'])
            masks = patient_data['structure_masks']
            input_data = np.concatenate([ct, masks], axis=-1)
            input_data = np.expand_dims(input_data, axis=0)
            
            # Predict
            pred = self.models[model_name].predict(input_data, verbose=0)
            
            # Apply dose mask and clip values
            pred = pred[0] * patient_data['possible_dose_mask']
            pred = np.clip(pred, 0, MODEL_CONFIG['MAX_DOSE'])
            
            return pred

        except Exception as e:
            print(f"Error in {model_name} prediction: {str(e)}")
            self.stats[model_name]['failed'] += 1
            return None

    def scale_dose(self, dose, structure_masks):
        """Scale dose to match prescribed doses"""
        try:
            for ptv_name, prescribed_dose in DOSE_PRESCRIPTION.items():
                ptv_idx = ROI_LIST.index(ptv_name)
                ptv_mask = structure_masks[..., ptv_idx]
                
                if np.sum(ptv_mask) > 0:
                    ptv_dose = dose * ptv_mask
                    current_mean = np.mean(ptv_dose[ptv_mask > 0])
                    if current_mean > 0:
                        scale_factor = prescribed_dose / current_mean
                        dose = np.where(ptv_mask > 0, dose * scale_factor, dose)
            
            return np.clip(dose, 0, MODEL_CONFIG['MAX_DOSE'])
            
        except Exception as e:
            print(f"Error scaling dose: {str(e)}")
            return dose

    def save_prediction(self, dose_pred, patient_id, model_name):
        """Save prediction"""
        try:
            sparse_pred = sparse_vector_function(dose_pred)
            if sparse_pred is None or len(sparse_pred['data']) == 0:
                print(f"No valid prediction data for {patient_id} with {model_name}")
                return False

            # Create DataFrame and save
            output_path = os.path.join(PATH_CONFIG['RESULTS_DIR'], 
                                     model_name, 
                                     f'{patient_id}_dose.csv')
            
            df = pd.DataFrame({
                'data': sparse_pred['data']
            }, index=sparse_pred['indices'])
            
            df.to_csv(output_path)
            
            print(f"\nSaved {model_name} prediction:")
            print(f"  Path: {output_path}")
            print(f"  Non-zero points: {len(sparse_pred['data'])}")
            print(f"  Range: [{sparse_pred['data'].min():.4f}, {sparse_pred['data'].max():.4f}] Gy")
            
            return True
            
        except Exception as e:
            print(f"Error saving {model_name} prediction: {str(e)}")
            return False

    def run_pipeline(self):
        """Run full prediction pipeline"""
        print(f"\nStarting pipeline at {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
        
        try:
            # Get patient list
            patients = sorted([d for d in os.listdir(self.data_dir) 
                            if os.path.isdir(os.path.join(self.data_dir, d))])
            
            if not patients:
                raise ValueError("No patients found in data directory")

            # Process each patient
            for patient_id in tqdm(patients, desc="Processing patients"):
                print(f"\nProcessing patient: {patient_id}")
                
                # Load patient data
                patient_data = self.load_patient_data(patient_id)
                if patient_data is None:
                    continue
                
                # Process with each model
                for model_name in self.models:
                    try:
                        print(f"Processing with model: {model_name}")
                        
                        # Predict dose
                        dose_pred = self.predict_dose(model_name, patient_data)
                        if dose_pred is None:
                            continue
                            
                        # Scale dose
                        dose_pred = self.scale_dose(dose_pred, patient_data['structure_masks'])
                        
                        # Save prediction
                        if self.save_prediction(dose_pred, patient_id, model_name):
                            self.stats[model_name]['processed'] += 1
                        
                    except Exception as e:
                        print(f"Error with {model_name} for patient {patient_id}: {str(e)}")
                        self.stats[model_name]['failed'] += 1
                        continue
                    
                    finally:
                        # Clear memory
                        tf.keras.backend.clear_session()
                        gc.collect()

        except Exception as e:
            print(f"Pipeline error: {str(e)}")
            raise

        finally:
            self.print_statistics()

    def print_statistics(self):
        """Print final statistics"""
        print("\nPrediction Statistics:")
        for model_name, stats in self.stats.items():
            print(f"\n{model_name}:")
            print(f"  Processed: {stats['processed']}")
            print(f"  Failed: {stats['failed']}")
            print(f"  Skipped: {stats['skipped']}")
            
            total = stats['processed'] + stats['failed']
            if total > 0:
                success_rate = (stats['processed'] / total) * 100
                print(f"  Success Rate: {success_rate:.2f}%")
            else:
                print("  Success Rate: N/A (no predictions attempted)")

def main():
    """Main execution function"""
    START_TIME = datetime.datetime.utcnow()
    USER = "aagneye-syam"
    
    print(f"Execution started at {START_TIME.strftime('%Y-%m-%d %H:%M:%S')} UTC by {USER}")
    
    try:
        # Verify data directory
        if not os.path.exists(PATH_CONFIG['DATA_DIR']):
            raise FileNotFoundError(f"Data directory not found: {PATH_CONFIG['DATA_DIR']}")
        
        # Run pipeline
        pipeline = DosePredictionPipeline(PATH_CONFIG['DATA_DIR'])
        pipeline.run_pipeline()
        
    except Exception as e:
        print(f"Pipeline execution failed: {str(e)}")
        raise
        
    finally:
        end_time = datetime.datetime.utcnow()
        print(f"\nExecution completed at {end_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        print(f"Total execution time: {end_time - START_TIME}")

if __name__ == "__main__":
    main()