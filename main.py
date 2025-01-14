import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from data_loader import DataLoader, normalize_dicom, get_paths
from tensorflow.keras.layers import concatenate
import tqdm
from config import PATH_CONFIG

class DosePredictionPipeline:
    def __init__(self, model_path, data_dir, output_dir):
        print(f"Loading model from {model_path}...")
        self.model = load_model(model_path)
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory created at {output_dir}")
        
    def sparse_vector_function(self, x, indices=None):
        """Convert tensor to dictionary of non-zero values and corresponding indices"""
        if indices is None:
            y = {'data': x[x > 0], 'indices': np.nonzero(x.flatten())[-1]}
        else:
            y = {'data': x[x > 0], 'indices': indices[x > 0]}
        return y
        
    def predict_single_case(self, patient_data):
        """Predict dose for a single patient case"""
        try:
            # Prepare input
            ct_normalized = normalize_dicom(patient_data['ct'])
            input_data = concatenate([
                ct_normalized.reshape(1, 128, 128, 128, 1),
                patient_data['structure_masks'].reshape(1, 128, 128, 128, 10)
            ], axis=-1)  # Add axis parameter
            
            # Make prediction
            dose_pred = self.model.predict(input_data, verbose=0)
            
            # Apply possible dose mask
            dose_pred = dose_pred * patient_data['possible_dose_mask']
            return dose_pred.squeeze()
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            raise
        
    def save_prediction(self, dose_pred, patient_id):
        """Save predicted dose to CSV"""
        try:
            dose_to_save = self.sparse_vector_function(dose_pred)
            dose_df = pd.DataFrame(
                data=dose_to_save['data'].squeeze(),
                index=dose_to_save['indices'].squeeze(),
                columns=['data']
            )
            output_path = os.path.join(self.output_dir, f'{patient_id}.csv')
            dose_df.to_csv(output_path)
            print(f"Saved prediction to {output_path}")
            return output_path
        except Exception as e:
            print(f"Error saving prediction: {str(e)}")
            raise
        
    def run_pipeline(self):
        """Run the complete prediction pipeline"""
        try:
            # Initialize data loader
            data_loader = DataLoader(
                get_paths(self.data_dir, ext=''),
                mode_name='dose_prediction'
            )
            
            number_of_batches = data_loader.number_of_batches()
            if number_of_batches == 0:
                print(f"No files found in {self.data_dir}")
                return
                
            print(f'Processing {number_of_batches} patients...')
            
            for idx in tqdm.tqdm(range(number_of_batches)):
                # Load patient data
                patient_batch = data_loader.get_batch(idx)
                patient_id = patient_batch['patient_list'][0]
                print(f'\nProcessing patient: {patient_id}')
                
                # Predict and save dose
                dose_pred = self.predict_single_case(patient_batch)
                output_path = self.save_prediction(dose_pred, patient_id)
                print(f'Completed prediction for {patient_id}')
                
        except Exception as e:
            print(f"Error in pipeline: {str(e)}")
            raise

def main():
    try:
        # Configuration
        model_path = PATH_CONFIG['MODEL_PATH']
        data_dir = PATH_CONFIG['DATA_DIR']
        output_dir = PATH_CONFIG['OUTPUT_DIR']
        
        # Validate paths
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory not found at {data_dir}")
        
        # Check for proper directory structure
        test_pats_dir = os.path.join(data_dir, 'provided-data', 'test-pats')
        if os.path.exists(test_pats_dir):
            data_dir = test_pats_dir
            print(f"Using test patients directory: {data_dir}")
        else:
            print(f"Warning: Could not find test-pats directory at {test_pats_dir}")
            print(f"Available directories in {data_dir}:")
            print(os.listdir(data_dir))
            
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Starting pipeline with:")
        print(f"Model path: {model_path}")
        print(f"Data directory: {data_dir}")
        print(f"Output directory: {output_dir}")
        
        # Run pipeline
        pipeline = DosePredictionPipeline(model_path, data_dir, output_dir)
        pipeline.run_pipeline()
        
    except Exception as e:
        print(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()