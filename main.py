import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from data_loader import DataLoader, normalize_dicom
from tensorflow.keras.layers import concatenate

class DosePredictionPipeline:
    def __init__(self, model_path, data_dir, output_dir):
        self.model = load_model(model_path)
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def sparse_vector_function(self, x, indices=None):
        """Convert tensor to dictionary of non-zero values and corresponding indices"""
        if indices is None:
            y = {'data': x[x > 0], 'indices': np.nonzero(x.flatten())[-1]}
        else:
            y = {'data': x[x > 0], 'indices': indices[x > 0]}
        return y
        
    def predict_single_case(self, patient_data):
        """Predict dose for a single patient case"""
        # Prepare input
        ct_normalized = normalize_dicom(patient_data['ct'])
        input_data = concatenate([
            ct_normalized.reshape(1, 128, 128, 128, 1),
            patient_data['structure_masks'].reshape(1, 128, 128, 128, 10)
        ])
        
        # Make prediction
        dose_pred = self.model.predict(input_data)
        
        # Apply possible dose mask
        dose_pred = dose_pred * patient_data['possible_dose_mask']
        return dose_pred.squeeze()
        
    def save_prediction(self, dose_pred, patient_id):
        """Save predicted dose to CSV"""
        dose_to_save = self.sparse_vector_function(dose_pred)
        dose_df = pd.DataFrame(
            data=dose_to_save['data'].squeeze(),
            index=dose_to_save['indices'].squeeze(),
            columns=['data']
        )
        output_path = os.path.join(self.output_dir, f'{patient_id}.csv')
        dose_df.to_csv(output_path)
        
    def run_pipeline(self):
        """Run the complete prediction pipeline"""
        # Initialize data loader
        data_loader = DataLoader(
            get_paths(self.data_dir, ext=''),
            mode_name='dose_prediction'
        )
        
        number_of_batches = data_loader.number_of_batches()
        print(f'Processing {number_of_batches} patients...')
        
        for idx in range(number_of_batches):
            # Load patient data
            patient_batch = data_loader.get_batch(idx)
            patient_id = patient_batch['patient_list'][0]
            print(f'Processing patient: {patient_id}')
            
            # Predict and save dose
            dose_pred = self.predict_single_case(patient_batch)
            self.save_prediction(dose_pred, patient_id)
            print(f'Completed prediction for {patient_id}')

if __name__ == "__main__":
    # Configuration
    MODEL_PATH = "models/u_net_model/3D_UNet128_100epochs.h5"
    DATA_DIR = "/open-kbp-master"
    OUTPUT_DIR = "results/u_net_prediction"
    
    # Run pipeline
    pipeline = DosePredictionPipeline(MODEL_PATH, DATA_DIR, OUTPUT_DIR)
    pipeline.run_pipeline()