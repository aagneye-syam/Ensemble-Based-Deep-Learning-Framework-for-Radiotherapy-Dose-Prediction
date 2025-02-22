import os
import datetime

# Record execution start time
START_TIME = datetime.datetime.utcnow()
USER = "aagneye-syam"
print(f"Execution started at {START_TIME.strftime('%Y-%m-%d %H:%M:%S')} UTC by {USER}")

# Set environment variables before importing tensorflow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from data_loader import DataLoader, normalize_dicom, get_paths
from tensorflow.keras.layers import concatenate
from tqdm import tqdm
from config import PATH_CONFIG
import gc

class MultiModelDosePredictionPipeline:
    def __init__(self, models_config, data_dir, batch_size=1):
        """
        Initialize the pipeline
        Args:
            models_config: Dictionary of model names and their paths
            data_dir: Directory containing the data
            batch_size: Batch size for processing (default: 1)
        """
        self.models = {}
        self.data_dir = data_dir
        self.active_models = []
        self.batch_size = batch_size
        
        # Create results directory in root
        os.makedirs('results', exist_ok=True)

        # Initialize data loader
        self.data_loader = DataLoader(
            get_paths(self.data_dir, ext=''),
            mode_name='dose_prediction',
            batch_size=self.batch_size,
            use_memmap=True
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
                    # Create model-specific results directory
                    os.makedirs(f'results/{model_name}', exist_ok=True)
                    print(f"Successfully loaded model: {model_name}")

            except Exception as e:
                print(f"Error loading model {model_name}: {str(e)}")
                continue

    def predict_single_case(self, model, patient_data):
        """
        Predict dose for a single case
        Args:
            model: The model to use for prediction
            patient_data: Patient data dictionary
        Returns:
            Predicted dose or None if error occurs
        """
        try:
            # Normalize and reshape CT data
            ct_normalized = normalize_dicom(patient_data['ct'])
            ct_reshaped = ct_normalized.reshape(1, 128, 128, 128, 1)
            
            # Reshape structure masks
            masks_reshaped = patient_data['structure_masks'].reshape(1, 128, 128, 128, 10)
            
            # Concatenate inputs
            input_data = concatenate([ct_reshaped, masks_reshaped], axis=-1)
            
            # Make prediction
            with tf.device('/CPU:0'):  # Force CPU to avoid GPU memory issues
                dose_pred = model.predict(input_data, verbose=0)
            
            # Apply dose mask
            dose_pred = dose_pred * patient_data['possible_dose_mask']
            
            return dose_pred.squeeze()

        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return None

    def save_prediction(self, dose_pred, patient_id, model_name):
        """
        Save prediction to CSV file in results directory
        Args:
            dose_pred: Predicted dose
            patient_id: Patient identifier
            model_name: Name of the model used
        Returns:
            Path to saved file or None if error occurs
        """
        try:
            # Extract just the patient ID from the full path
            patient_id = patient_id.split('\\')[-1]  # Get the last part of the path
            
            # Create output path in results directory
            output_dir = os.path.join('results', model_name)
            os.makedirs(output_dir, exist_ok=True)
            
            # Convert dose prediction to sparse format
            dose_to_save = self.sparse_vector_function(dose_pred)
            if dose_to_save is None:
                return None
            
            # Create DataFrame
            dose_df = pd.DataFrame(
                data=dose_to_save['data'].squeeze(),
                index=dose_to_save['indices'].squeeze(),
                columns=['data']
            )
            
            # Save to CSV in results directory
            output_path = os.path.join(output_dir, f'{patient_id}.csv')
            dose_df.to_csv(output_path)
            print(f"Saved prediction to {output_path}")
            
            return output_path
        except Exception as e:
            print(f"Error saving prediction: {str(e)}")
            return None

    def sparse_vector_function(self, x, indices=None):
        """
        Convert tensor to sparse format
        Args:
            x: Input tensor
            indices: Optional indices
        Returns:
            Dictionary with data and indices
        """
        try:
            if indices is None:
                return {
                    'data': x[x > 0],
                    'indices': np.nonzero(x.flatten())[-1]
                }
            else:
                return {
                    'data': x[x > 0],
                    'indices': indices[x > 0]
                }
        except Exception as e:
            print(f"Error in sparse vector conversion: {str(e)}")
            return None

    def run_pipeline(self):
        """Run the prediction pipeline"""
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
                            # Clear any existing tensors
                            tf.keras.backend.clear_session()
                            
                            model = self.models[model_name]
                            dose_pred = self.predict_single_case(model, patient_batch)
                            if dose_pred is not None:
                                self.save_prediction(dose_pred, patient_id, model_name)
                            
                        except Exception as e:
                            print(f"Error processing model {model_name}: {str(e)}")
                            continue

                    # Force garbage collection
                    gc.collect()

                except Exception as e:
                    print(f"Error processing batch {idx}: {str(e)}")
                    continue

        except Exception as e:
            print(f"Pipeline error: {str(e)}")
            raise

        finally:
            # Cleanup
            self.data_loader.cleanup()
            end_time = datetime.datetime.utcnow()
            duration = end_time - START_TIME
            print(f"\nPipeline completed at {end_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
            print(f"Total execution time: {duration}")

def main():
    """Main function"""
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

        # Create results directory
        os.makedirs('results', exist_ok=True)
        print("Created results directory")

        # Initialize and run pipeline
        print("Initializing pipeline...")
        pipeline = MultiModelDosePredictionPipeline(
            models_config, 
            data_dir,
            batch_size=1
        )
        
        print("Starting pipeline execution...")
        pipeline.run_pipeline()

    except Exception as e:
        print(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()