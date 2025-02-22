import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from data_loader import DataLoader, normalize_dicom, get_paths
from tensorflow.keras.layers import concatenate
import tqdm
from config import PATH_CONFIG

# Disable MKL to avoid primitive creation errors
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Set memory growth for GPU if available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

class MultiModelDosePredictionPipeline:
    def __init__(self, models_config, data_dir, batch_size=1):
        self.models = {}
        self.data_dir = data_dir
        self.active_models = []
        self.batch_size = batch_size

        # Initialize data loader with memory-efficient settings
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

                model = self.load_model_with_custom_objects(model_path, model_name)
                if model is not None:
                    self.models[model_name] = model
                    self.active_models.append(model_name)
                    output_dir = os.path.join('results', f'{model_name}_prediction')
                    os.makedirs(output_dir, exist_ok=True)

            except Exception as e:
                print(f"Error loading model {model_name}: {str(e)}")
                continue

    def load_model_with_custom_objects(self, model_path, model_name):
        """Load model with custom objects if needed"""
        try:
            if model_name == 'attention_u_net':
                custom_objects = {'AttentionLayer': AttentionLayer}
            else:
                custom_objects = None
            
            return load_model(model_path, custom_objects=custom_objects, compile=False)
        except Exception as e:
            print(f"Error loading model {model_name}: {str(e)}")
            return None

    def predict_single_case(self, model, patient_data):
        """Predict dose for a single case with error handling"""
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

    def run_pipeline(self):
        """Run the prediction pipeline with improved error handling"""
        if not self.active_models:
            print("No active models found")
            return

        try:
            number_of_batches = self.data_loader.number_of_batches()
            if number_of_batches == 0:
                print("No batches to process")
                return

            for idx in tqdm.tqdm(range(number_of_batches)):
                try:
                    # Get patient batch
                    patient_batch = self.data_loader.get_batch(idx)
                    if patient_batch is None or 'patient_list' not in patient_batch:
                        print(f"Invalid batch data for index {idx}")
                        continue

                    patient_id = patient_batch['patient_list'][0]
                    print(f"Processing patient: {patient_id}")

                    # Process with each model
                    for model_name in self.active_models:
                        try:
                            dose_pred = self.predict_single_case(self.models[model_name], patient_batch)
                            if dose_pred is not None:
                                self.save_prediction(dose_pred, patient_id, model_name)
                        except Exception as e:
                            print(f"Error processing model {model_name}: {str(e)}")
                            continue

                except Exception as e:
                    print(f"Error processing batch {idx}: {str(e)}")
                    continue

        except Exception as e:
            print(f"Pipeline error: {str(e)}")
            raise

def main():
    """Main function with improved error handling"""
    try:
        # Configure models
        models_config = {
            'u_net': PATH_CONFIG['U_NET_PATH'],
            'attention_u_net': PATH_CONFIG['ATTENTION_U_NET_PATH'],
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

        # Initialize and run pipeline
        pipeline = MultiModelDosePredictionPipeline(
            models_config, 
            data_dir,
            batch_size=1  # Use small batch size to avoid memory issues
        )
        pipeline.run_pipeline()

    except Exception as e:
        print(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()