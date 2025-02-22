import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from data_loader import DataLoader, normalize_dicom, get_paths
from tensorflow.keras.layers import concatenate
import tqdm
from config import PATH_CONFIG

# Define the AttentionLayer class
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
                    continue

                self.models[model_name] = self.load_model_with_custom_objects(model_path, model_name)
                self.active_models.append(model_name)
                
                output_dir = os.path.join('results', f'{model_name}_prediction')
                os.makedirs(output_dir, exist_ok=True)

            except Exception:
                continue

    def load_model_with_custom_objects(self, model_path, model_name):
        custom_objects = {
            'attention_u_net': {
                'AttentionLayer': AttentionLayer,
            },
            'dense_u_net': {
                # Add any custom layers needed for DenseUNet
            }
        }
        if model_name in custom_objects:
            return load_model(model_path, custom_objects=custom_objects[model_name], compile=False)
        else:
            return load_model(model_path, compile=False)

    def sparse_vector_function(self, x, indices=None):
        if indices is None:
            y = {'data': x[x > 0], 'indices': np.nonzero(x.flatten())[-1]}
        else:
            y = {'data': x[x > 0], 'indices': indices[x > 0]}
        return y

    def predict_single_case(self, model, patient_data, model_name):
        try:
            ct_normalized = normalize_dicom(patient_data['ct'])
            input_data = concatenate([
                ct_normalized.reshape(1, 128, 128, 128, 1),
                patient_data['structure_masks'].reshape(1, 128, 128, 128, 10)
            ], axis=-1)

            dose_pred = model.predict(input_data, verbose=0)
            dose_pred = dose_pred * patient_data['possible_dose_mask']
            return dose_pred.squeeze()

        except Exception:
            raise

    def save_prediction(self, dose_pred, patient_id, model_name):
        try:
            output_dir = os.path.join('results', f'{model_name}_prediction')
            dose_to_save = self.sparse_vector_function(dose_pred)
            dose_df = pd.DataFrame(
                data=dose_to_save['data'].squeeze(),
                index=dose_to_save['indices'].squeeze(),
                columns=['data']
            )
            output_path = os.path.join(output_dir, f'{patient_id}.csv')
            dose_df.to_csv(output_path)
            return output_path
        except Exception:
            raise

    def run_pipeline(self):
        if not self.active_models:
            return

        try:
            data_loader = DataLoader(
                get_paths(self.data_dir, ext=''),
                mode_name='dose_prediction'
            )

            number_of_batches = data_loader.number_of_batches()
            if number_of_batches == 0:
                return

            for idx in tqdm.tqdm(range(number_of_batches)):
                patient_batch = data_loader.get_batch(idx)
                patient_id = patient_batch['patient_list'][0]

                for model_name in self.active_models:
                    try:
                        dose_pred = self.predict_single_case(self.models[model_name], patient_batch, model_name)
                        self.save_prediction(dose_pred, patient_id, model_name)
                    except Exception:
                        continue

        except Exception:
            raise

def main():
    try:
        models_config = {
            'u_net': PATH_CONFIG['U_NET_PATH'],
            'attention_u_net': PATH_CONFIG['ATTENTION_U_NET_PATH'],
            'dense_u_net': PATH_CONFIG['DENSE_U_NET_PATH'],
            'gan': PATH_CONFIG['GAN_PATH'],
            'res_u_net': PATH_CONFIG['RES_U_NET_PATH']
        }

        data_dir = PATH_CONFIG['DATA_DIR']

        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory not found at {data_dir}")

        test_pats_dir = os.path.join(data_dir, 'provided-data', 'test-pats')
        if os.path.exists(test_pats_dir):
            data_dir = test_pats_dir

        pipeline = MultiModelDosePredictionPipeline(models_config, data_dir)
        pipeline.run_pipeline()

    except Exception:
        raise

if __name__ == "__main__":
    main()