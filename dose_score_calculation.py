import os
import pandas as pd
import numpy as np
import logging
from sklearn.metrics import mean_absolute_error
from config import PATH_CONFIG

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_csv(file_path):
    """Load dose data from a CSV file."""
    logger.debug(f"Loading CSV file from {file_path}")
    return pd.read_csv(file_path).values

def load_predictions_from_directory(directory):
    """Load all CSV files from a directory."""
    predictions = []
    for file_name in os.listdir(directory):
        if file_name.endswith('.csv'):
            file_path = os.path.join(directory, file_name)
            predictions.append(load_csv(file_path))
    return predictions

def pad_array(array, target_shape):
    """Pad an array to the target shape with zeros."""
    result = np.zeros(target_shape)
    result[:array.shape[0], :array.shape[1]] = array
    return result

def check_and_pad_predictions(predictions):
    """Check and pad predictions to ensure consistent shape."""
    # Find the maximum shape among the predictions
    if not predictions:
        logger.warning("No predictions to pad.")
        return []
    
    max_shape = tuple(np.max([pred.shape for pred in predictions], axis=0))
    logger.debug(f"Max shape for padding: {max_shape}")
    
    # Pad all predictions to the maximum shape
    padded_predictions = [pad_array(pred, max_shape) for pred in predictions]
    
    return padded_predictions

def calculate_dose_scores(true_dose, predictions):
    """Calculate dose scores (mean absolute error) for each prediction."""
    errors = []
    for prediction in predictions:
        mae = mean_absolute_error(true_dose.flatten(), prediction.flatten())
        errors.append(mae)
    return errors

def find_patient_dirs(base_dir):
    """Recursively find all directories containing dose.csv."""
    patient_dirs = []
    for root, dirs, files in os.walk(base_dir):
        if 'dose.csv' in files:
            patient_dirs.append(root)
    return patient_dirs

# Paths to the directories containing the predicted dose data
directories = [
    'results/dense_u_net_prediction',
    'results/gan_prediction',
    'results/res_u_net_prediction',
    'results/u_net_prediction'
]

# Function to load the true dose data for a specific patient
def load_true_dose(patient_dir):
    true_dose_file_path = os.path.join(patient_dir, 'dose.csv')
    if os.path.exists(true_dose_file_path):
        return load_csv(true_dose_file_path)
    else:
        raise FileNotFoundError(f"True dose file {true_dose_file_path} does not exist.")

# Base directory containing patient data
provided_data_dir = os.path.join(PATH_CONFIG['DATA_DIR'], 'provided-data')
# logger.debug(f"Provided data directory: {provided_data_dir}")

# Recursively find patient directories containing dose.csv
patient_dirs = find_patient_dirs(provided_data_dir)
# logger.debug(f"Found patient directories: {patient_dirs}")

# Create directory for storing dose scores if it doesn't exist
output_dir = 'dose_score_results'
os.makedirs(output_dir, exist_ok=True)
# logger.info(f"Output directory: {output_dir}")

# Calculate dose scores for each patient
for patient_dir in patient_dirs:
    patient_id = os.path.basename(patient_dir)
    # logger.info(f"Processing patient: {patient_id}")

    try:
        # Load true dose data for the patient
        true_dose = load_true_dose(patient_dir)
        # logger.debug(f"Loaded true dose data for patient {patient_id}")

        # Load predictions for the patient from each directory
        all_predictions = []
        for directory in directories:
            prediction_file = os.path.join(directory, f'{patient_id}.csv')
            if os.path.exists(prediction_file):
                prediction = load_csv(prediction_file)
                all_predictions.append(prediction)
            else:
                logger.warning(f"Prediction file {prediction_file} does not exist.")

        if not all_predictions:
            logger.warning(f"No predictions found for patient {patient_id}")
            continue

        # Check and pad predictions
        all_predictions = check_and_pad_predictions(all_predictions)
        # logger.debug(f"Padded predictions for patient {patient_id}")

        # Ensure true dose matches the shape of predictions
        true_dose_shape = true_dose.shape
        all_predictions = [pad_array(pred, true_dose_shape) for pred in all_predictions]
        # logger.debug(f"Ensured true dose shape matches predictions for patient {patient_id}")

        # Calculate dose scores
        dose_scores = calculate_dose_scores(true_dose, all_predictions)
        # logger.debug(f"Calculated dose scores for patient {patient_id}")

        # Save the dose scores and predictions for ensemble building
        dose_scores_path = f'{output_dir}/{patient_id}_dose_scores.npy'
        all_predictions_path = f'{output_dir}/{patient_id}_all_predictions.npy'
        true_dose_path = f'{output_dir}/{patient_id}_true_dose.npy'

        np.save(dose_scores_path, dose_scores)
        np.save(all_predictions_path, all_predictions)
        np.save(true_dose_path, true_dose)

        # logger.info(f"Dose scores for patient {patient_id} saved at {dose_scores_path}")
        # logger.info(f"All predictions for patient {patient_id} saved at {all_predictions_path}")
        # logger.info(f"True dose for patient {patient_id} saved at {true_dose_path}")

    except Exception as e:
        logger.error(f"Error processing patient {patient_id}: {e}")

logger.info("All dose scores calculated and saved.")