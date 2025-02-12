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
