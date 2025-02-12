import os
import numpy as np
import pandas as pd
import logging
import datetime
from sklearn.metrics import mean_absolute_error
from config import PATH_CONFIG

# Configure metadata
CURRENT_TIME = "2025-02-02 10:16:41"
CURRENT_USER = "aagneye-syam"

# Configure logging with timestamp
log_filename = f'ensemble_log_{CURRENT_TIME.replace(" ", "_").replace(":", "-")}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_filename)
    ]
)
logger = logging.getLogger(__name__)

def load_dose_file(file_path):
    """
    Load dose data from CSV file.
    Args:
        file_path (str): Path to the dose CSV file
    Returns:
        numpy.ndarray: 3D array of dose values
    """
    try:
        # Read the CSV file
        logger.info(f"Loading file: {file_path}")
        df = pd.read_csv(file_path)
        
        # Initialize a zero array
        dose_array = np.zeros((128 * 128 * 128,))
        
        # If the file is not empty
        if not df.empty:
            # Handle both index column cases
            if 'Unnamed: 0' in df.columns:
                indices = df['Unnamed: 0'].values
                if 'data' in df.columns:
                    data = df['data'].values
                else:
                    data = df.iloc[:, 1].values
            else:
                indices = df.index.values
                data = df.iloc[:, 0].values if 'data' not in df.columns else df['data'].values
            
            # Fill the array with non-zero values
            dose_array[indices] = data
            
            logger.info(f"Loaded {len(indices)} non-zero values")
            logger.info(f"Value range: [{np.min(data):.6f}, {np.max(data):.6f}]")
            
        return dose_array.reshape((128, 128, 128))
    except Exception as e:
        logger.error(f"Error loading file {file_path}: {str(e)}")
        return None
    
def save_dose_prediction(dose_array, output_path):
    """
    Save dose prediction in sparse format.
    Args:
        dose_array (numpy.ndarray): 3D array of dose values
        output_path (str): Path to save the CSV file
    """
    try:
        # Get non-zero elements
        flat_dose = dose_array.flatten()
        non_zero_indices = np.nonzero(flat_dose)[0]
        non_zero_values = flat_dose[non_zero_indices]
        
        # Ensure we have data to save
        if len(non_zero_values) == 0:
            logger.error("No non-zero values found in dose array!")
            return
        
        # Create DataFrame with explicit column name
        df = pd.DataFrame({
            'data': non_zero_values
        }, index=non_zero_indices)
        
        # Save to CSV with index but without index name
        df.to_csv(output_path)
        
        # Verify the save
        logger.info(f"Saved prediction to {output_path}")
        logger.info(f"Number of non-zero values: {len(non_zero_values)}")
        logger.info(f"Value range: [{np.min(non_zero_values):.6f}, {np.max(non_zero_values):.6f}]")
        
        # Read back and verify
        test_df = pd.read_csv(output_path)
        if len(test_df) != len(non_zero_values):
            logger.error(f"Verification failed: Expected {len(non_zero_values)} rows, got {len(test_df)}")
            raise ValueError("Data verification failed")
            
    except Exception as e:
        logger.error(f"Error saving prediction to {output_path}: {str(e)}")
        raise
    
def calculate_pixel_wise_mae(true_value, predicted_value):
    """Calculate mean absolute error for a single pixel."""
    return abs(true_value - predicted_value)

def create_ensemble_prediction(true_dose, predictions, model_names):
    """
    Create ensemble prediction by selecting pixels with lowest dose score.
    Args:
        true_dose (numpy.ndarray): True dose values
        predictions (list): List of prediction arrays
        model_names (list): List of model names
    Returns:
        tuple: (ensemble_prediction, best_model_map)
    """
    if not predictions:
        raise ValueError("No predictions provided for ensemble")
    
    shape = true_dose.shape
    ensemble_prediction = np.zeros(shape)
    best_model_map = np.zeros(shape, dtype='U20')
    
    # Get non-zero indices from true dose
    non_zero_indices = np.nonzero(true_dose)
    total_pixels = len(non_zero_indices[0])
    logger.info(f"Processing {total_pixels} non-zero pixels")
    
    # Process each non-zero position
    for idx in range(total_pixels):
        i, j, k = [coord[idx] for coord in non_zero_indices]
        true_value = true_dose[i, j, k]
        
        # Calculate MAE for each model's prediction
        pixel_scores = []
        pixel_predictions = []
        for pred in predictions:
            pred_value = pred[i, j, k]
            score = calculate_pixel_wise_mae(true_value, pred_value)
            pixel_scores.append(score)
            pixel_predictions.append(pred_value)
        
        # Select best prediction
        best_model_idx = np.argmin(pixel_scores)
        ensemble_prediction[i, j, k] = pixel_predictions[best_model_idx]
        best_model_map[i, j, k] = model_names[best_model_idx]
        
        # Log progress every 10%
        if idx % (total_pixels // 10) == 0:
            logger.info(f"Processing progress: {(idx/total_pixels)*100:.1f}%")
            
    # Verify ensemble prediction has non-zero values
    non_zero_count = np.count_nonzero(ensemble_prediction)
    logger.info(f"Ensemble prediction contains {non_zero_count} non-zero values")
    if non_zero_count == 0:
        raise ValueError("Ensemble prediction is empty!")
    
    return ensemble_prediction, best_model_map