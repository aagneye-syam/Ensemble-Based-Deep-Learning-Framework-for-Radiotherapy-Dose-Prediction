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