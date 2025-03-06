"""
Ensemble Prediction Generation Script
Created: 2025-03-06 17:10:31 UTC
Author: aagneye-syam
"""

import os
import numpy as np
import pandas as pd
import logging
import datetime
from sklearn.metrics import mean_absolute_error
from config import PATH_CONFIG

# Configure metadata
CURRENT_TIME = "2025-03-06 17:10:31"
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
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
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

def analyze_model_contributions(best_model_map):
    """
    Analyze how much each model contributed to the final ensemble.
    Args:
        best_model_map (numpy.ndarray): Array containing model names for each pixel
    Returns:
        dict: Model contribution percentages
    """
    unique_models, counts = np.unique(best_model_map[best_model_map != ''], return_counts=True)
    total_pixels = np.sum(counts)
    return {model: (count/total_pixels)*100 for model, count in zip(unique_models, counts)}

def main():
    print(f"\nEnsemble Prediction Generation")
    print(f"============================")
    print(f"Started at: {CURRENT_TIME}")
    print(f"User: {CURRENT_USER}")
    print("============================\n")
    
    # Setup directories
    base_dir = PATH_CONFIG['DATA_DIR']
    output_dir = 'ensemble_result'  # Root directory for ensemble results
    
    # Create directory structure
    subdirs = ['predictions', 'analysis', 'model_maps']
    for subdir in subdirs:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
    
    # Model configurations
    model_configs = {
        'dense_u_net': os.path.join('results', 'dense_u_net'),
        'gan': os.path.join('results', 'gan'),
        'res_u_net': os.path.join('results', 'res_u_net'),
        'u_net': os.path.join('results', 'u_net')
    }
    
    # Find patient directories
    patient_dirs = []
    true_dose_dir = os.path.join(base_dir, 'provided-data', 'test-pats')
    for root, dirs, files in os.walk(true_dose_dir):
        if 'dose.csv' in files:
            patient_dirs.append(root)
    
    logger.info(f"Found {len(patient_dirs)} patients to process")
    
    processed_patients = []
    failed_patients = []
    all_patient_scores = {}
    
    for patient_dir in patient_dirs:
        patient_id = os.path.basename(patient_dir)
        logger.info(f"\nProcessing patient: {patient_id}")
        logger.info("=" * 50)
        
        try:
            # Load true dose
            true_dose_path = os.path.join(patient_dir, 'dose.csv')
            true_dose = load_dose_file(true_dose_path)
            if true_dose is None:
                raise ValueError(f"Could not load true dose for patient {patient_id}")
            
            # Load predictions
            predictions = []
            model_names = []
            
            for model_name, pred_dir in model_configs.items():
                pred_path = os.path.join(pred_dir, f'{patient_id}.csv')  # Modified to match new naming
                if os.path.exists(pred_path):
                    pred_dose = load_dose_file(pred_path)
                    if pred_dose is not None:
                        predictions.append(pred_dose)
                        model_names.append(model_name)
                        logger.info(f"Loaded prediction from {model_name}")
                    else:
                        logger.warning(f"Failed to load prediction for {model_name}")
                else:
                    logger.warning(f"Missing prediction file for {model_name}")
            
            if not predictions:
                raise ValueError(f"No valid predictions found")
            
            # Create ensemble prediction
            logger.info("Creating ensemble prediction...")
            ensemble_prediction, best_model_map = create_ensemble_prediction(
                true_dose, predictions, model_names
            )
            
            # Verify ensemble prediction
            if np.count_nonzero(ensemble_prediction) == 0:
                raise ValueError("Generated ensemble prediction is empty!")
            
            # Save ensemble prediction
            pred_output_path = os.path.join(output_dir, 'predictions', f'{patient_id}.csv')
            save_dose_prediction(ensemble_prediction, pred_output_path)
            
            # Save model selection map
            np.save(
                os.path.join(output_dir, 'model_maps', f'{patient_id}_model_map.npy'),
                best_model_map
            )
            
            # Calculate metrics
            ensemble_mae = mean_absolute_error(true_dose.flatten(), ensemble_prediction.flatten())
            contributions = analyze_model_contributions(best_model_map)
            all_patient_scores[patient_id] = ensemble_mae
            
            # Save analysis results
            analysis_path = os.path.join(output_dir, 'analysis', f'{patient_id}_analysis.txt')
            with open(analysis_path, 'w') as f:
                f.write(f"Patient: {patient_id}\n")
                f.write(f"Generated: {CURRENT_TIME}\n")
                f.write(f"User: {CURRENT_USER}\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Ensemble Mean Absolute Error: {ensemble_mae:.6f}\n\n")
                f.write("Model Contributions:\n")
                f.write("-" * 20 + "\n")
                for model, contribution in contributions.items():
                    f.write(f"{model}: {contribution:.2f}%\n")
            
            processed_patients.append(patient_id)
            logger.info(f"Successfully processed patient {patient_id}")
            logger.info(f"MAE: {ensemble_mae:.6f}")
            
        except Exception as e:
            logger.error(f"Failed to process patient {patient_id}: {str(e)}")
            failed_patients.append((patient_id, str(e)))
    
    # Write summary report
    summary_path = os.path.join(output_dir, 'ensemble_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("Ensemble Generation Summary\n")
        f.write("========================\n\n")
        f.write(f"Generation Time: {CURRENT_TIME}\n")
        f.write(f"Generated by: {CURRENT_USER}\n\n")
        f.write(f"Total patients found: {len(patient_dirs)}\n")
        f.write(f"Successfully processed: {len(processed_patients)}\n")
        f.write(f"Failed: {len(failed_patients)}\n\n")
        
        if processed_patients:
            mean_mae = np.mean(list(all_patient_scores.values()))
            std_mae = np.std(list(all_patient_scores.values()))
            f.write(f"\nOverall Performance:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Mean MAE across all patients: {mean_mae:.6f} ± {std_mae:.6f}\n\n")
            
            f.write("Individual Patient Scores:\n")
            f.write("-" * 24 + "\n")
            for patient_id, mae in sorted(all_patient_scores.items()):
                f.write(f"{patient_id}: MAE = {mae:.6f}\n")
        
        if failed_patients:
            f.write("\nFailed Patients Details:\n")
            f.write("-" * 22 + "\n")
            for patient_id, error in failed_patients:
                f.write(f"{patient_id}: {error}\n")
    
    logger.info("\nEnsemble generation completed")
    logger.info(f"Results saved to {output_dir}")
    if processed_patients:
        logger.info(f"Overall Mean MAE: {mean_mae:.6f} ± {std_mae:.6f}")

if __name__ == '__main__':
    main()