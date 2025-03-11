"""
Model-wise Score Calculation Script
Created: 2025-03-09 10:14:35 UTC
Author: aagneye-syam
"""

import os
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from sklearn.metrics import mean_absolute_error

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('model_score_calculation.log')
    ]
)
logger = logging.getLogger(__name__)

# Current execution information
CURRENT_TIME = datetime.strptime("2025-03-09 10:14:35", "%Y-%m-%d %H:%M:%S")
CURRENT_USER = "aagneye-syam"

def load_dose_file(file_path):
    """Load dose data from CSV file with proper scaling."""
    try:
        logger.info(f"Loading file: {file_path}")
        df = pd.read_csv(file_path)
        dose_array = np.zeros((128 * 128 * 128,))
        
        if 'Unnamed: 0' in df.columns:
            indices = df['Unnamed: 0'].values
            data = df['data'].values if 'data' in df.columns else df.iloc[:, 1].values
        else:
            indices = df.index.values
            data = df.iloc[:, 0].values if 'data' not in df.columns else df['data'].values
            
        dose_array[indices] = data
        
        # Scale dose to Gy if the max value is less than 1
        if np.max(data) < 1:
            dose_array = dose_array * 70  # Scale to Gy units
            logger.info("Scaled dose values to Gy units")
            
        logger.info(f"Successfully loaded {len(indices)} non-zero values")
        logger.info(f"Dose range: [{np.min(data):.6f}, {np.max(data):.6f}] Gy")
        return dose_array.reshape((128, 128, 128))
    
    except Exception as e:
        logger.error(f"Error loading {file_path}: {str(e)}")
        return None

def calculate_dose_score(true_dose, prediction, possible_dose_mask=None):
    """Calculate dose score (MAE) in Gy units."""
    # Ensure both doses are in same scale (Gy)
    if np.max(true_dose) < 1:
        true_dose = true_dose * 70
    if np.max(prediction) < 1:
        prediction = prediction * 70
    
    if possible_dose_mask is not None:
        valid_voxels = possible_dose_mask > 0
        return np.sum(np.abs(true_dose[valid_voxels] - prediction[valid_voxels])) / np.sum(valid_voxels)
    else:
        return mean_absolute_error(true_dose.flatten(), prediction.flatten())

def calculate_dvh_score(true_dose, prediction):
    """Calculate DVH score using histogram comparison."""
    def calculate_dvh(dose):
        # Ensure dose is in Gy units for DVH calculation
        if np.max(dose) < 1:
            dose = dose * 70
        hist, bin_edges = np.histogram(dose.flatten(), bins=100, range=(0, dose.max()))
        dvh = np.cumsum(hist[::-1])[::-1] / hist.sum()
        return bin_edges[1:], dvh

    true_dose_bins, true_dvh = calculate_dvh(true_dose)
    pred_dose_bins, pred_dvh = calculate_dvh(prediction)
    return np.trapz(np.abs(true_dvh - pred_dvh), true_dose_bins)

def main():
    print(f"\nModel-wise Score Calculation")
    print(f"==========================")
    print(f"Time: {CURRENT_TIME.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"User: {CURRENT_USER}")
    print("==========================\n")

    # Setup directories
    test_pats_dir = os.path.join('open-kbp-master', 'provided-data', 'test-pats')
    model_dirs = {
        'u_net': os.path.join('results', 'u_net'),
        'gan': os.path.join('results', 'gan'),
        'dense_u_net': os.path.join('results', 'dense_u_net'),
        'res_u_net': os.path.join('results', 'res_u_net'),
        'ensemble': os.path.join('ensemble_result', 'predictions')
    }
    output_dir = 'model_scores'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize score tracking for each model
    scores = {model: {'patient_id': [], 'dose_score': [], 'dvh_score': []} 
             for model in model_dirs.keys()}

    # Process each patient
    for patient_dir in sorted(os.listdir(test_pats_dir)):
        if not patient_dir.startswith('pt_'):
            continue

        logger.info(f"\nProcessing patient: {patient_dir}")
        
        try:
            # Load true dose
            true_dose_path = os.path.join(test_pats_dir, patient_dir, 'dose.csv')
            if not os.path.exists(true_dose_path):
                logger.error(f"True dose file not found for {patient_dir}")
                continue
                
            true_dose = load_dose_file(true_dose_path)
            if true_dose is None:
                continue

            # Process each model's prediction
            for model_name, model_dir in model_dirs.items():
                try:
                    # All prediction files use the format pt_xxx.csv
                    pred_path = os.path.join(model_dir, f"{patient_dir}.csv")
                    
                    if not os.path.exists(pred_path):
                        logger.warning(f"No prediction found for {model_name}: {pred_path}")
                        continue
                    
                    prediction = load_dose_file(pred_path)
                    if prediction is None:
                        continue
                    
                    # Calculate scores
                    dose_score = calculate_dose_score(true_dose, prediction)
                    dvh_score = calculate_dvh_score(true_dose, prediction)
                    
                    # Store scores
                    scores[model_name]['patient_id'].append(patient_dir)
                    scores[model_name]['dose_score'].append(dose_score)
                    scores[model_name]['dvh_score'].append(dvh_score)
                    
                    logger.info(f"{model_name} - Dose Score: {dose_score:.6f} Gy, DVH Score: {dvh_score:.6f}")
                
                except Exception as e:
                    logger.error(f"Error processing {model_name} for {patient_dir}: {str(e)}")
                    continue

        except Exception as e:
            logger.error(f"Error processing {patient_dir}: {str(e)}")
            continue

    # Calculate and save results
    summary_path = os.path.join(output_dir, 'model_scores_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("Model Score Summary\n")
        f.write("=================\n\n")
        f.write(f"Generated on: {CURRENT_TIME.strftime('%Y-%m-%d %H:%M:%S')} UTC\n")
        f.write(f"Generated by: {CURRENT_USER}\n\n")
        
        # Print and save results for each model
        print("\nResults by Model:")
        print("===============")
        
        f.write("Results by Model:\n")
        f.write("===============\n")
        
        for model_name, model_scores in scores.items():
            if model_scores['patient_id']:  # Only process if we have scores
                mean_dose = np.mean(model_scores['dose_score'])
                std_dose = np.std(model_scores['dose_score'])
                mean_dvh = np.mean(model_scores['dvh_score'])
                std_dvh = np.std(model_scores['dvh_score'])
                
                model_summary = f"\n{model_name}:\n"
                model_summary += f"  Processed patients: {len(model_scores['patient_id'])}\n"
                model_summary += f"  Average Dose Score: {mean_dose:.6f} ± {std_dose:.6f} Gy\n"
                model_summary += f"  Average DVH Score: {mean_dvh:.6f} ± {std_dvh:.6f}\n"
                
                # Write to file
                f.write(model_summary)
                
                # Print to console
                print(model_summary)
                
                # Save detailed scores for each model
                model_detail_path = os.path.join(output_dir, f'{model_name}_detailed_scores.txt')
                with open(model_detail_path, 'w') as mf:
                    mf.write(f"{model_name} Detailed Scores\n")
                    mf.write("=" * (len(model_name) + 15) + "\n\n")
                    mf.write(f"Generated on: {CURRENT_TIME.strftime('%Y-%m-%d %H:%M:%S')} UTC\n\n")
                    
                    for i, pat_id in enumerate(model_scores['patient_id']):
                        mf.write(f"\n{pat_id}:\n")
                        mf.write(f"  Dose Score: {model_scores['dose_score'][i]:.6f} Gy\n")
                        mf.write(f"  DVH Score: {model_scores['dvh_score'][i]:.6f}\n")

        print(f"\nDetailed results saved to {output_dir}/")

if __name__ == '__main__':
    main()