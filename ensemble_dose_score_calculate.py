"""
Ensemble Score Calculation Script
Created: 2025-03-09 15:05:00 UTC
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
        logging.FileHandler('ensemble_score_calculation.log')
    ]
)
logger = logging.getLogger(__name__)

def load_dose_file(file_path):
    """Load dose data from CSV file."""
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
        logger.info(f"Successfully loaded {len(indices)} non-zero values")
        logger.info(f"Value range: [{np.min(data):.6f}, {np.max(data):.6f}]")
        return dose_array.reshape((128, 128, 128))
    
    except Exception as e:
        logger.error(f"Error loading {file_path}: {str(e)}")
        return None

def calculate_dvh_score(true_dose, prediction):
    """Calculate DVH score using histogram comparison."""
    def calculate_dvh(dose):
        hist, bin_edges = np.histogram(dose.flatten(), bins=100, range=(0, dose.max()))
        dvh = np.cumsum(hist[::-1])[::-1] / hist.sum()
        return bin_edges[1:], dvh

    true_dose_bins, true_dvh = calculate_dvh(true_dose)
    pred_dose_bins, pred_dvh = calculate_dvh(prediction)
    return np.trapz(np.abs(true_dvh - pred_dvh), true_dose_bins)

def main():
    print(f"\nEnsemble Score Calculation")
    print(f"========================")
    print(f"Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"User: aagneye-syam")
    print("========================\n")

    # Setup directories
    test_pats_dir = os.path.join('open-kbp-master', 'provided-data', 'test-pats')
    ensemble_dir = os.path.join('ensemble_result', 'predictions')
    output_dir = 'ensemble_scores'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Verify directories exist
    if not os.path.exists(test_pats_dir):
        logger.error(f"Test data directory not found: {test_pats_dir}")
        return
    if not os.path.exists(ensemble_dir):
        logger.error(f"Ensemble predictions directory not found: {ensemble_dir}")
        return

    # Initialize score tracking
    scores = {
        'patient_id': [],
        'dose_score': [],
        'dvh_score': []
    }

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

            # Load ensemble prediction
            ensemble_path = os.path.join(ensemble_dir, f"{patient_dir}.csv")
            if not os.path.exists(ensemble_path):
                logger.warning(f"No ensemble prediction found for {patient_dir}")
                continue
                
            ensemble_pred = load_dose_file(ensemble_path)
            if ensemble_pred is None:
                continue
            
            # Calculate scores
            dose_score = mean_absolute_error(true_dose.flatten(), ensemble_pred.flatten())
            dvh_score = calculate_dvh_score(true_dose, ensemble_pred)
            
            # Store scores
            scores['patient_id'].append(patient_dir)
            scores['dose_score'].append(dose_score)
            scores['dvh_score'].append(dvh_score)
            
            logger.info(f"Scores calculated - Dose: {dose_score:.6f}, DVH: {dvh_score:.6f}")

        except Exception as e:
            logger.error(f"Error processing {patient_dir}: {str(e)}")
            continue

    # Calculate and save final results
    if scores['patient_id']:
        mean_dose = np.mean(scores['dose_score'])
        std_dose = np.std(scores['dose_score'])
        mean_dvh = np.mean(scores['dvh_score'])
        std_dvh = np.std(scores['dvh_score'])
        
        summary_path = os.path.join(output_dir, 'ensemble_score_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("Ensemble Score Summary\n")
            f.write("====================\n\n")
            f.write(f"Generated on: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC\n")
            f.write(f"Generated by: aagneye-syam\n\n")
            
            f.write("Overall Results:\n")
            f.write("--------------\n")
            f.write(f"Total patients processed: {len(scores['patient_id'])}\n")
            f.write(f"Average Dose Score (MAE): {mean_dose:.6f} ± {std_dose:.6f}\n")
            f.write(f"Average DVH Score: {mean_dvh:.6f} ± {std_dvh:.6f}\n\n")
            
            f.write("Individual Patient Results:\n")
            f.write("------------------------\n")
            for i, pat_id in enumerate(scores['patient_id']):
                f.write(f"\n{pat_id}:\n")
                f.write(f"  Dose Score: {scores['dose_score'][i]:.6f}\n")
                f.write(f"  DVH Score: {scores['dvh_score'][i]:.6f}\n")
        
        # Print results to console
        print("\nResults Summary:")
        print("===============")
        print(f"Total patients processed: {len(scores['patient_id'])}")
        print(f"Average Dose Score (MAE): {mean_dose:.6f} ± {std_dose:.6f}")
        print(f"Average DVH Score: {mean_dvh:.6f} ± {std_dvh:.6f}")
        print(f"\nDetailed results saved to {summary_path}")
    else:
        print("No scores were calculated")

if __name__ == '__main__':
    main()