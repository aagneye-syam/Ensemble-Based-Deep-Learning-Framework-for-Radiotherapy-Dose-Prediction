"""
Dose Score Calculation Script
Created: 2025-03-06 17:32:38 UTC
Author: aagneye-syam
"""

import os
import pandas as pd
import numpy as np
import logging
import sys
from sklearn.metrics import mean_absolute_error
from config import PATH_CONFIG, ROI_CONFIG
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('dose_calculation.log')
    ]
)
logger = logging.getLogger(__name__)

# Current execution information
CURRENT_TIME = datetime.strptime("2025-03-06 17:32:38", "%Y-%m-%d %H:%M:%S")
CURRENT_USER = "aagneye-syam"

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
        logger.info(f"Successfully loaded {len(indices)} non-zero values from {file_path}")
        return dose_array.reshape((128, 128, 128))
    except Exception as e:
        logger.error(f"Error loading {file_path}: {str(e)}")
        raise

def calculate_dvh_score(true_dose, prediction):
    """Calculate DVH score with proper scaling."""
    def calculate_dvh(dose):
        # Scale the dose values to the expected range (typically 0-70 Gy)
        dose = dose * 70.0  # Apply scaling factor
        hist, bin_edges = np.histogram(dose.flatten(), bins=100, range=(0, dose.max()))
        dvh = np.cumsum(hist[::-1])[::-1] / hist.sum()
        return bin_edges[1:], dvh

    true_dose_bins, true_dvh = calculate_dvh(true_dose)
    pred_dose_bins, pred_dvh = calculate_dvh(prediction)
    dvh_score = np.trapz(np.abs(true_dvh - pred_dvh), true_dose_bins)
    # Scale the DVH score to the expected range
    return dvh_score * 70.0  # Apply scaling factor

def calculate_mae_score(true_dose, prediction):
    """Calculate MAE with proper scaling."""
    # Scale the values to the expected range (typically 0-70 Gy)
    scaled_true = true_dose * 70.0
    scaled_pred = prediction * 70.0
    return mean_absolute_error(scaled_true.flatten(), scaled_pred.flatten())

def main():
    """Main execution function"""
    print(f"\nDose Score Calculation")
    print(f"====================")
    print(f"Started at: {CURRENT_TIME}")
    print(f"User: {CURRENT_USER}")
    print("====================\n")

    # Directories setup
    test_pats_dir = os.path.join(PATH_CONFIG['DATA_DIR'], 'provided-data', 'test-pats')
    model_dirs = {
        'dense_u_net': os.path.join('results', 'dense_u_net'),
        'gan': os.path.join('results', 'gan'),
        'res_u_net': os.path.join('results', 'res_u_net'),
        'u_net': os.path.join('results', 'u_net')
    }
    output_dir = os.path.join('dose_score_results')
    os.makedirs(output_dir, exist_ok=True)

    # Score tracking
    scores = {model: {'dose': [], 'dvh': []} for model in model_dirs}
    all_scores = {'dose': [], 'dvh': []}
    stats = {
        'total_patients': 0,
        'processed': 0,
        'errors': []
    }

    # Process each patient
    patient_dirs = [d for d in os.listdir(test_pats_dir) if d.startswith('pt_')]
    stats['total_patients'] = len(patient_dirs)
    logger.info(f"Found {len(patient_dirs)} patients to process")

    for patient_id in sorted(patient_dirs):
        logger.info(f"\nProcessing patient: {patient_id}")
        try:
            # Load true dose
            true_dose_path = os.path.join(test_pats_dir, patient_id, 'dose.csv')
            true_dose = load_dose_file(true_dose_path)

            # Process each model's prediction
            patient_predictions = {}
            for model_name, model_dir in model_dirs.items():
                pred_path = os.path.join(model_dir, f'{patient_id}.csv')
                if os.path.exists(pred_path):
                    try:
                        prediction = load_dose_file(pred_path)
                        
                        # Calculate scores with scaling
                        mae = calculate_mae_score(true_dose, prediction)
                        dvh = calculate_dvh_score(true_dose, prediction)
                        
                        # Store scores
                        scores[model_name]['dose'].append(mae)
                        scores[model_name]['dvh'].append(dvh)
                        all_scores['dose'].append(mae)
                        all_scores['dvh'].append(dvh)
                        
                        patient_predictions[model_name] = {
                            'mae': mae,
                            'dvh': dvh
                        }
                        
                        logger.info(f"{model_name} scores - MAE: {mae:.6f}, DVH: {dvh:.6f}")
                    except Exception as e:
                        logger.error(f"Error processing {model_name} prediction: {str(e)}")
                else:
                    logger.warning(f"No prediction found for {model_name}")

            if patient_predictions:
                stats['processed'] += 1
                
                # Save patient scores
                patient_results_path = os.path.join(output_dir, f'{patient_id}_scores.txt')
                with open(patient_results_path, 'w') as f:
                    f.write(f"Patient: {patient_id}\n")
                    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("=" * 50 + "\n\n")
                    for model_name, scores_dict in patient_predictions.items():
                        f.write(f"{model_name}:\n")
                        f.write(f"  MAE Score: {scores_dict['mae']:.6f}\n")
                        f.write(f"  DVH Score: {scores_dict['dvh']:.6f}\n\n")

        except Exception as e:
            logger.error(f"Error processing patient {patient_id}: {str(e)}")
            stats['errors'].append(f"{patient_id}: {str(e)}")
            continue

    # Calculate and save final results
    summary_path = os.path.join(output_dir, 'final_results.txt')
    with open(summary_path, 'w') as f:
        f.write("Dose Score Calculation Results\n")
        f.write("=============================\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Generated by: {CURRENT_USER}\n\n")
        
        f.write("Overall Statistics:\n")
        f.write("-----------------\n")
        f.write(f"Total patients: {stats['total_patients']}\n")
        f.write(f"Successfully processed: {stats['processed']}\n")
        f.write(f"Success rate: {(stats['processed']/stats['total_patients'])*100:.2f}%\n\n")
        
        if all_scores['dose']:
            f.write("Combined Results (All Models):\n")
            f.write("--------------------------\n")
            mean_dose = np.mean(all_scores['dose'])
            std_dose = np.std(all_scores['dose'])
            mean_dvh = np.mean(all_scores['dvh'])
            std_dvh = np.std(all_scores['dvh'])
            f.write(f"Average Dose Score (MAE): {mean_dose:.6f} ± {std_dose:.6f}\n")
            f.write(f"Average DVH Score: {mean_dvh:.6f} ± {std_dvh:.6f}\n\n")
            
            # Print to console as well
            print(f"\nOverall Results:")
            print(f"Average Dose Score (MAE): {mean_dose:.6f} ± {std_dose:.6f}")
            print(f"Average DVH Score: {mean_dvh:.6f} ± {std_dvh:.6f}")
        
        f.write("Individual Model Results:\n")
        f.write("----------------------\n")
        for model_name, model_scores in scores.items():
            if model_scores['dose']:
                mean_dose = np.mean(model_scores['dose'])
                std_dose = np.std(model_scores['dose'])
                mean_dvh = np.mean(model_scores['dvh'])
                std_dvh = np.std(model_scores['dvh'])
                
                f.write(f"\n{model_name}:\n")
                f.write(f"  Dose Score (MAE): {mean_dose:.6f} ± {std_dose:.6f}\n")
                f.write(f"  DVH Score: {mean_dvh:.6f} ± {std_dvh:.6f}\n")
                
                # Print to console as well
                print(f"\n{model_name}:")
                print(f"  Average Dose Score (MAE): {mean_dose:.6f} ± {std_dose:.6f}")
                print(f"  Average DVH Score: {mean_dvh:.6f} ± {std_dvh:.6f}")
        
        if stats['errors']:
            f.write("\nErrors Encountered:\n")
            f.write("----------------\n")
            for error in stats['errors']:
                f.write(f"- {error}\n")

    logger.info(f"\nResults saved to {summary_path}")
    print(f"\nDetailed results saved to {summary_path}")

if __name__ == "__main__":
    main()