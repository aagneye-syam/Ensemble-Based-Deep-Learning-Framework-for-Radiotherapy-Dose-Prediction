"""
Dose Distribution Visualization Script
Created: 2025-02-24 15:09:22 UTC
Author: aagneye-syam
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('visualization.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    'INPUT_SHAPE': (128, 128, 128),
    'SLICE_POSITIONS': [
        (64, 64, 64),   # middle slices
        (32, 32, 32),   # anterior slices
        (96, 96, 96)    # posterior slices
    ],
    'PATHS': {
        'BASE_DIR': 'ensemble_result',
        'VIZ_DIR': 'ensemble_result/visualizations',
        'PRED_DIR': 'ensemble_result/predictions',
        'TRUE_DOSE_DIR': 'open-kbp-master/provided-data/test-pats'
    }
}

def load_dose_file(file_path):
    """
    Load dose data from CSV file and reshape to 3D array.
    
    Args:
        file_path (str): Path to the dose CSV file
    Returns:
        numpy.ndarray: 3D array of dose values
    """
    try:
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return None
            
        df = pd.read_csv(file_path)
        dose_array = np.zeros((128 * 128 * 128,))
        
        if not df.empty:
            if 'Unnamed: 0' in df.columns:
                indices = df['Unnamed: 0'].values
                data = df['data'].values if 'data' in df.columns else df.iloc[:, 1].values
            else:
                indices = df.index.values
                data = df.iloc[:, 0].values if 'data' not in df.columns else df['data'].values
            
            dose_array[indices] = data
            logger.debug(f"Loaded {len(indices)} non-zero values from {file_path}")
            logger.debug(f"Value range: [{data.min():.2f}, {data.max():.2f}]")
        
        return dose_array.reshape(CONFIG['INPUT_SHAPE'])
    
    except Exception as e:
        logger.error(f"Error loading file {file_path}: {str(e)}")
        return None

def create_dose_colormap():
    """
    Create a custom colormap for dose visualization.
    
    Returns:
        matplotlib.colors.LinearSegmentedColormap: Custom colormap for dose visualization
    """
    colors = [
        (0, 0, 0),           # black for 0
        (0.267, 0, 0.329),   # dark purple
        (0.133, 0, 0.671),   # blue
        (0, 0.467, 0.698),   # light blue
        (0.067, 0.667, 0.467), # cyan
        (0.333, 0.800, 0.200), # green
        (0.667, 0.867, 0.133), # yellow-green
        (0.867, 0.800, 0.133), # yellow
        (1, 0.667, 0.133),   # orange
        (1, 0.333, 0.133),   # red-orange
        (1, 0, 0)           # red
    ]
    
    return LinearSegmentedColormap.from_list('dose_colormap', colors)

def plot_dose_comparison(true_dose, pred_dose, slice_indices=None, save_path=None):
    """
    Create a side-by-side comparison of true and predicted dose distributions.
    
    Args:
        true_dose (numpy.ndarray): 3D array of true dose values
        pred_dose (numpy.ndarray): 3D array of predicted dose values
        slice_indices (tuple): Optional (axial, sagittal, coronal) indices
        save_path (str): Path to save the visualization
    """
    if slice_indices is None:
        slice_indices = tuple(shape // 2 for shape in CONFIG['INPUT_SHAPE'])
    
    # Create figure with proper spacing
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 3, width_ratios=[1, 1, 0.1], figure=fig)
    
    # Add metadata
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fig.suptitle(
        f"Dose Distribution Comparison\n"
        f"Generated: {current_time}\n"
        f"Slice Position: ({slice_indices[0]}, {slice_indices[1]}, {slice_indices[2]})",
        fontsize=14, y=0.95
    )
    
    # Setup visualization parameters
    cmap = create_dose_colormap()
    vmin = min(true_dose.min(), pred_dose.min())
    vmax = max(true_dose.max(), pred_dose.max())
    
    views = ['Axial', 'Sagittal', 'Coronal']
    slices = [
        (slice_indices[0], slice(None), slice(None)),
        (slice(None), slice_indices[1], slice(None)),
        (slice(None), slice(None), slice_indices[2])
    ]
    
    # Create visualizations for each view
    for idx, (view, slice_idx) in enumerate(zip(views, slices)):
        # True dose plot
        ax1 = fig.add_subplot(gs[idx, 0])
        im1 = ax1.imshow(
            true_dose[slice_idx].squeeze().T,
            cmap=cmap, vmin=vmin, vmax=vmax
        )
        ax1.set_title(f'True Dose\n{view} View')
        ax1.axis('off')
        
        # Add slice indicator
        ax1.text(0.02, 0.98,
                 f'Slice: {slice_idx[0] if isinstance(slice_idx[0], int) else slice_idx[1]}',
                 transform=ax1.transAxes,
                 color='white',
                 fontsize=8,
                 verticalalignment='top',
                 bbox=dict(facecolor='black', alpha=0.5))
        
        # Predicted dose plot
        ax2 = fig.add_subplot(gs[idx, 1])
        im2 = ax2.imshow(
            pred_dose[slice_idx].squeeze().T,
            cmap=cmap, vmin=vmin, vmax=vmax
        )
        ax2.set_title(f'Ensemble Prediction\n{view} View')
        ax2.axis('off')
        
        # Add slice indicator
        ax2.text(0.02, 0.98,
                 f'Slice: {slice_idx[0] if isinstance(slice_idx[0], int) else slice_idx[1]}',
                 transform=ax2.transAxes,
                 color='white',
                 fontsize=8,
                 verticalalignment='top',
                 bbox=dict(facecolor='black', alpha=0.5))
    
    # Add colorbar
    cbar_ax = fig.add_subplot(gs[:, 2])
    cbar = fig.colorbar(im2, cax=cbar_ax)
    cbar.set_label('Dose (Gy)', rotation=270, labelpad=15)
    
    # Adjust layout
    plt.subplots_adjust(
        left=0.05, right=0.95,
        bottom=0.05, top=0.9,
        wspace=0.2, hspace=0.3
    )
    
    # Save or display
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved visualization to: {save_path}")
    else:
        plt.show()
    
    plt.close()

def process_patient(patient_id, paths_config):
    """
    Process a single patient's dose distributions.
    
    Args:
        patient_id (str): Patient identifier
        paths_config (dict): Configuration dictionary containing file paths
    """
    try:
        # Load prediction
        pred_path = os.path.join(paths_config['PRED_DIR'], f'{patient_id}.csv')
        pred_dose = load_dose_file(pred_path)
        
        # Load true dose
        true_path = os.path.join(paths_config['TRUE_DOSE_DIR'], patient_id, 'dose.csv')
        true_dose = load_dose_file(true_path)
        
        if pred_dose is None or true_dose is None:
            logger.error(f"Could not load dose data for patient {patient_id}")
            return False
        
        # Create visualizations for different slices
        for slice_set in CONFIG['SLICE_POSITIONS']:
            viz_path = os.path.join(
                paths_config['VIZ_DIR'],
                f'{patient_id}_slice_{slice_set[0]}_{slice_set[1]}_{slice_set[2]}.png'
            )
            
            plot_dose_comparison(
                true_dose,
                pred_dose,
                slice_indices=slice_set,
                save_path=viz_path
            )
        
        return True
    
    except Exception as e:
        logger.error(f"Error processing patient {patient_id}: {str(e)}")
        return False

def main():
    """Main execution function"""
    logger.info("Starting dose visualization pipeline")
    
    # Create visualization directory
    os.makedirs(CONFIG['PATHS']['VIZ_DIR'], exist_ok=True)
    
    # Process all patients
    processed = 0
    failed = 0
    
    pred_files = [f for f in os.listdir(CONFIG['PATHS']['PRED_DIR']) 
                 if f.endswith('.csv')]
    
    for pred_file in pred_files:
        patient_id = pred_file.replace('.csv', '')
        logger.info(f"\nProcessing patient: {patient_id}")
        
        if process_patient(patient_id, CONFIG['PATHS']):
            processed += 1
            logger.info(f"Successfully processed patient {patient_id}")
        else:
            failed += 1
            logger.error(f"Failed to process patient {patient_id}")
    
    # Log summary
    logger.info("\nVisualization Pipeline Summary")
    logger.info("============================")
    logger.info(f"Total patients: {len(pred_files)}")
    logger.info(f"Successfully processed: {processed}")
    logger.info(f"Failed: {failed}")
    logger.info("============================")

if __name__ == '__main__':
    main()