import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_loader import DataLoader, get_paths, sparse_vector_function
from pathlib import Path

# Record execution info
CURRENT_TIME = datetime.datetime.strptime("2025-02-24 14:50:04", "%Y-%m-%d %H:%M:%S")
USER = "aagneye-syam"
print(f"Visualization started at {CURRENT_TIME} UTC by {USER}")

class DoseComparisonVisualizer:
    def __init__(self, results_dir='results', data_dir=None):
        """Initialize visualizer with results and data directories"""
        self.results_dir = results_dir
        # Update data directory path to include provided-data structure
        self.data_dir = os.path.join('open-kbp-master', 'provided-data', 'test-pats') if data_dir is None else data_dir
        self.viz_dir = os.path.join('visualizations', 'dose_comparisons')
        os.makedirs(self.viz_dir, exist_ok=True)
        print(f"Visualization directory created at: {self.viz_dir}")

        # Initialize data loader for true doses
        try:
            paths = get_paths(self.data_dir)
            if not paths:
                print(f"No data found in: {self.data_dir}")
                self.data_loader = None
                return
                
            self.data_loader = DataLoader(
                paths,
                mode_name='evaluation',  # Changed to evaluation mode to include dose data
                batch_size=1
            )
            print(f"Data loader initialized with {len(paths)} paths")
        except Exception as e:
            print(f"Error initializing data loader: {str(e)}")
            self.data_loader = None

    def load_true_dose(self, patient_id):
        """Load true dose from data loader"""
        try:
            # First try loading from the data loader
            if self.data_loader is not None:
                base_patient_id = patient_id.split('_dose')[0].split('.')[0]
                print(f"Attempting to load true dose for patient: {base_patient_id}")
                
                patient_data = self.data_loader.get_batch(patient_list=[base_patient_id])
                if patient_data is not None and 'dose' in patient_data and patient_data['dose'] is not None:
                    true_dose = patient_data['dose'][0, ..., 0]
                    ct_data = patient_data['ct'][0, ..., 0] if 'ct' in patient_data else None
                    
                    print(f"\nTrue dose statistics from data loader:")
                    print(f"Shape: {true_dose.shape}")
                    print(f"Range: [{true_dose.min():.4f}, {true_dose.max():.4f}] Gy")
                    print(f"Mean: {true_dose.mean():.4f} Gy")
                    return true_dose, ct_data

            # If data loader failed, try loading directly from dose_score_results
            dose_score_path = os.path.join('dose_score_results', f'{base_patient_id}_true_dose.npy')
            if os.path.exists(dose_score_path):
                print(f"Loading true dose from: {dose_score_path}")
                true_dose = np.load(dose_score_path)
                print(f"\nTrue dose statistics from saved file:")
                print(f"Shape: {true_dose.shape}")
                print(f"Range: [{true_dose.min():.4f}, {true_dose.max():.4f}] Gy")
                print(f"Mean: {true_dose.mean():.4f} Gy")
                return true_dose, None

            # If both methods fail, try loading from the original path
            original_dose_path = os.path.join(self.data_dir, base_patient_id, 'dose.csv')
            if os.path.exists(original_dose_path):
                print(f"Loading true dose from original path: {original_dose_path}")
                dose_df = pd.read_csv(original_dose_path, index_col=0)
                true_dose = np.zeros((128, 128, 128), dtype=np.float32)
                
                indices = dose_df.index.values.astype(int)
                data = dose_df['data'].values.astype(np.float32)
                true_dose.ravel()[indices] = data
                
                print(f"\nTrue dose statistics from CSV:")
                print(f"Shape: {true_dose.shape}")
                print(f"Range: [{true_dose.min():.4f}, {true_dose.max():.4f}] Gy")
                print(f"Mean: {true_dose.mean():.4f} Gy")
                return true_dose, None

            print(f"Could not load true dose for patient {patient_id}")
            return None, None
            
        except Exception as e:
            print(f"Error loading true dose: {str(e)}")
            print("Debug info:")
            if 'patient_data' in locals():
                print(f"Keys in patient_data: {patient_data.keys() if patient_data else 'None'}")
            return None, None

    def load_prediction(self, model_name, patient_id):
        """Load predicted dose from CSV"""
        try:
            # Handle both with and without extension
            base_patient_id = patient_id.split('.')[0]
            pred_path = os.path.join(self.results_dir, model_name, f"{base_patient_id}_dose.csv")
            print(f"\nLoading prediction from: {pred_path}")
            
            if not os.path.exists(pred_path):
                print(f"Prediction file not found: {pred_path}")
                # Try alternative path without _dose suffix
                alt_path = os.path.join(self.results_dir, model_name, f"{base_patient_id}.csv")
                if os.path.exists(alt_path):
                    pred_path = alt_path
                    print(f"Found alternative path: {alt_path}")
                else:
                    return None
                
            pred_df = pd.read_csv(pred_path, index_col=0)
            pred_dose = np.zeros((128, 128, 128), dtype=np.float32)
            
            # Get indices and data
            indices = pred_df.index.values.astype(int)
            data = pred_df['data'].values.astype(np.float32)
            
            # Validate indices
            if indices.max() >= pred_dose.size:
                valid_mask = indices < pred_dose.size
                indices = indices[valid_mask]
                data = data[valid_mask]
                print(f"Warning: Some indices were out of bounds. Filtered {(~valid_mask).sum()} points.")
            
            pred_dose.ravel()[indices] = data
            
            print(f"\nPredicted dose statistics:")
            print(f"Shape: {pred_dose.shape}")
            print(f"Range: [{pred_dose.min():.4f}, {pred_dose.max():.4f}] Gy")
            print(f"Mean: {pred_dose.mean():.4f} Gy")
            print(f"Non-zero points: {(pred_dose > 0).sum()}")
            
            return pred_dose
            
        except Exception as e:
            print(f"Error loading prediction: {str(e)}")
            print("Debug info:")
            if 'pred_df' in locals():
                print(f"DataFrame shape: {pred_df.shape}")
                print(f"DataFrame columns: {pred_df.columns}")
            return None

    def visualize_comparison(self, model_name, patient_id, slices=None, view='axial'):
        """Visualize predicted dose alongside true dose"""
        try:
            if slices is None:
                slices = [32, 64, 96]

            print(f"\nLoading dose data for comparison...")
            # Load doses
            true_dose, ct_data = self.load_true_dose(patient_id)
            pred_dose = self.load_prediction(model_name, patient_id)
            
            if true_dose is None and pred_dose is None:
                print("Failed to load both true and predicted dose data")
                return
            elif true_dose is None:
                print("Warning: True dose data not available, showing prediction only")
                true_dose = np.zeros_like(pred_dose)
            elif pred_dose is None:
                print("Warning: Prediction data not available, showing true dose only")
                pred_dose = np.zeros_like(true_dose)

            # Validate slice numbers
            max_slice = true_dose.shape[2] - 1
            valid_slices = [s for s in slices if 0 <= s <= max_slice]
            if len(valid_slices) != len(slices):
                print(f"Warning: Some slice numbers were invalid. Using {valid_slices}")
                slices = valid_slices

            if not slices:
                print("Error: No valid slice numbers provided")
                return

            # Create figure with two columns: True Dose and Predicted Dose side by side
            n_slices = len(slices)
            fig = plt.figure(figsize=(15, 5 * n_slices))
            
            # Get global range for consistent colormaps
            vmin = min(true_dose.min(), pred_dose.min())
            vmax = max(true_dose.max(), pred_dose.max())

            # Plot each slice
            for slice_idx, slice_num in enumerate(slices):
                try:
                    if view == 'axial':
                        true_slice = true_dose[:, :, slice_num]
                        pred_slice = pred_dose[:, :, slice_num]
                        view_label = 'z'
                    elif view == 'sagittal':
                        true_slice = true_dose[slice_num, :, :]
                        pred_slice = pred_dose[slice_num, :, :]
                        view_label = 'x'
                    else:  # coronal
                        true_slice = true_dose[:, slice_num, :]
                        pred_slice = pred_dose[:, slice_num, :]
                        view_label = 'y'

                    # Create subplot for this slice
                    plt.subplot(n_slices, 2, 2*slice_idx + 1)
                    plt.title(f'True Dose (Slice {view_label}={slice_num})\nRange: [{true_slice.min():.1f}, {true_slice.max():.1f}] Gy')
                    im_true = plt.imshow(true_slice, cmap='jet', vmin=vmin, vmax=vmax)
                    plt.colorbar(im_true)
                    plt.xlabel('Position (pixels)')
                    plt.ylabel('Position (pixels)')

                    plt.subplot(n_slices, 2, 2*slice_idx + 2)
                    plt.title(f'Predicted Dose (Slice {view_label}={slice_num})\nRange: [{pred_slice.min():.1f}, {pred_slice.max():.1f}] Gy')
                    im_pred = plt.imshow(pred_slice, cmap='jet', vmin=vmin, vmax=vmax)
                    plt.colorbar(im_pred)
                    plt.xlabel('Position (pixels)')
                    plt.ylabel('Position (pixels)')

                except Exception as e:
                    print(f"Error plotting slice {slice_num}: {str(e)}")
                    continue

            # Add overall title
            plt.suptitle(
                f'Dose Distribution Comparison\n'
                f'Model: {model_name}, Patient: {patient_id}\n'
                f'Global Dose Range: [{vmin:.1f}, {vmax:.1f}] Gy', 
                size=14, y=1.02
            )

            # Adjust layout
            plt.tight_layout()

            # Save figure
            save_path = os.path.join(
                self.viz_dir, 
                f'{patient_id}_{model_name}_{view}_comparison.png'
            )
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nSaved comparison to: {save_path}")
            plt.close()

        except Exception as e:
            print(f"Error in visualization: {str(e)}")
            import traceback
            print(traceback.format_exc())

def main():
    """Main function with interactive selection"""
    try:
        print("\n=== Dose Distribution Comparison Visualizer ===")
        visualizer = DoseComparisonVisualizer()
        
        # Check if results directory exists
        if not os.path.exists('results'):
            print("Error: 'results' directory not found")
            return
        
        # Get available models
        models = [d for d in os.listdir('results') 
                 if os.path.isdir(os.path.join('results', d))]
        
        if not models:
            print("No model directories found in 'results' directory")
            return
        
        print("\nAvailable models:")
        for i, model in enumerate(models, 1):
            print(f"{i}. {model}")
        
        # Get model selection
        while True:
            try:
                model_idx = int(input("\nSelect model number: ")) - 1
                if 0 <= model_idx < len(models):
                    model_name = models[model_idx]
                    break
                print("Invalid selection. Please try again.")
            except ValueError:
                print("Please enter a number.")

        # Get available patients
        model_dir = os.path.join('results', model_name)
        patients = [f.split('_dose.csv')[0] for f in os.listdir(model_dir) 
                   if f.endswith('.csv')]
        
        if not patients:
            print(f"No patient files found in {model_dir}")
            return
        
        print("\nAvailable patients:")
        for i, patient in enumerate(patients, 1):
            print(f"{i}. {patient}")
        
        # Get patient selection
        while True:
            try:
                patient_idx = int(input("\nSelect patient number: ")) - 1
                if 0 <= patient_idx < len(patients):
                    patient_id = patients[patient_idx]
                    break
                print("Invalid selection. Please try again.")
            except ValueError:
                print("Please enter a number.")

        # Get viewing plane
        views = ['axial', 'sagittal', 'coronal']
        print("\nAvailable views:")
        for i, view in enumerate(views, 1):
            print(f"{i}. {view}")
        
        while True:
            try:
                view_idx = int(input("\nSelect view number: ")) - 1
                if 0 <= view_idx < len(views):
                    view = views[view_idx]
                    break
                print("Invalid selection. Please try again.")
            except ValueError:
                print("Please enter a number.")

        # Get slice indices
        while True:
            try:
                slice_input = input("\nEnter slice numbers (comma-separated, or press Enter for default [32,64,96]): ")
                if slice_input.strip() == "":
                    slices = [32, 64, 96]
                    break
                slices = [int(s.strip()) for s in slice_input.split(',')]
                if all(0 <= s < 128 for s in slices):
                    break
                print("Invalid slice numbers. Please enter numbers between 0 and 127.")
            except ValueError:
                print("Please enter valid numbers.")

        # Create visualization
        print(f"\nCreating visualization for:")
        print(f"Model: {model_name}")
        print(f"Patient: {patient_id}")
        print(f"View: {view}")
        print(f"Slices: {slices}")
        
        visualizer.visualize_comparison(model_name, patient_id, slices, view)

        # Print completion information
        end_time = datetime.datetime.utcnow()
        duration = end_time - CURRENT_TIME
        print(f"\nVisualization completed at {end_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        print(f"Total execution time: {duration}")

    except Exception as e:
        print(f"\nError in main: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()