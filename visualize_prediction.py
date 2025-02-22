import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_loader import DataLoader, get_paths
from pathlib import Path

# Record execution info
CURRENT_TIME = datetime.datetime.strptime("2025-02-22 08:11:30", "%Y-%m-%d %H:%M:%S")
USER = "aagneye-syam"
print(f"Visualization started at {CURRENT_TIME} UTC by {USER}")

class DoseComparisonVisualizer:
    def __init__(self, results_dir='results', data_dir=None):
        """Initialize visualizer with results and data directories"""
        self.results_dir = results_dir
        self.data_dir = data_dir if data_dir else os.path.join('provided-data', 'test-pats')
        self.viz_dir = os.path.join('visualizations', 'dose_comparisons')
        os.makedirs(self.viz_dir, exist_ok=True)
        
        # Initialize data loader for true doses
        self.data_loader = DataLoader(
            get_paths(self.data_dir),
            mode_name='dose_prediction',
            batch_size=1
        )
        print(f"Visualization directory created at: {self.viz_dir}")

    def load_true_dose(self, patient_id):
        """Load true dose from data loader"""
        try:
            patient_data = self.data_loader.get_batch(patient_list=[patient_id])
            if patient_data is None:
                print(f"Could not load true dose for patient {patient_id}")
                return None
            
            true_dose = patient_data['dose'][0, ..., 0]
            print(f"\nTrue dose statistics:")
            print(f"Shape: {true_dose.shape}")
            print(f"Range: [{true_dose.min():.4f}, {true_dose.max():.4f}] Gy")
            print(f"Mean: {true_dose.mean():.4f} Gy")
            return true_dose, patient_data['ct'][0, ..., 0]
            
        except Exception as e:
            print(f"Error loading true dose: {str(e)}")
            return None, None

    def load_prediction(self, model_name, patient_id):
        """Load predicted dose from CSV"""
        try:
            pred_path = os.path.join(self.results_dir, model_name, f"{patient_id}.csv")
            print(f"\nLoading prediction from: {pred_path}")
            
            if not os.path.exists(pred_path):
                print(f"Prediction file not found: {pred_path}")
                return None
                
            pred_df = pd.read_csv(pred_path)
            pred_dose = np.zeros((128, 128, 128), dtype=np.float32)
            
            # Get indices and data
            indices = pred_df.iloc[:, 0].values.astype(int)
            data = pred_df.iloc[:, 1].values.astype(np.float32)
            
            # Validate indices
            if indices.max() >= pred_dose.size:
                valid_mask = indices < pred_dose.size
                indices = indices[valid_mask]
                data = data[valid_mask]
            
            pred_dose.ravel()[indices] = data
            
            print(f"\nPredicted dose statistics:")
            print(f"Shape: {pred_dose.shape}")
            print(f"Range: [{pred_dose.min():.4f}, {pred_dose.max():.4f}] Gy")
            print(f"Mean: {pred_dose.mean():.4f} Gy")
            
            return pred_dose
            
        except Exception as e:
            print(f"Error loading prediction: {str(e)}")
            return None

    def visualize_comparison(self, model_name, patient_id, slices=None, view='axial'):
        """Visualize predicted dose alongside true dose"""
        try:
            if slices is None:
                slices = [32, 64, 96]

            # Load doses
            true_dose, ct_data = self.load_true_dose(patient_id)
            pred_dose = self.load_prediction(model_name, patient_id)
            
            if true_dose is None or pred_dose is None:
                print("Failed to load dose data")
                return

            # Create figure with three rows: CT, True Dose, Predicted Dose
            n_slices = len(slices)
            fig, axes = plt.subplots(3, n_slices, figsize=(6*n_slices, 15))

            # Get global range for consistent colormaps
            vmin = min(true_dose.min(), pred_dose.min())
            vmax = max(true_dose.max(), pred_dose.max())

            # Plot each slice
            for slice_idx, slice_num in enumerate(slices):
                if view == 'axial':
                    ct_slice = ct_data[:, :, slice_num]
                    true_slice = true_dose[:, :, slice_num]
                    pred_slice = pred_dose[:, :, slice_num]
                    view_label = 'z'
                elif view == 'sagittal':
                    ct_slice = ct_data[slice_num, :, :]
                    true_slice = true_dose[slice_num, :, :]
                    pred_slice = pred_dose[slice_num, :, :]
                    view_label = 'x'
                else:  # coronal
                    ct_slice = ct_data[:, slice_num, :]
                    true_slice = true_dose[:, slice_num, :]
                    pred_slice = pred_dose[:, slice_num, :]
                    view_label = 'y'

                # Plot CT
                im_ct = axes[0, slice_idx].imshow(ct_slice, cmap='gray')
                axes[0, slice_idx].set_title(f'CT Slice {view_label}={slice_num}')
                plt.colorbar(im_ct, ax=axes[0, slice_idx])

                # Plot True Dose
                im_true = axes[1, slice_idx].imshow(true_slice, cmap='jet', vmin=vmin, vmax=vmax)
                axes[1, slice_idx].set_title(f'True Dose\nRange: [{true_slice.min():.1f}, {true_slice.max():.1f}] Gy')
                plt.colorbar(im_true, ax=axes[1, slice_idx])

                # Plot Predicted Dose
                im_pred = axes[2, slice_idx].imshow(pred_slice, cmap='jet', vmin=vmin, vmax=vmax)
                axes[2, slice_idx].set_title(f'Predicted Dose\nRange: [{pred_slice.min():.1f}, {pred_slice.max():.1f}] Gy')
                plt.colorbar(im_pred, ax=axes[2, slice_idx])

                # Add labels to all plots
                for ax_row in axes:
                    ax_row[slice_idx].set_xlabel('Position (pixels)')
                    ax_row[slice_idx].set_ylabel('Position (pixels)')

            # Add overall title
            fig.suptitle(
                f'Dose Distribution Comparison\n'
                f'Model: {model_name}, Patient: {patient_id}\n'
                f'Global Dose Range: [{vmin:.1f}, {vmax:.1f}] Gy', 
                size=14, y=0.95
            )

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
        
        # Get available models
        models = [d for d in os.listdir('results') 
                 if os.path.isdir(os.path.join('results', d))]
        
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
        patients = [f.split('.')[0] for f in os.listdir(model_dir) 
                   if f.endswith('.csv')]
        
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