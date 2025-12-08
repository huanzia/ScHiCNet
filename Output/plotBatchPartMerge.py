import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# ---------------- Configuration ----------------

chromosomes = [2, 6, 10, 12]  # Chromosomes to process
res = 40000
percent = 0.75
cell_no = 1
crop_bin = 40
cell_line = 'Human'

# Use relative paths dynamically based on the script location
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
base_dir = os.path.join(project_root, 'Training', f'DataFull_{cell_line}_cell{cell_no}_{percent}_{res}')

full_dir = os.path.join(base_dir, 'Full_Mats')
pred_root = os.path.join(base_dir, 'Preds')

# Models to compare (must match the model_id used during prediction)
model_ids = [
    # "hicsrNet",
    # "hiedsrgan",
    # "deephic",
    # "schicatt",
    "ScHiCNet",
]

# Block size used during prediction
block_size = 40

# Whether to save the figure; Save directory: <project_root>/All_models/<model_id>/
save_fig = True
dpi = 600

# ---------------- Utility Functions ----------------
def find_latest_pred(pred_root_dir: str, model_id: str, chr_id: int, reso: int, block: int) -> str:
    """
    Find the prediction file for the current chromosome under Preds/<model_id>/.
    If multiple candidates exist, return the one with the latest modification time.
    Returns an empty string if not found.
    Target pattern:
      GSE131811_full_chr_{chr_id}_{reso}_piece_{block}_predict-{model_id}_bs{block}_w*.npy
    """
    model_dir = os.path.join(pred_root_dir, model_id)
    if not os.path.isdir(model_dir):
        return ""

    pattern = os.path.join(
        model_dir,
        f"GSE131811_full_chr_{chr_id}_{reso}_piece_{block}_predict-{model_id}_bs{block}_w*.npy"
    )
    candidates = glob.glob(pattern)
    if not candidates:
        # Fallback for legacy naming convention
        legacy = os.path.join(pred_root_dir, f"GSE131811_full_chr_{chr_id}_{reso}_piece_predict{block}.npy")
        return legacy if os.path.isfile(legacy) else ""

    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]

# ---------------- Normalization ----------------
def normalize(data: np.ndarray) -> np.ndarray:
    """Normalize data to the range [0, 1]."""
    return (data - np.min(data)) / (np.max(data) - np.min(data))

# ---------------- Resize Function ----------------
def resize_to_target(data: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Resize the image to match the target dimensions."""
    return np.resize(data, target_shape)

# ---------------- Main Process ----------------
for chr_id in chromosomes:
    full_path = os.path.join(full_dir, f'GSE131811_mat_full_chr_{chr_id}_{res}.npy')

    try:
        full = np.load(full_path)
    except Exception as e:
        print(f"[Skip] Failed to load target matrix for Chr {chr_id}: {e}")
        continue

    # ==== Only plot the top-left window (approx. 40kb size) ====
    crop_bins = crop_bin  # Approx. 40x40 visual window
    full = full[:min(crop_bins, full.shape[0]), :min(crop_bins, full.shape[1])]

    # Load downsampled data
    downsampled_path = os.path.join(full_dir, f'GSE131811_mat_{percent}_chr_{chr_id}_{res}.npy')
    try:
        downsampled = np.load(downsampled_path)
    except Exception as e:
        print(f"[Skip] Failed to load downsampled data for Chr {chr_id}: {e}")
        downsampled = np.zeros_like(full)  # Fill with zeros if loading fails

    # Normalize
    downsampled = normalize(downsampled)

    # Ensure downsampled size matches the full image
    downsampled = downsampled[:crop_bins, :crop_bins]

    # Create a figure to merge predictions and ground truth
    fig, axes = plt.subplots(1, len(model_ids) + 2, figsize=(20, 6))
    fig.suptitle(f"Comparison of Predictions vs Ground Truth (Chr {chr_id})", fontsize=12)

    # Plot downsampled image (first one)
    axes[0].imshow(downsampled, cmap="Reds", vmin=0, vmax=1)
    axes[0].set_title("Downsampled")
    axes[0].axis('off')

    # Iterate through models and plot predictions
    for i, model_id in enumerate(model_ids):
        pred_path = find_latest_pred(pred_root, model_id, chr_id, res, block_size)
        if not pred_path:
            print(f"[Skip] Prediction file not found for {model_id} / Chr {chr_id}.")
            continue

        try:
            pred = np.load(pred_path)
        except Exception as e:
            print(f"[Skip] Failed to load prediction for {model_id} / Chr {chr_id}: {e}")
            continue

        # ==== Also only plot the top-left window ====
        pred = pred[:min(crop_bins, pred.shape[0]), :min(crop_bins, pred.shape[1])]

        # Normalize
        pred = normalize(pred)

        # Plot prediction
        ax = axes[i + 1]
        model_titles = {
            # "hicsrNet": "HiCSR",
            # "hiedsrgan": "ScHiCEDRN",
            # "deephic": "DeepHiC",
            # "schicatt": "ScHiCAtt",
            "ScHiCNet": "ScHiCNet"
        }
        # Use default name if model_id is not in dictionary
        title = model_titles.get(model_id, model_id)
        
        ax.imshow(pred, cmap="Reds", vmin=0, vmax=1)
        ax.set_title(f"Prediction ({title})")
        ax.axis('off')

    # Plot target image (last one)
    full = normalize(full)
    axes[-1].imshow(full, cmap="Reds", vmin=0, vmax=1)
    axes[-1].set_title("Target (Ground Truth)")
    axes[-1].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_fig:
        save_dir = os.path.join(project_root, 'All_models', f'{cell_line}_cell{cell_no}_{percent}')
        os.makedirs(save_dir, exist_ok=True)
        fig_name = f"chr{chr_id}_{res}_piece{block_size}_all_models.png"
        fig_path = os.path.join(save_dir, fig_name)
        plt.savefig(fig_path, dpi=dpi)
        print(f"[Saved] Chr {chr_id} -> {fig_path}")
        plt.close(fig)
    else:
        plt.show()
