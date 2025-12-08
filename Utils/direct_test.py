import numpy as np
import gzip
import scipy.sparse as sps
import sys
import os

# Use relative path to add GenomeDISCO to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
# Assuming GenomeDISCO.py is in the same folder (Utils), no append needed for import if running as module.
# But if you need to add it specifically:
# sys.path.append(current_dir)

from GenomeDISCO import compute_reproducibility, to_transition


def load_hic_data(filepath, resolution):
    """A simple function to load data from gz files and create a matrix."""
    coords = []
    vals = []
    with gzip.open(filepath, 'rt') as f:
        for line in f:
            parts = line.strip().split()
            # bedpe-like format: chr1 pos1 chr2 pos2 val
            p1, p2, val = int(parts[1]), int(parts[3]), float(parts[4])
            coords.append((p1 // resolution, p2 // resolution))
            vals.append(val)

    if not coords:
        return sps.csr_matrix((1, 1))

    coords = np.array(coords)
    vals = np.array(vals)

    # Create symmetric matrix
    all_coords = np.concatenate([coords, coords[:, ::-1]], axis=0)
    all_vals = np.concatenate([vals, vals], axis=0)

    # Determine dimensions
    min_bin = all_coords.min()
    max_bin = all_coords.max()
    dim = max_bin - min_bin + 1

    # Offset coordinates and create sparse matrix
    rows = all_coords[:, 0] - min_bin
    cols = all_coords[:, 1] - min_bin

    matrix = sps.csr_matrix((all_vals, (rows, cols)), shape=(dim, dim))
    return matrix


# --- Main Diagnostic Logic ---
if __name__ == '__main__':
    RES = 40000

    # Determine project root from current script location
    project_root = os.path.dirname(current_dir)
    input_base = os.path.join(project_root, 'hicqc_inputs', 'Human7_0.75_part100')

    # Use relative paths
    original_file = os.path.join(input_base, 'original_6.gz')
    hiedsrgan_file = os.path.join(input_base, 'hiedsrgan_6.gz')

    print("Loading original data...")
    # Check if file exists before loading to avoid crash
    if not os.path.exists(original_file):
        print(f"Error: File not found: {original_file}")
        sys.exit(1)
        
    m_original = load_hic_data(original_file, RES)

    print("Loading hiedsrgan data...")
    if not os.path.exists(hiedsrgan_file):
        print(f"Error: File not found: {hiedsrgan_file}")
        sys.exit(1)

    m_hiedsrgan = load_hic_data(hiedsrgan_file, RES)

    # Ensure matrix dimensions match
    max_dim = max(m_original.shape[0], m_hiedsrgan.shape[0])
    m_original.resize((max_dim, max_dim))
    m_hiedsrgan.resize((max_dim, max_dim))

    print(f"Matrices loaded with shape: {m_original.shape}")

    # --- Apply preprocessing ---
    print("Applying log2(x+1) transformation...")
    m_original_log = m_original.log1p() / np.log(2)
    m_hiedsrgan_log = m_hiedsrgan.log1p() / np.log(2)

    # --- Call core algorithm directly ---
    print("Computing GenomeDISCO score directly...")
    # Note: transition=True performs the second normalization internally
    score = compute_reproducibility(m_original_log, m_hiedsrgan_log, transition=True, tmin=3, tmax=3)

    print("\n=====================================")
    print(f"  FINAL DIRECT SCORE: {score:.4f}")
    print("=====================================")
