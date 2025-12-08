# -*- coding: utf-8 -*-
import os
import sys
import subprocess
import torch
import numpy as np
from tqdm import tqdm
import shutil
import gzip
import torch.nn.functional as F

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from ProcessData.PrepareData_tensor import GSE131811Module
from ProcessData.PrepareData_tensorH import GSE130711Module
from ProcessData.PrepareData_tensorMouse import GSE162511Module

# import Models.schicedrn_gan as schicedrn
# import Models.deephic as deephic
# import Models.hicsr as hicsr
# import Models.ScHiCAtt as ScHiCAtt
import Models.schicnet as schicnet

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

CELL_LIN = "Mouse"
PERCENTAGE = 0.1
CELL_NOS_TO_PROCESS = [1]
RES = 40000
PIECE_SIZE = 40
SAMPLES_TO_PROCESS = 101
PRETRAINED_CELL_LINT = "Mouse"
PRETRAINED_CELL_NOT = 1

chros_all = [2, 6, 10, 12] if CELL_LIN == "Human" or CELL_LIN == "Mouse" else [2, 6]

print("Loading all baseline models...")
# Use relative path for pretrained models
weights_dir = os.path.join(project_root, 'pretrained')
file_inter = f"Downsample_{PERCENTAGE}_{PRETRAINED_CELL_LINT}{PRETRAINED_CELL_NOT}"

# schicnet
schicnet_model = schicnet.schicnet_Block().to(device)
schicnet_path = os.path.join(weights_dir, file_inter, f"bestg_40kb_c40_s40_{PRETRAINED_CELL_LINT}{PRETRAINED_CELL_NOT}_schicnetNet.pth")
schicnet_model.load_state_dict(torch.load(schicnet_path, map_location=device))
schicnet_model.eval()

models = {
    # 'ScHiCEDRN': schicedrn_model,
    # 'DeepHiC': deephic_model,
    # 'HiCSR': hicsr_model,
    # 'ScHiCAtt': schicatt_model,
    'schicnet': schicnet_model
}

file_prefixes = ['original', 'down'] + list(models.keys())
print("All models loaded successfully.")

def HicRepInput_Optimized(file_path, resolution):
    try:
        contact_map = np.loadtxt(file_path)
    except (IOError, ValueError):
        return np.array([[]], dtype=int)
    if contact_map.ndim == 1: contact_map = contact_map.reshape(1, -1)
    if contact_map.shape[0] == 0: return np.array([[]], dtype=int)
    rows, cols, vals = (contact_map[:, 0] / resolution).astype(int), (contact_map[:, 1] / resolution).astype(
        int), contact_map[:, 2]
    min_bin, max_bin = min(rows.min(), cols.min()), max(rows.max(), cols.max())
    dim = max_bin - min_bin + 1
    matrix = np.zeros((dim, dim), dtype=vals.dtype)
    rows_offset, cols_offset = rows - min_bin, cols - min_bin
    np.add.at(matrix, (rows_offset, cols_offset), vals)
    np.add.at(matrix, (cols_offset, rows_offset), vals)
    diag_mask = (rows_offset == cols_offset)
    if np.any(diag_mask): np.add.at(matrix, (rows_offset[diag_mask], cols_offset[diag_mask]), -vals[diag_mask])
    return matrix.astype(int)

for cell_no in CELL_NOS_TO_PROCESS:
    print(f"\n\n<<<<<<<<<< Generating files for Cell No: {cell_no} >>>>>>>>>>")

    # Use relative paths for inputs/outputs
    hicqc_dir = os.path.join(project_root, 'hicqc_inputs')
    hicrep_dir = os.path.join(project_root, 'hicRep_inputs')
    
    root_dir = os.path.join(hicqc_dir, f"{CELL_LIN}{cell_no}_{PERCENTAGE}_part100")
    root_dirR = os.path.join(hicrep_dir, f"{CELL_LIN}{cell_no}_{PERCENTAGE}_part100")

    for CHRO in chros_all:
        print(f"\n----- Processing Chromosome {CHRO} for Cell {cell_no} -----")
        os.makedirs(root_dir, exist_ok=True)
        os.makedirs(root_dirR, exist_ok=True)

        file_handlers = {}
        for prefix in file_prefixes:
            file_handlers[f'{prefix}_hic'] = open(f"{root_dir}/{prefix}_{CHRO}", 'w')
            file_handlers[f'{prefix}_hicRep'] = open(f"{root_dirR}/{prefix}_{CHRO}.txt", 'w')
        bins_file_path = f"{root_dir}/bins_{CHRO}.bed"
        file_handlers['bins'] = open(bins_file_path, 'w')

        dm_test = GSE162511Module(batch_size=1, percent=PERCENTAGE,
                                  cell_No=cell_no) if CELL_LIN == "Mouse" else GSE131811Module(batch_size=1,
                                                                                               percent=PERCENTAGE,
                                                                                               cell_No=cell_no)
        dm_test.prepare_data()
        dm_test.setup(stage=CHRO)

        test_bar = tqdm(dm_test.test_dataloader(), desc=f"Cell {cell_no}, Chr{CHRO}")
        with torch.no_grad():
            for s, sample in enumerate(test_bar):
                if s >= SAMPLES_TO_PROCESS: break

                data, target, _ = sample
                data, target = data.to(device), target.to(device)

                if target.sum() == 0: continue

                # --- Run all models ---
                outputs = {'down': data[0][0], 'target': target[0][0]}
                for name, model in models.items():
                    if name in ['HiCSR']:
                        padded_data = F.pad(data, (6, 6, 6, 6), mode='constant')
                        outputs[name] = model(padded_data).cpu()[0][0][6:-6, 6:-6]
                    else:
                        outputs[name] = model(data).cpu()[0][0]

                # --- Write to files ---
                for i in range(PIECE_SIZE):
                    bina = (PIECE_SIZE * s * RES) + (i * RES)
                    file_handlers['bins'].write(f"{CHRO}\t{bina}\t{bina + RES}\t{bina}\n")
                    for j in range(i, PIECE_SIZE):
                        binb = (PIECE_SIZE * s * RES) + (j * RES)

                        for prefix in file_prefixes:
                            matrix = outputs.get(prefix, outputs.get('target'))  # 'original' uses 'target' data

                            # Apply processing logic
                            val = max(0, int(matrix[i, j].item() * 100))

                            if val > 0:
                                file_handlers[f'{prefix}_hic'].write(f"{CHRO}\t{bina}\t{CHRO}\t{binb}\t{val}\n")
                                file_handlers[f'{prefix}_hicRep'].write(f"{bina}\t{binb}\t{val}\n")

        # --- Close all files ---
        for handler in file_handlers.values():
            handler.close()

        # --- Post-processing ---
        print(f"Post-processing files for Cell {cell_no}, Chr {CHRO}...")
        for prefix in file_prefixes:
            subprocess.run(["gzip", "-f", f"{root_dir}/{prefix}_{CHRO}"], check=True)

            data_matrix = HicRepInput_Optimized(f"{root_dirR}/{prefix}_{CHRO}.txt", RES)
            if data_matrix.size > 0:
                np.savetxt(f"{root_dirR}/{prefix}_{CHRO}.matrix", data_matrix, fmt='%d')

        with open(bins_file_path, 'rb') as f_in, gzip.open(f"{bins_file_path}.gz", 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
        os.remove(bins_file_path)

        # Updated relative path for BASE_STR
        # This writes paths into a file that downstream tools might read.
        # Ensure that whatever reads this can handle relative paths, or use os.path.abspath if needed.
        # Here we construct a relative path assuming execution from project root.
        BASE_STR = f"../hicqc_inputs/{CELL_LIN}{cell_no}_{PERCENTAGE}_part100/"
        
        for model_name in models.keys():
            with open(f"{root_dir}/metric_{model_name}_{CHRO}.samples", 'w') as f:
                f.write(f"original\t{BASE_STR}original_{CHRO}.gz\n")
                f.write(f"{model_name}\t{BASE_STR}{model_name}_{CHRO}.gz\n")

            with open(f"{root_dir}/metric_{model_name}_{CHRO}.pairs", 'w') as f:
                f.write(f"original\t{model_name}")

print("\n\nAll batch file generation jobs finished!")
