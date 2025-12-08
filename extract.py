import numpy as np
import pandas as pd
import cooler
import subprocess
import os
import shutil
import torch

def extract_constraint_mats(resolution):
    """
    Extract contact matrices for specified resolution from .mcool files.
    """
    # Clean up and recreate the directory for storing constraints
    if os.path.exists("DataFull/Constraints"):
        shutil.rmtree("DataFull/Constraints")  # Removes directory and all contents
    os.makedirs("DataFull/Constraints")  # Recursively creates directories

    # IMPORTANT: Update this path to match your data location or pass as argument
    filepath = '../Datasets/Drosophila/GSM3820057_Cell1.10000.mcool'
    
    if not os.path.exists(filepath):
        print(f"Error: Input file not found at {filepath}")
        return

    # List all available resolutions in the file
    # AllRes = cooler.fileops.list_coolers(filepath)
    # print(AllRes)

    res = resolution
    c = cooler.Cooler(filepath + '::resolutions/' + str(res))
    c1 = c.chroms()[:]  # Chromosome size information
    print(f"Processing: {c1.loc[0, 'name']}, Indices: {c1.index}")

    for i in c1.index:
        print(i, c1.loc[i, 'name'])
        chro = c1.loc[i, 'name']
        
        # Fetch the matrix for the specific chromosome
        c2 = c.matrix(balance=True, as_pixels=True, join=True).fetch(chro)
        
        # Extract start positions and balanced values
        c2 = c2[['start1', 'start2', 'balanced']]
        
        # Fill NaN values with 0
        c2.fillna(0, inplace=True)
        
        # Save to file (skipping specific index if needed, logic preserved from original)
        if i == 6:
            pass
        else:
            output_file = 'DataFull/Constraints/chrom_' + str(i+1) + '_' + str(res) + '.txt'
            c2.to_csv(output_file, sep='\t', index=False, header=False)

if __name__ == '__main__':
    # Example usage:
    # extract_constraint_mats(40000)
    pass
