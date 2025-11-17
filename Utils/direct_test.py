import numpy as np
import gzip
import scipy.sparse as sps
import sys

sys.path.append('/home/Work_Project/ScHiCNet/Utils/GenomeDISCO.py')
from GenomeDISCO import compute_reproducibility, to_transition


def load_hic_data(filepath, resolution):
    """一个简单的函数，用于从您的gz文件中加载数据并创建矩阵。"""
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

    # 创建对称矩阵
    all_coords = np.concatenate([coords, coords[:, ::-1]], axis=0)
    all_vals = np.concatenate([vals, vals], axis=0)

    # 确定矩阵维度
    min_bin = all_coords.min()
    max_bin = all_coords.max()
    dim = max_bin - min_bin + 1

    # 坐标偏移并创建稀疏矩阵
    rows = all_coords[:, 0] - min_bin
    cols = all_coords[:, 1] - min_bin

    matrix = sps.csr_matrix((all_vals, (rows, cols)), shape=(dim, dim))
    return matrix


# --- 主诊断逻辑 ---
if __name__ == '__main__':
    RES = 40000

    # --- 请在这里填入您要比较的两个文件的绝对路径 ---
    original_file = '/home/Work_Project/ScHiCNet/hicqc_inputs/Human7_0.75_part100/original_6.gz'
    hiedsrgan_file = '/home/Work_Project/ScHiCNet/hicqc_inputs/Human7_0.75_part100/hiedsrgan_6.gz'

    print("Loading original data...")
    m_original = load_hic_data(original_file, RES)

    print("Loading hiedsrgan data...")
    m_hiedsrgan = load_hic_data(hiedsrgan_file, RES)

    # 确保两个矩阵维度一致
    max_dim = max(m_original.shape[0], m_hiedsrgan.shape[0])
    m_original.resize((max_dim, max_dim))
    m_hiedsrgan.resize((max_dim, max_dim))

    print(f"Matrices loaded with shape: {m_original.shape}")

    # --- 应用我们发现的“秘密”预处理 ---
    print("Applying log2(x+1) transformation...")
    m_original_log = m_original.log1p() / np.log(2)
    m_hiedsrgan_log = m_hiedsrgan.log1p() / np.log(2)

    # --- 直接调用核心算法 ---
    print("Computing GenomeDISCO score directly...")
    # 注意：这里的 transition=True 会在函数内部进行第二次归一化，这正是算法要求
    score = compute_reproducibility(m_original_log, m_hiedsrgan_log, transition=True, tmin=3, tmax=3)

    print("\n=====================================")
    print(f"  FINAL DIRECT SCORE: {score:.4f}")
    print("=====================================")