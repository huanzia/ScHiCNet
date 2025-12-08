import os
import math
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import Models.schicnet as schicnet
# import Models.schicedrn_gan as hiedsr
# import Models.hicsr as hicsr
# import Models.deephic as deephic
# import Models.hicplus as hicplus
# import Models.ScHiCAtt as ScHiCAtt

# ---------------- Configuration ----------------
cell_lin = "Mouse" 
model_cell_line = "Human"
cell_not = 1
percent = 0.75 
file_inter = f'Downsample_{percent}_{cell_lin}{cell_not}/'
plot_cell_not = 7
plot_file_inter = f'Downsample_{percent}_{model_cell_line}{plot_cell_not}/'

# ===================== Public Tools =====================
def load_weights(model: torch.nn.Module, weights_path: str, device: torch.device):
    ckpt = torch.load(weights_path, map_location=device)
    state = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model


def predict_on_splits(model, split_path, device, batch_size=64):
    """
    Run prediction on split data.
    Supports input shapes (N,H,W) or (N,1,H,W).
    """
    data_np = np.load(split_path)  # (N,C,H,W) or (N,H,W)
    data_tensor = torch.from_numpy(data_np).float().to(device)
    if data_tensor.ndim == 3:
        data_tensor = data_tensor.unsqueeze(1)  # -> (N,1,H,W)
    dataset = TensorDataset(data_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    preds = []
    model.eval()
    with torch.no_grad():
        for (x,) in loader:
            y = model(x)
            y = y.detach().cpu().numpy()
            preds.append(y)
    return np.concatenate(preds, axis=0)


def _make_weight_window(P: int, mode: str = "hann") -> np.ndarray:
    if mode == "uniform":
        W = np.ones((P, P), dtype=np.float32)
    elif mode == "hann":
        w = np.hanning(P).astype(np.float32)
        W = np.outer(w, w)
    elif mode == "bartlett":
        w = np.bartlett(P).astype(np.float32)
        W = np.outer(w, w)
    elif mode == "gaussian":
        x = np.linspace(-1, 1, P, dtype=np.float32)
        sigma = 1 / 3
        g = np.exp(-0.5 * (x / sigma) ** 2).astype(np.float32)
        W = np.outer(g, g)
    else:
        raise ValueError(f"Unknown weight mode: {mode}")
    m = W.max()
    if not np.isfinite(m) or m <= 0:
        return np.ones((P, P), dtype=np.float32)
    return (W / m).astype(np.float32)


def _infer_stride_from_blocks(H, W, out_h, out_w, N, prefer_leq=None):
    if N <= 0:
        raise ValueError("N (num blocks) must be positive")
    candidates = []
  
    for n_rows in range(1, int(np.sqrt(N)) + 1):
        if N % n_rows != 0:
            continue
        n_cols = N // n_rows

        # Reverse infer row/col stride candidates
        def infer_stride(L, k, out_k):
            if k == 1:
                return max(1, L)  # Treat the single block as starting at origin
            s = (L - out_k) / float(k - 1)
            s_floor = max(1, int(np.floor(s)))
            calc_k = (L - out_k) // s_floor + 1
            return s_floor if calc_k == k else None

        sr = infer_stride(H, n_rows, out_h)
        sc = infer_stride(W, n_cols, out_w)
        if sr is None or sc is None:
            continue
        if sr != sc:
            continue
        stride = sr
        calc_rows = (H - out_h) // stride + 1
        calc_cols = (W - out_w) // stride + 1
        if calc_rows == n_rows and calc_cols == n_cols:
            candidates.append(stride)

    if not candidates:
        return None

    if prefer_leq is not None:
        leq = [s for s in candidates if s <= prefer_leq]
        if leq:
            return max(leq)
        return min(candidates, key=lambda s: abs(s - prefer_leq))
    return min(candidates)  # Conservatively take the smallest valid stride


def combine_blocks_overlap(
        blocks: np.ndarray,
        full_shape: tuple,
        block_size: int = 40,
        out_shrink: int = 0,
        weight_mode: str = "hann",
        symmetrize: bool = True,
        eps: float = 1e-6,
) -> np.ndarray:
    H, W = int(full_shape[0]), int(full_shape[1])
    P = int(block_size)
    out_h = P - int(out_shrink)
    out_w = P - int(out_shrink)

    if out_h <= 0 or out_w <= 0:
        raise ValueError(f"Invalid out_shrink={out_shrink} for block_size={block_size}")

    # (N, h, w)
    if blocks.ndim == 4 and blocks.shape[1] == 1:
        blocks = blocks[:, 0, :, :]
    elif blocks.ndim == 4:
        blocks = blocks[:, 0, :, :]
    elif blocks.ndim != 3:
        raise ValueError(f"blocks ndim must be 3 or 4, got {blocks.ndim}")

    N = blocks.shape[0]
    Nh, Nw = blocks.shape[-2], blocks.shape[-1]

    if Nh != out_h or Nw != out_w:
        out_h, out_w = Nh, Nw

    stride = _infer_stride_from_blocks(H, W, out_h, out_w, N, prefer_leq=block_size)
    if stride is None:
        print(f"[WARN] Cannot infer stride from N={N}, fallback to stride={out_h} (may differ from original patches)")
        stride = out_h
    n_rows = (H - out_h) // stride + 1
    n_cols = (W - out_w) // stride + 1
    expected_N = n_rows * n_cols
    if expected_N != N:
        print(
            f"[INFO] stride={stride} infer {n_rows}x{n_cols}={expected_N} and N={N} is different, taking min(N, expected_N)")

    Wwin = _make_weight_window(out_h, mode=weight_mode).astype(np.float32)
    if out_w != out_h:
        Wx = _make_weight_window(out_w, mode=weight_mode).astype(np.float32)
        Wwin = np.sqrt(np.outer(Wwin[:, 0], Wx[0, :])).astype(np.float32)

    canvas = np.zeros((H, W), dtype=np.float32)
    count = np.zeros((H, W), dtype=np.float32)

    idx = 0
    for r in range(n_rows):
        for c in range(n_cols):
            if idx >= N:
                break
            i = r * stride
            j = c * stride
            i_end = min(i + out_h, H)
            j_end = min(j + out_w, W)
            bh, bw = i_end - i, j_end - j
            if bh <= 0 or bw <= 0:
                idx += 1
                continue
            blk = blocks[idx][:bh, :bw]
            Ww = Wwin[:bh, :bw]
            canvas[i:i_end, j:j_end] += (blk * Ww)
            count[i:i_end, j:j_end] += Ww
            idx += 1

    mask = count > 0
    canvas[mask] = canvas[mask] / (count[mask] + eps)

    if symmetrize and H == W:
        canvas = 0.5 * (canvas + canvas.T)
    return canvas


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def make_save_path(data_root: str, model_id: str, weights_path: str, split_path: str,
                   block_size: int, out_dir_name: str = "Preds") -> str:
    split_stem = stem(split_path)
    weights_stem = stem(weights_path) if weights_path else "noweights"
    out_dir = os.path.join(data_root, out_dir_name, model_id)
    ensure_dir(out_dir)
    filename = f"{split_stem}_predict-{model_id}_bs{block_size}_w{weights_stem}.npy"
    return os.path.join(out_dir, filename)

# Dynamic path resolution
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
DATA_ROOT_DEFAULT = os.path.join(project_root, 'Training', f'DataFull_{cell_lin}_cell{cell_not}_{percent}_40000')

SPLIT_FMT_DEFAULT = "{root}/Splits/GSE162511_full_chr_{chr_id}_40000_piece_{block}.npy"
FULL_FMT_DEFAULT = "{root}/Full_Mats/GSE162511_mat_full_chr_{chr_id}_40000.npy"

# CHROMS_DEFAULT = [1,2,3,4,5,6,10,12]
CHROMS_DEFAULT = [2, 6, 10, 12]
BLOCK_SIZE_DEFAULT = 40
BATCH_SIZE_DEFAULT = 64

# Weight path resolution
weights_dir = os.path.join(project_root, 'pretrained', plot_file_inter)

MODEL_REGISTRY = [
    # ===== ScHiCNet =====
    {
        "model_id": "schicnet",
        "build_model": lambda device: schicnet.schicnet_Block().to(device),
        "weights_path": os.path.join(weights_dir, f"bestg_40kb_c40_s40_{model_cell_line}{plot_cell_not}_schicnetNet.pth"),
        "data_root": DATA_ROOT_DEFAULT,
        "block_size": BLOCK_SIZE_DEFAULT,
        "batch_size": BATCH_SIZE_DEFAULT,
        "chromosomes": CHROMS_DEFAULT,
        "split_npy_fmt": SPLIT_FMT_DEFAULT,
        "full_mat_fmt": FULL_FMT_DEFAULT,
        "out_shrink": 0,  # Hout = Hin
    },
]


# ===================== Main =====================
def run_one_model(cfg, device):
    model_id = cfg["model_id"]
    data_root = cfg["data_root"]
    weights_path = cfg.get("weights_path")
    block_size = int(cfg.get("block_size", 40))
    batch_size = int(cfg.get("batch_size", 64))
    chroms = list(cfg.get("chromosomes", []))
    out_shrink = int(cfg.get("out_shrink", 0))

    split_fmt = cfg["split_npy_fmt"]
    full_fmt = cfg["full_mat_fmt"]

    print(f"\n==== [{model_id}] Loading ====")
    model = cfg["build_model"](device)
    if weights_path and os.path.isfile(weights_path):
        print(f"[{model_id}] Loading weights: {weights_path}")
        load_weights(model, weights_path, device)
    else:
        print(f"[{model_id}] Weights file not found: {weights_path}")

    for chr_id in chroms:
        full_mat_path = full_fmt.format(root=data_root, chr_id=chr_id)
        split_path = split_fmt.format(root=data_root, chr_id=chr_id, block=block_size)

        if not os.path.isfile(full_mat_path):
            print(f"[{model_id}] [Chr{chr_id}] Missing Full_Mat: {full_mat_path} (Skipping)")
            continue
        if not os.path.isfile(split_path):
            print(f"[{model_id}] [Chr{chr_id}] Missing Splits: {split_path} (Skipping)")
            continue

        full_shape = np.load(full_mat_path, mmap_mode='r').shape
        print(f"[{model_id}] [Chr{chr_id}] full_shape={full_shape}, split={split_path}, out_shrink={out_shrink}")

        preds_np = predict_on_splits(model, split_path, device, batch_size=batch_size)
        full_pred = combine_blocks_overlap(
            preds_np, full_shape=full_shape, block_size=block_size, out_shrink=out_shrink
        )

        save_path = make_save_path(
            data_root=data_root, model_id=model_id, weights_path=weights_path,
            split_path=split_path, block_size=block_size, out_dir_name="Preds"
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, full_pred)
        print(f"[✔] [{model_id}] Chr{chr_id} prediction finished -> {save_path}")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    for cfg in MODEL_REGISTRY:
        try:
            run_one_model(cfg, device)
        except Exception as e:
            print(f"[!] Model {cfg.get('model_id')} failed: {e}")


if __name__ == "__main__":
    main()
