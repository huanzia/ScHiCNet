import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# ---------------- 配置区 ----------------

chromosomes = [2,6,10,12]  # 只处理染色体 6
res = 40000
percent = 0.75
cell_no = 1
crop_bin = 40
cell_line = 'Human'

# base_dir = f'/home/Work_Project/ScHiCNet/Training/DataFull_{cell_line}_cell{cell_no}_{percent}_{res}'
base_dir = f'./ScHiCNet/Training/DataFull_{cell_line}_cell{cell_no}_{percent}_{res}'
full_dir = os.path.join(base_dir, 'Full_Mats')
pred_root = os.path.join(base_dir, 'Preds')

# 要对比的模型（需与预测时的 model_id 一致）
model_ids = [
    # "hicsrNet",
    # "hiedsrgan",
    # "deephic",
    # "schicatt",
    "ScHiCNet",
]

# 预测时使用的块大小 block_size（命名里会用到）
block_size = 40

# 是否将图片保存到文件；保存目录：<base_dir>/Plots/<model_id>/
save_fig = True
dpi = 600

# ---------------- 工具函数 ----------------
def find_latest_pred(pred_root_dir: str, model_id: str, chr_id: int, reso: int, block: int) -> str:
    """
    根据命名模式在 Preds/<model_id>/ 下寻找当前染色体的预测文件，
    若有多个候选，返回修改时间最新的一个；找不到则返回空字符串。
    目标模式类似：
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

        legacy = os.path.join(pred_root_dir, f"GSE131811_full_chr_{chr_id}_{reso}_piece_predict{block}.npy")
        return legacy if os.path.isfile(legacy) else ""

    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]

# ---------------- 归一化函数 ----------------
def normalize(data: np.ndarray) -> np.ndarray:
    """将数据归一化到 [0, 1] 范围"""
    return (data - np.min(data)) / (np.max(data) - np.min(data))

# ---------------- 调整尺寸函数 ----------------
def resize_to_target(data: np.ndarray, target_shape: tuple) -> np.ndarray:
    """调整图像的尺寸以匹配目标图像的尺寸"""
    return np.resize(data, target_shape)

# ---------------- 主流程 ----------------
for chr_id in chromosomes:
    full_path = os.path.join(full_dir, f'GSE131811_mat_full_chr_{chr_id}_{res}.npy')

    try:
        full = np.load(full_path)
    except Exception as e:
        print(f"[跳过] Chr {chr_id} 目标整图加载失败：{e}")
        continue

    # ==== 只绘制约 40kb 大小的左上角窗口（其余逻辑不变） ====
    crop_bins = crop_bin  # 约 40×40 的可视窗口
    full = full[:min(crop_bins, full.shape[0]), :min(crop_bins, full.shape[1])]

    # 读取下采样数据
    downsampled_path = os.path.join(full_dir, f'GSE131811_mat_{percent}_chr_{chr_id}_{res}.npy')
    try:
        downsampled = np.load(downsampled_path)
    except Exception as e:
        print(f"[跳过] Chr {chr_id} 下采样数据加载失败：{e}")
        downsampled = np.zeros_like(full)  # 若加载失败，使用全零矩阵填充

    # 归一化处理
    downsampled = normalize(downsampled)

    # 确保 downsampled 的尺寸和 full 图像一致
    downsampled = downsampled[:crop_bins, :crop_bins]

    # 创建一个图形来合并所有模型的预测与目标
    fig, axes = plt.subplots(1, len(model_ids) + 2, figsize=(20, 6))  # 增加了图像大小
    fig.suptitle(f"Comparison of Predictions vs Ground Truth (Chr {chr_id})", fontsize=12)

    # 绘制下采样图（第一个图）
    axes[0].imshow(downsampled, cmap="Reds",vmin=0, vmax=1)  # 使用不同的色图
    axes[0].set_title("Downsampled")
    axes[0].axis('off')

    # 遍历每个模型，绘制每个模型的预测结果
    for i, model_id in enumerate(model_ids):
        pred_path = find_latest_pred(pred_root, model_id, chr_id, res, block_size)
        if not pred_path:
            print(f"[跳过] {model_id} / Chr {chr_id} 未找到预测文件（按约定命名）。")
            continue

        try:
            pred = np.load(pred_path)
        except Exception as e:
            print(f"[跳过] {model_id} / Chr {chr_id} 预测加载失败：{e}")
            continue

        # ==== 同样只绘制约 40kb 大小的左上角窗口 ====
        pred = pred[:min(crop_bins, pred.shape[0]), :min(crop_bins, pred.shape[1])]

        # 归一化处理
        pred = normalize(pred)

        # 绘制预测图
        ax = axes[i + 1]
        model_titles = {
            # "hicsrNet": "HiCSR",
            # "hiedsrgan": "ScHiCEDRN",
            # "deephic": "DeepHiC",
            # "schicatt": "ScHiCAtt",
            "ScHiCNet": "ScHiCNet"
        }
        ax.imshow(pred, cmap="Reds",vmin=0, vmax=1)
        ax.set_title(f"Prediction ({model_titles[model_id]})")
        ax.axis('off')

    # 绘制目标图（最后一个图）
    full = normalize(full)  # 对目标图也进行归一化处理
    axes[-1].imshow(full, cmap="Reds",vmin=0, vmax=1)
    axes[-1].set_title("Target (Ground Truth)")
    axes[-1].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_fig:
        # save_dir = f"/home/Work_Project/ScHiCNet/All_models/{cell_line}_cell{cell_no}_{percent}"
        save_dir = f'./ScHiCNet/All_models/{cell_line}_cell{cell_no}_{percent}'
        os.makedirs(save_dir, exist_ok=True)
        fig_name = f"chr{chr_id}_{res}_piece{block_size}_all_models.png"
        fig_path = os.path.join(save_dir, fig_name)
        plt.savefig(fig_path, dpi=dpi)
        print(f"[保存] Chr {chr_id} → {fig_path}")
        plt.close(fig)
    else:
        plt.show()

