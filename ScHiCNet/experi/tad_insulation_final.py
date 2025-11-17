import sys
import csv  # ✅ 1. 导入csv模块

sys.path.append(".")
sys.path.append("../")
from Utils.loss import insulation as ins
import os
import numpy as np
from numpy import inf
from tqdm import tqdm

import torch
import torch.nn.functional as F

# load data module
from ProcessData.PrepareData_tensor import GSE131811Module
from ProcessData.PrepareData_tensorH import GSE130711Module
from ProcessData.PrepareData_tensorMouse import GSE162511Module

# Load Models
# import Models.schicedrn_gan as schicedrn
# import Models.deephic as deephic
# import Models.hicsr as hicsr
# import Models.ScHiCAtt as ScHiCAtt
import Models.schicnet as schicnet

# --- 配置区 ---
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 绝缘分数计算模块
getIns = ins.computeInsulation().to(device)

# --- 测试参数设置 ---
RES = 40000
PIECE_SIZE = 40
CELL_LIN = "Dros"
PERCENTAGE = 0.45

CELL_NOS_TO_PROCESS = [1,2,3,4,5,6,7,8,9,10]

# --- 预训练模型路径参数 ---
PRETRAINED_CELL_LINT = "Human"
PRETRAINED_CELL_NOT = 1


PERCENTAGE_num = "0.45"

PRETRAINED_cell_no = 1

PRETRAINED_cell_lint = "Human"
model_file_inter = f"Downsample_{PERCENTAGE_num}_{PRETRAINED_cell_lint}{PRETRAINED_cell_no}/"
# 染色体配置
chros_all = [2, 6, 10, 12] if CELL_LIN == "Human" or CELL_LIN == "Mouse" else [2, 6]

# --- 加载所有模型 (一次性加载) ---
print("Loading all models...")
file_inter = f"Downsample_{PERCENTAGE}_{PRETRAINED_CELL_LINT}{PRETRAINED_CELL_NOT}/"

# schicedrn_model = schicedrn.Generator().to(device)
# schicedrn_path = f"../pretrained/{model_file_inter}bestg_40kb_c40_s40_{PRETRAINED_CELL_LINT}{PRETRAINED_CELL_NOT}_hiedsrgan.pytorch"
# schicedrn_model.load_state_dict(torch.load(schicedrn_path, weights_only=True))
# schicedrn_model.eval()
#
# # DeepHiC
# deephic_model = deephic.Generator(scale_factor=1, in_channel=1, resblock_num=5).to(device)
# deephic_path = f"../pretrained/{model_file_inter}bestg_40kb_c40_s40_{PRETRAINED_CELL_LINT}{PRETRAINED_CELL_NOT}_deephic.pytorch"
# deephic_model.load_state_dict(torch.load(deephic_path, weights_only=True))
# deephic_model.eval()
#
# # HiCSR
# hicsr_model = hicsr.Generator(num_res_blocks=15).to(device)
# hicsr_path = f"../pretrained/{model_file_inter}bestg_40kb_c40_s40_{PRETRAINED_CELL_LINT}{PRETRAINED_CELL_NOT}_hicsrNet.pytorch"
# hicsr_model.load_state_dict(torch.load(hicsr_path, weights_only=True))
# hicsr_model.eval()
#
# # ScHiCAtt
# schicatt_model = ScHiCAtt.ScHiCAtt().to(device)
# schicatt_path = f"../pretrained/{model_file_inter}bestg_40kb_c40_s40_{PRETRAINED_CELL_LINT}{PRETRAINED_CELL_NOT}_ScHiCAtt.pth"
# schicatt_model.load_state_dict(torch.load(schicatt_path, weights_only=True))
# schicatt_model.eval()

# schicnet
schicnet_model = schicnet.schicnet_Block().to(device)
schicnet_path = f"../pretrained/{model_file_inter}bestg_40kb_c40_s40_{PRETRAINED_CELL_LINT}{PRETRAINED_CELL_NOT}_schicnetNet.pth"
schicnet_model.load_state_dict(torch.load(schicnet_path, weights_only=True))
schicnet_model.eval()

# 将所有模型整合到一个字典中，方便统一调用
models = {
    # 'ScHiCEDRN': schicedrn_model,
    # 'DeepHiC': deephic_model,
    # 'HiCSR': hicsr_model,
    # 'ScHiCAtt': schicatt_model,
    'schicnet': schicnet_model
}
print("All models loaded successfully.")


def filterNum(arr1):
    """过滤计算中可能出现的 inf 和 nan 值"""
    arr1 = np.array(arr1)
    arr1[arr1 == inf] = 0
    arr1[arr1 == -inf] = 0
    arr2 = np.nan_to_num(arr1)
    return arr2



outdir = "../Insolation_score_results"
os.makedirs(outdir, exist_ok=True)
csv_outfile_path = f"{outdir}/summary_all_cells_{CELL_LIN}_{PERCENTAGE}.csv"
csv_file = open(csv_outfile_path, 'w', newline='')
csv_writer = csv.writer(csv_file)

# 写入CSV表头
csv_header = ['Cell_No'] + list(models.keys()) + ['Downsampled']
csv_writer.writerow(csv_header)

# --- 开始批量处理的外层循环 ---
for CELL_NO in CELL_NOS_TO_PROCESS:
    print(f"\n\n<<<<<<<<<< Starting Batch Job for Cell No: {CELL_NO} >>>>>>>>>>")

    final_scores = {model_name: [] for model_name in models.keys()}
    final_scores['Downsampled'] = []

    # --- 内层循环：按染色体进行评估 ---
    for CHRO in chros_all:
        print(f"\n----- Processing Chromosome {CHRO} for Cell {CELL_NO} -----")

        # dm_test = GSE130711Module(batch_size=1, percent=PERCENTAGE,
        #                           cell_No=CELL_NO) if CELL_LIN == "Human" elif GSE131811Module(batch_size=1,
        #                                                                                        percent=PERCENTAGE,
        #                                                                                        cell_No=CELL_NO)

        if CELL_LIN == "Human":
            dm_test = GSE130711Module(batch_size=1, percent=PERCENTAGE, cell_No=CELL_NO)
        elif CELL_LIN == "Dros":
            dm_test = GSE131811Module(batch_size=1, percent=PERCENTAGE, cell_No=CELL_NO)
        elif CELL_LIN == "Mouse":  # 请替换为实际的条件，比如 "Rat" 或其他
            dm_test = GSE162511Module(batch_size=1, percent=PERCENTAGE, cell_No=CELL_NO)
        else:
            # 可选：设置默认值或抛出异常
            raise ValueError(f"Unsupported CELL_LIN: {CELL_LIN}")
        dm_test.prepare_data()
        dm_test.setup(stage=CHRO)

        insulation_vectors = {model_name: [] for model_name in models.keys()}
        insulation_vectors['Downsampled'] = []
        insulation_vectors['Target'] = []

        test_bar = tqdm(dm_test.test_dataloader(), desc=f"Cell {CELL_NO}, Chr{CHRO} Inference")
        with torch.no_grad():
            for s, sample in enumerate(test_bar):
                data, target, _ = sample
                data, target = data.to(device), target.to(device)

                if target.sum() == 0 or data.sum() == 0:
                    continue

                insulation_vectors['Target'].extend(getIns.forward(target.reshape(1, 1, 40, 40))[1][0][0][:].tolist())
                insulation_vectors['Downsampled'].extend(
                    getIns.forward(data.reshape(1, 1, 40, 40))[1][0][0][:].tolist())

                for model_name, model in models.items():
                    if model_name in ['HiCSR']:
                        padded_data = F.pad(data, (6, 6, 6, 6), mode='constant')
                        output_large = model(padded_data).cpu()[0][0]
                        output = output_large[6:-6, 6:-6]
                    else:
                        output = model(data).cpu()[0][0]

                    insulation_vectors[model_name].extend(
                        getIns.forward(output.reshape(1, 1, 40, 40))[1][0][0][:].tolist())

        for key in insulation_vectors:
            insulation_vectors[key] = filterNum(insulation_vectors[key])

        print(f"------ Results for Cell {CELL_NO}, Chr: {CHRO} ------")
        target_vector = np.array(insulation_vectors['Target'])

        down_diff_norm = np.linalg.norm(np.array(insulation_vectors['Downsampled']) - target_vector)
        final_scores['Downsampled'].append(down_diff_norm)
        print(f"Downsampled L2 Norm Diff: {down_diff_norm:.4f}")

        for model_name in models.keys():
            model_diff_norm = np.linalg.norm(np.array(insulation_vectors[model_name]) - target_vector)
            final_scores[model_name].append(model_diff_norm)
            print(f"{model_name} L2 Norm Diff:   {model_diff_norm:.4f}")

    # --- 计算并输出当前细胞的最终平均结果 ---
    print("\n==============================================")
    print(f"  Final Average Results for {CELL_LIN} cell {CELL_NO} @ {PERCENTAGE * 100}% ")
    print("==============================================")
    print("Note: Lower scores are better.")
    sorted_scores_avg = sorted(final_scores.items(), key=lambda item: np.mean(item[1]))
    avg_scores_dict = {model_name: np.mean(scores) for model_name, scores in final_scores.items()}

    for model_name, avg_score in sorted_scores_avg:
        print(f"Average {model_name} L2 Norm Diff: {np.mean(avg_score):.4f}")
    print("==============================================")


    # 按照表头的顺序准备好要写入的一行数据
    csv_row = [CELL_NO] + [avg_scores_dict[model_name] for model_name in models.keys()] + [
        avg_scores_dict['Downsampled']]
    csv_writer.writerow(csv_row)
    print(f"Results for Cell {CELL_NO} have been appended to {csv_outfile_path}")

    # --- 将当前细胞的详细结果写入独立的.txt文件 ---
    outfile = f"{outdir}/comparison_{CELL_LIN}{CELL_NO}_{PERCENTAGE}.txt"
    with open(outfile, "w") as results_file:
        results_file.write(f"Results for {CELL_LIN} Cell {CELL_NO} at {PERCENTAGE * 100}% downsampling\n\n")
        results_file.write("Method\tAverage_L2_Norm_Difference\n")
        for model_name, avg_score in sorted_scores_avg:
            results_file.write(f"{model_name}\t{np.mean(avg_score):.4f}\n")

        results_file.write("\n\n--- Per-Chromosome Scores ---\n")
        header = "Chromosome\t" + "\t".join(final_scores.keys()) + "\n"
        results_file.write(header)
        for i, chro in enumerate(chros_all):
            scores_line = [f"{final_scores[model_name][i]:.4f}" for model_name in final_scores.keys()]
            results_file.write(f"{chro}\t" + "\t".join(scores_line) + "\n")
    print(f"Detailed results for Cell {CELL_NO} have been saved to: {outfile}")


csv_file.close()
print(f"\n\nAll batch jobs finished. Summary of all cells saved to: {csv_outfile_path}")