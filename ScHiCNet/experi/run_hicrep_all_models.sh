#!/bin/bash
CELL_NOS=(1)
CHRO_NOS=(2 6)
MODEL_NAMES=("ScHiCEDRN" "DeepHiC" "HiCSR" "ScHiCAtt" "SCHiCNet")

CELL_LIN="Dros"
PERCENTAGE="0.1"
BASE_INPUT_DIR="/home/Work_Project/ScHiCNet/hicqc_inputs"
BASE_OUTPUT_DIR="/home/Work_Project/ScHiCNet/hicqc_results_all"

SUMMARY_FILE="hicrep_all_models_summary_Dros_0.1.csv"

echo "Cell_No,Chromosome,Model,SCC_Score" > "$SUMMARY_FILE"
echo "Starting batch processing for HiCRep across all models..."


for CELL_NO in "${CELL_NOS[@]}"; do
    for CHRO_NO in "${CHRO_NOS[@]}"; do
        for MODEL_NAME in "${MODEL_NAMES[@]}"; do
            echo ""
            echo "------------------------------------------------------------"
            echo ">>> Processing: Cell ${CELL_NO}, Chr ${CHRO_NO}, Model ${MODEL_NAME}"
            echo "------------------------------------------------------------"
            INPUT_SUBDIR="${CELL_LIN}${CELL_NO}_${PERCENTAGE}_part100"

            SAMPLE_FILE="${BASE_INPUT_DIR}/${INPUT_SUBDIR}/metric_${MODEL_NAME}_${CHRO_NO}.samples"
            PAIRS_FILE="${BASE_INPUT_DIR}/${INPUT_SUBDIR}/metric_${MODEL_NAME}_${CHRO_NO}.pairs"
            BINS_FILE="${BASE_INPUT_DIR}/${INPUT_SUBDIR}/bins_${CHRO_NO}.bed.gz"

            if [ ! -f "$SAMPLE_FILE" ] || [ ! -f "$PAIRS_FILE" ]; then
                echo "Warning: Input files for ${MODEL_NAME} not found. Skipping."
                echo "${CELL_NO},${CHRO_NO},${MODEL_NAME},ERROR_INPUT_FILE_NOT_FOUND" >> "$SUMMARY_FILE"
                continue
            fi

            OUT_DIR="${BASE_OUTPUT_DIR}/hicrep_${CELL_LIN,,}${CELL_NO}_chr${CHRO_NO}_${MODEL_NAME}"

            # --- 3DChromatin_ReplicateQC ---
            3DChromatin_ReplicateQC run_all \
                --metadata_samples "$SAMPLE_FILE" \
                --metadata_pairs "$PAIRS_FILE" \
                --bins "$BINS_FILE" \
                --outdir "$OUT_DIR" \
                --methods HiCRep

            RESULT_FILE="${OUT_DIR}/results/reproducibility/HiCRep/original.vs.${MODEL_NAME}.txt"

            if [ -f "$RESULT_FILE" ]; then
                SCORE=$(cat "$RESULT_FILE")
                echo "Successfully found score: ${SCORE}"
                echo "${CELL_NO},${CHRO_NO},${MODEL_NAME},${SCORE}" >> "$SUMMARY_FILE"
            else
                echo "Error: Result file not found at ${RESULT_FILE}!"
                echo "${CELL_NO},${CHRO_NO},${MODEL_NAME},ERROR_RESULT_FILE_NOT_FOUND" >> "$SUMMARY_FILE"
            fi

        done
    done
done

echo ""
echo "============================================================"
echo "All batch jobs finished!"
echo "Summary of all scores has been saved to: ${SUMMARY_FILE}"
echo "============================================================"