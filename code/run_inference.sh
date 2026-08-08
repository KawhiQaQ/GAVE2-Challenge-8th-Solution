#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKUP_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SOLUTION="${SCRIPT_DIR}/gave2_solution"
BIOMARKER="${SCRIPT_DIR}/biomarker_tools/local_biomarker_eval.py"
MINIMA="${SCRIPT_DIR}/external/MINIMA"
MINIMA_WEIGHT="${BACKUP_ROOT}/weights/external/minima_loftr.ckpt"
DISC_WEIGHT="${BACKUP_ROOT}/weights/external/Model_DiscSeg_ORIGA.h5"
TASK1_WEIGHT="${BACKUP_ROOT}/weights/task1/segmentation_cfp.pt"
TASK2_WEIGHT="${BACKUP_ROOT}/weights/task2/segmentation_multimodal.pt"
CALIBER_WEIGHT="${BACKUP_ROOT}/weights/task3/caliber_expert.pt"
ARTERY_DENSITY_WEIGHT="${BACKUP_ROOT}/weights/task3/artery_density_expert.pt"
VEIN_PARENT_WEIGHT="${BACKUP_ROOT}/weights/task3/vein_density_parent.pt"
PHASE_GEOMETRY_WEIGHT="${BACKUP_ROOT}/weights/task3/phase_geometry_expert.pt"
VEIN_REFINER_WEIGHT="${BACKUP_ROOT}/weights/task3/vein_density_refiner.pt"
ARTERY_DENSITY_CALIBRATION="${BACKUP_ROOT}/configs/artery_density_calibration.json"

DATA_ROOT=""
WORK_ROOT=""
OUTPUT_ZIP=""
REGISTERED_FFA_ROOT=""
DISC_ROOT=""
ZONE_ROOT=""
PYTHON_BIN="python3"
DEVICE="cuda"

usage() {
    cat <<'EOF'
Usage:
  bash code/run_inference.sh \
    --data-root /path/to/GAVE2_private \
    --work-root /path/to/work \
    --output-zip /path/to/kawhi00.zip \
    [--registered-ffa-root /path/to/cache] \
    [--disc-root /path/to/disc_masks] \
    [--zone-root /path/to/zone_c_masks] \
    [--python /path/to/python] [--device cuda]

DATA_ROOT must contain validation/{images,masks,FFA_A,FFA_AV}.  A supplied
registered-FFA root must contain validation/{FFA_A,FFA_AV}; a supplied disc
root must contain validation/g_*.png; a supplied zone root points directly to
the directory containing g_*.png C-zone masks.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-root) DATA_ROOT="$2"; shift 2 ;;
        --work-root) WORK_ROOT="$2"; shift 2 ;;
        --output-zip) OUTPUT_ZIP="$2"; shift 2 ;;
        --registered-ffa-root) REGISTERED_FFA_ROOT="$2"; shift 2 ;;
        --disc-root) DISC_ROOT="$2"; shift 2 ;;
        --zone-root) ZONE_ROOT="$2"; shift 2 ;;
        --python) PYTHON_BIN="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "${DATA_ROOT}" || -z "${WORK_ROOT}" || -z "${OUTPUT_ZIP}" ]]; then
    usage >&2
    exit 2
fi
DATA_ROOT="$(cd "${DATA_ROOT}" && pwd)"
if [[ -e "${WORK_ROOT}" ]]; then
    echo "Refusing to overwrite work root: ${WORK_ROOT}" >&2
    exit 1
fi
if [[ -e "${OUTPUT_ZIP}" ]]; then
    echo "Refusing to overwrite output ZIP: ${OUTPUT_ZIP}" >&2
    exit 1
fi
mkdir -p "${WORK_ROOT}" "$(dirname "${OUTPUT_ZIP}")"
WORK_ROOT="$(cd "${WORK_ROOT}" && pwd)"
OUTPUT_ZIP="$(cd "$(dirname "${OUTPUT_ZIP}")" && pwd)/$(basename "${OUTPUT_ZIP}")"

for required in \
    "${DATA_ROOT}/validation/images" \
    "${DATA_ROOT}/validation/masks" \
    "${DATA_ROOT}/validation/FFA_A" \
    "${DATA_ROOT}/validation/FFA_AV" \
    "${TASK1_WEIGHT}" \
    "${TASK2_WEIGHT}" \
    "${CALIBER_WEIGHT}" \
    "${ARTERY_DENSITY_WEIGHT}" \
    "${VEIN_PARENT_WEIGHT}" \
    "${PHASE_GEOMETRY_WEIGHT}" \
    "${VEIN_REFINER_WEIGHT}" \
    "${MINIMA_WEIGHT}" \
    "${DISC_WEIGHT}" \
    "${ARTERY_DENSITY_CALIBRATION}"; do
    if [[ ! -e "${required}" ]]; then
        echo "Missing required VascFusion asset: ${required}" >&2
        exit 1
    fi
done

read -r START_INDEX CASE_COUNT < <(
    "${PYTHON_BIN}" - "${DATA_ROOT}/validation/images" <<'PY'
import re
import sys
from pathlib import Path

paths = sorted(Path(sys.argv[1]).glob("g_*.png"))
indices = []
for path in paths:
    match = re.fullmatch(r"g_(\d+)", path.stem)
    if match is None:
        raise SystemExit(f"Unexpected case name: {path.name}")
    indices.append(int(match.group(1)))
if not indices:
    raise SystemExit("No validation g_*.png cases found")
expected = list(range(indices[0], indices[0] + len(indices)))
if indices != expected:
    raise SystemExit("Case IDs must form one contiguous numeric range")
print(indices[0], len(indices))
PY
)

export PYTHONPATH="${SOLUTION}${PYTHONPATH:+:${PYTHONPATH}}"
export TORCH_HOME="${WORK_ROOT}/torch_cache"

if [[ -z "${REGISTERED_FFA_ROOT}" ]]; then
    REGISTERED_FFA_ROOT="${WORK_ROOT}/registered_ffa"
    "${PYTHON_BIN}" "${SOLUTION}/register_ffa_minima.py" \
        --data-root "${DATA_ROOT}" \
        --output-root "${REGISTERED_FFA_ROOT}" \
        --minima-root "${MINIMA}" \
        --checkpoint "${MINIMA_WEIGHT}" \
        --splits validation
else
    REGISTERED_FFA_ROOT="$(cd "${REGISTERED_FFA_ROOT}" && pwd)"
fi

if [[ -z "${DISC_ROOT}" ]]; then
    DISC_ROOT="${WORK_ROOT}/disc_masks"
    "${PYTHON_BIN}" "${SCRIPT_DIR}/biomarker_tools/local_optic_disc.py" \
        --data-root "${DATA_ROOT}" \
        --weights "${DISC_WEIGHT}" \
        --output-dir "${DISC_ROOT}" \
        --split validation \
        --device "${DEVICE}"
else
    DISC_ROOT="$(cd "${DISC_ROOT}" && pwd)"
fi

if [[ -z "${ZONE_ROOT}" ]]; then
    ZONE_ROOT="${WORK_ROOT}/zone_c_validation"
    "${PYTHON_BIN}" "${SOLUTION}/prepare_zone_c_masks.py" \
        --disc-dir "${DISC_ROOT}/validation" \
        --output-dir "${ZONE_ROOT}" \
        --report-output "${WORK_ROOT}/zone_c_report.json"
else
    ZONE_ROOT="$(cd "${ZONE_ROOT}" && pwd)"
fi

mkdir -p "${WORK_ROOT}/raw" "${WORK_ROOT}/biomarkers" "${WORK_ROOT}/build"

# Task1 is strictly CFP-only. Task2 uses the registered FFA cache.
"${PYTHON_BIN}" "${SOLUTION}/predict_segmentation.py" \
    --task 1 \
    --checkpoint "${TASK1_WEIGHT}" \
    --data-root "${DATA_ROOT}" \
    --split validation \
    --output-dir "${WORK_ROOT}/raw/segmentation_cfp" \
    --device "${DEVICE}"
"${PYTHON_BIN}" "${SOLUTION}/predict_segmentation.py" \
    --task 2 \
    --checkpoint "${TASK2_WEIGHT}" \
    --data-root "${DATA_ROOT}" \
    --ffa-root "${REGISTERED_FFA_ROOT}" \
    --split validation \
    --output-dir "${WORK_ROOT}/raw/segmentation_multimodal" \
    --device "${DEVICE}"

"${PYTHON_BIN}" "${SOLUTION}/postprocess_av.py" \
    --input-dir "${WORK_ROOT}/raw/segmentation_cfp" \
    --masks-dir "${DATA_ROOT}/validation/masks" \
    --output-dir "${WORK_ROOT}/build/Task1" \
    --radius 3
"${PYTHON_BIN}" "${SOLUTION}/postprocess_av.py" \
    --input-dir "${WORK_ROOT}/raw/segmentation_multimodal" \
    --masks-dir "${DATA_ROOT}/validation/masks" \
    --output-dir "${WORK_ROOT}/build/Task2" \
    --radius 3

# The caliber and artery-density experts preserve the original FFA geometry.
"${PYTHON_BIN}" "${SOLUTION}/predict_quantification.py" \
    --task 2 \
    --checkpoint "${CALIBER_WEIGHT}" \
    --data-root "${DATA_ROOT}" \
    --split validation \
    --output-dir "${WORK_ROOT}/raw/caliber" \
    --device "${DEVICE}"
"${PYTHON_BIN}" "${SOLUTION}/predict_quantification.py" \
    --task 2 \
    --checkpoint "${ARTERY_DENSITY_WEIGHT}" \
    --data-root "${DATA_ROOT}" \
    --split validation \
    --output-dir "${WORK_ROOT}/raw/artery_density" \
    --device "${DEVICE}"

# Phase geometry and the C-zone vein refiner use CFP-aligned FFA.
"${PYTHON_BIN}" "${SOLUTION}/predict_phase_geometry.py" \
    --checkpoint "${PHASE_GEOMETRY_WEIGHT}" \
    --data-root "${DATA_ROOT}" \
    --ffa-root "${REGISTERED_FFA_ROOT}" \
    --split validation \
    --output-dir "${WORK_ROOT}/raw/phase_geometry" \
    --device "${DEVICE}"
"${PYTHON_BIN}" "${SOLUTION}/predict_vein_density.py" \
    --checkpoint "${VEIN_REFINER_WEIGHT}" \
    --parent-checkpoint "${VEIN_PARENT_WEIGHT}" \
    --data-root "${DATA_ROOT}" \
    --ffa-root "${REGISTERED_FFA_ROOT}" \
    --zone-c-dir "${ZONE_ROOT}" \
    --split validation \
    --output-dir "${WORK_ROOT}/raw/vein_density" \
    --device "${DEVICE}"

"${PYTHON_BIN}" "${BIOMARKER}" \
    --data-root "${DATA_ROOT}" --disc-dir "${DISC_ROOT}/validation" \
    --split validation --source "${WORK_ROOT}/raw/caliber" \
    --threshold 0.5 --mask-policy vessel_argmax \
    --output-dir "${WORK_ROOT}/biomarkers/caliber"

"${PYTHON_BIN}" "${BIOMARKER}" \
    --data-root "${DATA_ROOT}" --disc-dir "${DISC_ROOT}/validation" \
    --split validation --source "${WORK_ROOT}/raw/artery_density" \
    --threshold 0.5 --mask-policy vessel_argmax \
    --output-dir "${WORK_ROOT}/biomarkers/artery_density_raw"
"${PYTHON_BIN}" "${BIOMARKER}" \
    --data-root "${DATA_ROOT}" --disc-dir "${DISC_ROOT}/validation" \
    --split validation --source "${WORK_ROOT}/raw/artery_density" \
    --threshold 0.5 --mask-policy independent_av \
    --output-dir "${WORK_ROOT}/biomarkers/artery_density_caliber"
"${PYTHON_BIN}" "${SOLUTION}/calibrate_density.py" \
    --base-dir "${WORK_ROOT}/biomarkers/artery_density_raw" \
    --caliber-dir "${WORK_ROOT}/biomarkers/artery_density_caliber" \
    --calibration "${ARTERY_DENSITY_CALIBRATION}" \
    --output-dir "${WORK_ROOT}/biomarkers/artery_density" \
    --report-output "${WORK_ROOT}/artery_density_assembly.json" \
    --start-index "${START_INDEX}" --count "${CASE_COUNT}"

"${PYTHON_BIN}" "${SOLUTION}/postprocess_branch_consistency.py" \
    --input-dir "${WORK_ROOT}/raw/caliber" \
    --masks-dir "${DATA_ROOT}/validation/masks" \
    --output-dir "${WORK_ROOT}/raw/artery_fd"
"${PYTHON_BIN}" "${BIOMARKER}" \
    --data-root "${DATA_ROOT}" --disc-dir "${DISC_ROOT}/validation" \
    --split validation --source "${WORK_ROOT}/raw/artery_fd" \
    --threshold 0.5 --mask-policy vessel_argmax \
    --output-dir "${WORK_ROOT}/biomarkers/artery_fd"

for threshold in 0.4 0.5 0.6; do
    suffix="${threshold/./}"
    "${PYTHON_BIN}" "${BIOMARKER}" \
        --data-root "${DATA_ROOT}" --disc-dir "${DISC_ROOT}/validation" \
        --split validation --source "${WORK_ROOT}/raw/phase_geometry" \
        --threshold "${threshold}" --mask-policy vessel_argmax \
        --output-dir "${WORK_ROOT}/biomarkers/phase_geometry_t${suffix}"
done
"${PYTHON_BIN}" "${BIOMARKER}" \
    --data-root "${DATA_ROOT}" --disc-dir "${DISC_ROOT}/validation" \
    --split validation --source "${WORK_ROOT}/raw/vein_density" \
    --threshold 0.5 --mask-policy vessel_argmax \
    --output-dir "${WORK_ROOT}/biomarkers/vein_density"

"${PYTHON_BIN}" "${SCRIPT_DIR}/assemble_biomarkers.py" \
    --caliber-dir "${WORK_ROOT}/biomarkers/caliber" \
    --artery-density-dir "${WORK_ROOT}/biomarkers/artery_density" \
    --vein-density-dir "${WORK_ROOT}/biomarkers/vein_density" \
    --artery-fd-dir "${WORK_ROOT}/biomarkers/artery_fd" \
    --vein-fd-per-case \
        "${WORK_ROOT}/biomarkers/phase_geometry_t04/per_case.json" \
        "${WORK_ROOT}/biomarkers/phase_geometry_t05/per_case.json" \
        "${WORK_ROOT}/biomarkers/phase_geometry_t06/per_case.json" \
    --output-dir "${WORK_ROOT}/build/Task3" \
    --report-output "${WORK_ROOT}/biomarker_assembly.json"

"${PYTHON_BIN}" "${SOLUTION}/compact_submission.py" \
    --source-root "${WORK_ROOT}/build" \
    --output-root "${WORK_ROOT}/build_step4" \
    --step 4
"${PYTHON_BIN}" "${SOLUTION}/package_submission.py" \
    --submission-root "${WORK_ROOT}/build_step4" \
    --output "${OUTPUT_ZIP}" \
    --start-index "${START_INDEX}" \
    --count "${CASE_COUNT}"

echo "VascFusion inference complete: ${OUTPUT_ZIP}"
