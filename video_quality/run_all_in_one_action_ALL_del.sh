#!/usr/bin/env bash
set -euo pipefail

SELF_PATH="$(cd -- "$(dirname "$0")" && pwd)/$(basename "$0")"

CPU_PARTITION="intel"
GPU_PARTITION="L40"

# 默认排除节点。可以通过第 5 个命令行参数覆盖。
EXCLUDE_NODES_DEFAULT=""

ROOT="/ssdfs/datahome/usersht/dev/lwh/WorldArena"
VQ_DIR="${ROOT}/video_quality"

PREP_SCRIPT="${VQ_DIR}/prepare_eval_package_ALL_del.py"
SPLIT_SCRIPT="${VQ_DIR}/split_formal_package_shards_del.py"
AGG_SCRIPT="${VQ_DIR}/csv_results/aggregate_and_build_leaderboard_action_ALL_del.py"
DATA_PROCESS_SCRIPT="${ROOT}/data_process.py"
RUN_JEPA_SCRIPT="${VQ_DIR}/run_evaluation_JEPA.sh"

CONFIG_PATH="${VQ_DIR}/config/config_ALL.yaml"
SIF_PATH="/share/home/usersht/images/ubuntu_2204.sif"
CONDA_SH="/share/apps/miniconda3/etc/profile.d/conda.sh"
METRICS="semantic_alignment,aesthetic_quality,background_consistency,dynamic_degree,flow_score,photometric_smoothness,motion_smoothness,subject_consistency,image_quality,depth_accuracy,trajectory_accuracy"

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

warn() {
  echo "[WARN] $*" >&2
}

is_positive_int() {
  [[ "${1:-}" =~ ^[0-9]+$ ]] && [ "${1}" -gt 0 ]
}

usage() {
  cat <<EOF
Usage:
  bash ${SELF_PATH} <base_model_name> <num_servers> [shards_per_server] [mail_user] [exclude_nodes] [gt_video_dir] [task_jsonl]
  bash ${SELF_PATH} submit <base_model_name> <num_servers> [shards_per_server] [mail_user] [exclude_nodes] [gt_video_dir] [task_jsonl]

Internal modes (normally do not call manually):
  bash ${SELF_PATH} prepare <base_model_name>
  bash ${SELF_PATH} split <base_model_name> <num_servers> [shards_per_server]
  bash ${SELF_PATH} worker <formal_model_name_with_server_suffix> [shards_per_server]
  bash ${SELF_PATH} jepa <base_model_name> [gt_video_dir]
  bash ${SELF_PATH} aggregate <base_model_name> [task_jsonl]

Examples:
  bash ${SELF_PATH} runid_xxx 5
  bash ${SELF_PATH} runid_xxx 5 4
  bash ${SELF_PATH} runid_xxx 5 4 your_email@qq.com
  bash ${SELF_PATH} runid_xxx 5 4 your_email@qq.com gpu4041
  bash ${SELF_PATH} runid_xxx 5 4 your_email@qq.com gpu4041,gpu4008
  bash ${SELF_PATH} runid_xxx 5 4 your_email@qq.com none /path/to/gt_mp4_folder
  bash ${SELF_PATH} runid_xxx 5 4 your_email@qq.com gpu4041 /path/to/gt_mp4_folder

Notes:
  - gt_video_dir is optional.
  - If gt_video_dir is provided, JEPA uses that folder as GT source.
  - gt_video_dir should directly contain .mp4 files.
  - If gt_video_dir is not provided, JEPA defaults to:
      ${VQ_DIR}/data/<formal_model_name>/gt_videos
EOF
}

derive_names() {
  local base_model_name="$1"

  if [[ "${base_model_name}" == *_formal ]]; then
    RAW_MODEL_NAME="${base_model_name%_formal}"
    FORMAL_MODEL_NAME="${base_model_name}"
  else
    RAW_MODEL_NAME="${base_model_name}"
    FORMAL_MODEL_NAME="${base_model_name}_formal"
  fi

  INPUT_ROOT="${VQ_DIR}/data/${RAW_MODEL_NAME}"
  OUTPUT_ROOT="${VQ_DIR}/data/${FORMAL_MODEL_NAME}"
  LOG_DIR="${VQ_DIR}/nohup_log/pipeline_${FORMAL_MODEL_NAME}"
}

has_action_following_dirs() {
  local root_dir="$1"
  local root_name
  root_name="$(basename "$root_dir")"

  if [ -d "${root_dir}/example_test_1" ] && [ -d "${root_dir}/example_test_2" ]; then
    return 0
  fi

  if [ -d "${root_dir}/${root_name}_test_1" ] && [ -d "${root_dir}/${root_name}_test_2" ]; then
    return 0
  fi

  return 1
}

prepare_jepa_matched_gt_dir() {
  local gen_dir="${1:?missing gen_dir}"
  local gt_src_dir="${2:?missing gt_src_dir}"
  local matched_dir="${3:?missing matched_dir}"

  [ -d "${gen_dir}" ] || die "JEPA gen_dir not found: ${gen_dir}"
  [ -d "${gt_src_dir}" ] || die "JEPA gt_video_dir not found: ${gt_src_dir}"

  local gen_count
  local gt_count
  gen_count=$(find "${gen_dir}" -maxdepth 1 -type f -name "*.mp4" | wc -l)
  gt_count=$(find "${gt_src_dir}" -maxdepth 1 -type f -name "*.mp4" | wc -l)

  if [ "${gen_count}" -le 0 ]; then
    die "No mp4 files found in JEPA gen_dir: ${gen_dir}"
  fi

  if [ "${gt_count}" -le 0 ]; then
    die "No mp4 files found in JEPA gt_video_dir: ${gt_src_dir}"
  fi

  rm -rf "${matched_dir}"
  mkdir -p "${matched_dir}"

  local matched_count=0
  local missing_count=0

  shopt -s nullglob

  local gen
  for gen in "${gen_dir}"/*.mp4; do
    local name
    local stem
    local target=""
    local episode_tail=""

    name="$(basename "${gen}")"
    stem="${name%.mp4}"

    # 1. 完全同名匹配：GT/xxx.mp4 == GEN/xxx.mp4
    if [ -f "${gt_src_dir}/${stem}.mp4" ]; then
      target="${gt_src_dir}/${stem}.mp4"
    fi

    # 2. 如果 stem 末尾包含 episode数字，则抽取 episodeXX
    #    例如 robotwin_480_640_episode20 -> episode20
    if [ -z "${target}" ] && [[ "${stem}" =~ (episode[0-9]+)$ ]]; then
      episode_tail="${BASH_REMATCH[1]}"

      if [ -f "${gt_src_dir}/${episode_tail}.mp4" ]; then
        target="${gt_src_dir}/${episode_tail}.mp4"
      fi
    fi

    # 3. 通配匹配：*_stem.mp4 或 *stem.mp4
    if [ -z "${target}" ]; then
      local -a matches=()
      mapfile -t matches < <(
        find "${gt_src_dir}" -maxdepth 1 -type f \
          \( -name "*_${stem}.mp4" -o -name "*${stem}.mp4" \) \
          | sort
      )

      if [ "${#matches[@]}" -gt 0 ]; then
        target="${matches[0]}"
      fi
    fi

    # 4. 如果有 episode_tail，再尝试 *_episodeXX.mp4 或 *episodeXX.mp4
    if [ -z "${target}" ] && [ -n "${episode_tail}" ]; then
      local -a matches_tail=()
      mapfile -t matches_tail < <(
        find "${gt_src_dir}" -maxdepth 1 -type f \
          \( -name "*_${episode_tail}.mp4" -o -name "*${episode_tail}.mp4" \) \
          | sort
      )

      if [ "${#matches_tail[@]}" -gt 0 ]; then
        target="${matches_tail[0]}"
      fi
    fi

    if [ -n "${target}" ] && [ -f "${target}" ]; then
      ln -s "${target}" "${matched_dir}/${name}"
      matched_count=$((matched_count + 1))
    else
      warn "JEPA GT match missing for generated video: ${name}"
      missing_count=$((missing_count + 1))
    fi
  done

  # 如果一个都没按名字匹配上，则退化为按排序顺序匹配。
  # 注意：这可能存在错配风险，但比 JEPA 直接失败更适合兜底排查。
  if [ "${matched_count}" -eq 0 ]; then
    warn "No filename-based JEPA GT matches found. Falling back to ordered matching."

    rm -rf "${matched_dir}"
    mkdir -p "${matched_dir}"

    local -a gen_files=()
    local -a gt_files=()

    mapfile -t gen_files < <(find "${gen_dir}" -maxdepth 1 -type f -name "*.mp4" | sort)
    mapfile -t gt_files < <(find "${gt_src_dir}" -maxdepth 1 -type f -name "*.mp4" | sort)

    local n_gen="${#gen_files[@]}"
    local n_gt="${#gt_files[@]}"
    local n="${n_gen}"

    if [ "${n_gt}" -lt "${n}" ]; then
      n="${n_gt}"
    fi

    if [ "${n}" -le 0 ]; then
      die "Ordered JEPA matching failed: gen_count=${n_gen}, gt_count=${n_gt}"
    fi

    if [ "${n_gen}" -ne "${n_gt}" ]; then
      warn "Ordered JEPA matching with unequal counts: gen=${n_gen}, gt=${n_gt}, using first ${n} pairs."
    fi

    local i
    for ((i=0; i<n; i++)); do
      local gen_name
      gen_name="$(basename "${gen_files[$i]}")"
      ln -s "${gt_files[$i]}" "${matched_dir}/${gen_name}"
      matched_count=$((matched_count + 1))
    done

    missing_count=0
  fi

  if [ "${matched_count}" -le 0 ]; then
    die "No JEPA GT pairs created. gen_dir=${gen_dir}, gt_src_dir=${gt_src_dir}"
  fi

  echo "[JEPA] gen mp4 count: ${gen_count}"
  echo "[JEPA] gt source mp4 count: ${gt_count}"
  echo "[JEPA] matched gt count: ${matched_count}"
  echo "[JEPA] missing gt count: ${missing_count}"
  echo "[JEPA] matched gt dir: ${matched_dir}"
}

run_prepare() {
  local base_model_name="${1:?missing base model name}"
  derive_names "${base_model_name}"

  source "${CONDA_SH}"
  conda activate WorldArena

  local -a prep_args=(
    python "${PREP_SCRIPT}"
    --input_root "${INPUT_ROOT}"
    --output_root "${OUTPUT_ROOT}"
    --model_name "${FORMAL_MODEL_NAME}"
    --src_main example_test
    --id1 usersht
    --image_mode keep
    --overwrite
  )

  if has_action_following_dirs "${INPUT_ROOT}"; then
    echo "[prepare] Found example_test_1 and example_test_2. Enable action_following package fields."
    prep_args+=(
      --src_alt1 example_test_1
      --src_alt2 example_test_2
    )
  else
    echo "[prepare] example_test_1/example_test_2 not found. Skip action_following package fields."
  fi

  "${prep_args[@]}"
}

run_split() {
  local base_model_name="${1:?missing base model name}"
  local num_servers="${2:?missing num_servers}"
  local shards_per_server="${3:-8}"

  is_positive_int "${num_servers}" || die "num_servers must be a positive integer, got: ${num_servers}"
  is_positive_int "${shards_per_server}" || die "shards_per_server must be a positive integer, got: ${shards_per_server}"

  derive_names "${base_model_name}"

  source "${CONDA_SH}"
  conda activate WorldArena

  WORLD_ARENA_NUM_SERVERS="${num_servers}" \
  WORLD_ARENA_SHARDS_PER_SERVER="${shards_per_server}" \
  python "${SPLIT_SCRIPT}" "${OUTPUT_ROOT}"
}

run_worker() {
  local shard_model="${1:?missing shard model name}"
  local shards_per_server="${2:-8}"

  is_positive_int "${shards_per_server}" || die "shards_per_server must be a positive integer, got: ${shards_per_server}"

  local summary_json="${VQ_DIR}/data/${shard_model}/summary.json"
  local video_dir="${VQ_DIR}/data/${shard_model}/${shard_model}_test_vlm"
  local gen_video_dir="${VQ_DIR}/data/${shard_model}/${shard_model}_test"

  local shard_root="${VQ_DIR}/data/${shard_model}"
  local action_following_enabled=0

  if has_action_following_dirs "${shard_root}"; then
    action_following_enabled=1
    echo "[worker] action_following enabled for ${shard_model}"
  else
    echo "[worker] action_following skipped for ${shard_model}: *_test_1 and *_test_2 not found"
  fi

  local ALL_GPUS=(0 1 2 3 4 5 6 7)
  if [ "${shards_per_server}" -gt "${#ALL_GPUS[@]}" ]; then
    die "shards_per_server (${shards_per_server}) exceeds available GPUs (${#ALL_GPUS[@]})"
  fi

  local GPUS=("${ALL_GPUS[@]:0:${shards_per_server}}")
  local NUM_SHARDS=${#GPUS[@]}

  mkdir -p "${VQ_DIR}/nohup_log/${shard_model}"

  echo "=================================================="
  echo "[START] MODEL_NAME=${shard_model}"
  echo "[INFO] NUM_SHARDS=${NUM_SHARDS}"
  echo "=================================================="

  cd "${VQ_DIR}"

  echo "[0/3] Skip data_process.py for ${shard_model} (assume already prepared)"

  echo "[1/3] Run VLM judge for ${shard_model}"

  source "${CONDA_SH}"
  conda activate WorldArena

  VLM_PIDS=()

  for shard_idx in "${!GPUS[@]}"; do
    gpu=${GPUS[$shard_idx]}
    env CUDA_VISIBLE_DEVICES=$gpu bash "${VQ_DIR}/run_VLM_judge.sh" \
      "${shard_model}" \
      "${video_dir}" \
      "${summary_json}" \
      all \
      "${CONFIG_PATH}" \
      "${shard_idx}" "${NUM_SHARDS}" \
      > "${VQ_DIR}/nohup_log/${shard_model}/output_run_VLM_judge_${shard_model}_shard${shard_idx}_gpu${gpu}.log" 2>&1 &
    VLM_PIDS+=("$!")
  done

  echo "[2/3] Run 8-metrics + action_following in singularity for ${shard_model}"

  source /etc/profile.d/modules.sh 2>/dev/null || true
  module load Singularity/4.2.1

  unset -f module 2>/dev/null || true
  unset -f ml 2>/dev/null || true

  singularity exec --cleanenv --nv \
    -B /ssdfs:/ssdfs \
    -B /share:/share \
    "${SIF_PATH}" \
    bash -lc "
set -euo pipefail
source '${CONDA_SH}'
conda activate WorldArena_test
cd '${VQ_DIR}'

GPUS=(${GPUS[*]})
NUM_SHARDS=${NUM_SHARDS}

EVAL_PIDS=()

for shard_idx in \"\${!GPUS[@]}\"; do
  gpu=\${GPUS[\$shard_idx]}
  WORLD_ARENA_OVERWRITE_RESULTS=0 CUDA_VISIBLE_DEVICES=\$gpu \
  bash '${VQ_DIR}/run_evaluation_8metrics_ALL.sh' \
    '${shard_model}' \
    '${gen_video_dir}' \
    '${summary_json}' \
    '${METRICS}' \
    '${CONFIG_PATH}' \
    \"\$shard_idx\" \"\$NUM_SHARDS\" \
    > '${VQ_DIR}/nohup_log/${shard_model}/output_run_evaluation_8metrics_ALL_${shard_model}_shard'\${shard_idx}'_gpu'\${gpu}'.log' 2>&1 &
  EVAL_PIDS+=(\"\$!\")
done

if [ \"${action_following_enabled}\" = \"1\" ]; then
  echo '[2B] action_following enabled for ${shard_model}'
  for shard_idx in \"\${!GPUS[@]}\"; do
    gpu=\${GPUS[\$shard_idx]}
    WORLD_ARENA_OVERWRITE_RESULTS=0 CUDA_VISIBLE_DEVICES=\$gpu \
    bash '${VQ_DIR}/run_action_following.sh' \
      '${shard_model}' \
      '${gen_video_dir}' \
      '${summary_json}' \
      '${CONFIG_PATH}' \
      \"\$shard_idx\" \"\$NUM_SHARDS\" \
      > '${VQ_DIR}/nohup_log/${shard_model}/output_run_action_following_${shard_model}_shard'\${shard_idx}'_gpu'\${gpu}'.log' 2>&1 &
    EVAL_PIDS+=(\"\$!\")
  done
else
  echo '[2B] action_following skipped for ${shard_model}: *_test_1 and *_test_2 not found'
fi

eval_failed=0
for pid in \"\${EVAL_PIDS[@]}\"; do
  if ! wait \"\$pid\"; then
    eval_failed=1
  fi
done

if [ \"\$eval_failed\" -ne 0 ]; then
  echo '[ERROR] One or more 8-metrics/action_following shards failed for ${shard_model}.'
  exit 1
fi

echo '[2/3] 8-metrics + action_following finished for ${shard_model}'
"

  echo "[3/3] Wait host-side VLM judge for ${shard_model}"

  vlm_failed=0
  for pid in "${VLM_PIDS[@]}"; do
    if ! wait "$pid"; then
      vlm_failed=1
    fi
  done

  if [ "$vlm_failed" -ne 0 ]; then
    echo "[ERROR] One or more VLM judge shards failed for ${shard_model}."
    exit 1
  fi

  echo "=================================================="
  echo "[DONE] MODEL_NAME=${shard_model}"
  echo "=================================================="
}

run_jepa() {
  local base_model_name="${1:?missing base model name}"
  local gt_video_dir="${2:-}"

  derive_names "${base_model_name}"
  mkdir -p "${LOG_DIR}"

  source "${CONDA_SH}"
  conda activate WorldArena

  echo "[JEPA] Run data_process.py for formal model: ${FORMAL_MODEL_NAME}"
  python "${DATA_PROCESS_SCRIPT}" \
    --results_dir "${OUTPUT_ROOT}" \
    --model_name "${FORMAL_MODEL_NAME}" \
    > "${LOG_DIR}/output_data_process_JEPA_${FORMAL_MODEL_NAME}.log" 2>&1 || true

  echo "[JEPA] data_process log: ${LOG_DIR}/output_data_process_JEPA_${FORMAL_MODEL_NAME}.log"

  local gen_dir="${OUTPUT_ROOT}/${FORMAL_MODEL_NAME}_test_vlm"

  if [ -z "${gt_video_dir}" ]; then
    gt_video_dir="${OUTPUT_ROOT}/gt_videos"
    echo "[JEPA] GT video dir not provided. Use default: ${gt_video_dir}"
  else
    echo "[JEPA] GT video dir provided: ${gt_video_dir}"
  fi

  local matched_gt_dir="${OUTPUT_ROOT}/gt_videos_jepa_matched"

  prepare_jepa_matched_gt_dir \
    "${gen_dir}" \
    "${gt_video_dir}" \
    "${matched_gt_dir}"

  echo "[JEPA] Run JEPA for formal model: ${FORMAL_MODEL_NAME}"
  echo "[JEPA] gen_dir=${gen_dir}"
  echo "[JEPA] gt_dir=${matched_gt_dir}"

  cd "${ROOT}"

  unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT SLURM_PROCID SLURM_LOCALID SLURM_NTASKS

  env CUDA_VISIBLE_DEVICES=0 bash "${RUN_JEPA_SCRIPT}" \
    "${FORMAL_MODEL_NAME}" \
    "${gen_dir}" \
    "${matched_gt_dir}" \
    > "${LOG_DIR}/output_run_evaluation_JEPA_${FORMAL_MODEL_NAME}.log" 2>&1
}

run_aggregate() {
  local base_model_name="${1:?missing base model name}"
  local task_jsonl="${2:-}"

  derive_names "${base_model_name}"

  source "${CONDA_SH}"
  conda activate WorldArena

  local -a agg_args=(
    python "${AGG_SCRIPT}"
    --model_name "${FORMAL_MODEL_NAME}"
  )

  if [ -n "${task_jsonl}" ]; then
    [ -f "${task_jsonl}" ] || die "task_jsonl does not exist or is not a file: ${task_jsonl}"
    agg_args+=(--task_jsonl "${task_jsonl}")
  fi

  "${agg_args[@]}"
}

submit_pipeline() {
  local base_model_name="${1:?missing base model name}"
  local num_servers="${2:?missing num_servers}"
  local shards_per_server="${3:-8}"
  local mail_user="${4:-}"
  local exclude_nodes="${5:-${EXCLUDE_NODES_DEFAULT}}"
  local gt_video_dir="${6:-}"
  local task_jsonl="${7:-}"

  is_positive_int "${num_servers}" || die "num_servers must be a positive integer, got: ${num_servers}"
  is_positive_int "${shards_per_server}" || die "shards_per_server must be a positive integer, got: ${shards_per_server}"

  if [ -n "${task_jsonl}" ] && [ ! -f "${task_jsonl}" ]; then
    die "task_jsonl does not exist or is not a file: ${task_jsonl}"
  fi

  if [ "${exclude_nodes}" = "none" ] || [ "${exclude_nodes}" = "NONE" ]; then
    exclude_nodes=""
  fi

  if [ -n "${gt_video_dir}" ] && [ ! -d "${gt_video_dir}" ]; then
    die "gt_video_dir does not exist or is not a directory: ${gt_video_dir}"
  fi

  derive_names "${base_model_name}"
  mkdir -p "${LOG_DIR}"

  local worker_cpus=$((shards_per_server * 7))

  local -a GPU_EXCLUDE_ARGS=()
  if [ -n "${exclude_nodes}" ]; then
    GPU_EXCLUDE_ARGS+=(--exclude="${exclude_nodes}")
  fi

  echo "=================================================="
  echo "[PIPELINE] RAW_MODEL_NAME=${RAW_MODEL_NAME}"
  echo "[PIPELINE] FORMAL_MODEL_NAME=${FORMAL_MODEL_NAME}"
  echo "[PIPELINE] NUM_SERVERS=${num_servers}"
  echo "[PIPELINE] SHARDS_PER_SERVER=${shards_per_server}"
  echo "[PIPELINE] WORKER_CPUS=${worker_cpus}"
  if [ -n "${mail_user}" ]; then
    echo "[PIPELINE] MAIL_USER=${mail_user}"
  else
    echo "[PIPELINE] MAIL_USER=(disabled)"
  fi
  if [ -n "${exclude_nodes}" ]; then
    echo "[PIPELINE] EXCLUDE_NODES=${exclude_nodes}"
  else
    echo "[PIPELINE] EXCLUDE_NODES=(none)"
  fi
  if [ -n "${gt_video_dir}" ]; then
    echo "[PIPELINE] GT_VIDEO_DIR=${gt_video_dir}"
  else
    echo "[PIPELINE] GT_VIDEO_DIR=(default: ${OUTPUT_ROOT}/gt_videos)"
  fi
  if [ -n "${task_jsonl}" ]; then
    echo "[PIPELINE] TASK_JSONL=${task_jsonl}"
  else
    echo "[PIPELINE] TASK_JSONL=(auto)"
  fi
  echo "[PIPELINE] LOG_DIR=${LOG_DIR}"
  echo "=================================================="

  PREP_JOB_ID=$(sbatch --parsable \
    -J "prep_${FORMAL_MODEL_NAME}" \
    -p "${CPU_PARTITION}" \
    -N 1 -n 1 -c 7 \
    -o "${LOG_DIR}/prep_%j.out" \
    --wrap="bash ${SELF_PATH} prepare ${RAW_MODEL_NAME}")

  echo "[SUBMIT] prepare job id: ${PREP_JOB_ID}"

  SPLIT_JOB_ID=$(sbatch --parsable \
    --dependency=afterok:${PREP_JOB_ID} \
    -J "split_${FORMAL_MODEL_NAME}" \
    -p "${CPU_PARTITION}" \
    -N 1 -n 1 -c 7 \
    -o "${LOG_DIR}/split_%j.out" \
    --wrap="bash ${SELF_PATH} split ${RAW_MODEL_NAME} ${num_servers} ${shards_per_server}")

  echo "[SUBMIT] split job id: ${SPLIT_JOB_ID}"

  local -a WORKER_JOB_IDS=()
  local sid
  for ((sid=0; sid<num_servers; sid++)); do
    local shard_model="${FORMAL_MODEL_NAME}_${sid}"
    local jid
    jid=$(sbatch --parsable \
      --dependency=afterok:${SPLIT_JOB_ID} \
      -J "eval_${FORMAL_MODEL_NAME}_${sid}" \
      -p "${GPU_PARTITION}" \
      -N 1 -n 1 -c "${worker_cpus}" --gres=gpu:l40:${shards_per_server} \
      "${GPU_EXCLUDE_ARGS[@]}" \
      -o "${LOG_DIR}/worker_${sid}_%j.out" \
      --wrap="bash ${SELF_PATH} worker ${shard_model} ${shards_per_server}")
    WORKER_JOB_IDS+=("${jid}")
    echo "[SUBMIT] worker shard=${sid} job id: ${jid}"
  done

  local WORKER_DEP
  WORKER_DEP=$(IFS=:; echo "${WORKER_JOB_IDS[*]}")

  local jepa_wrap
  if [ -n "${gt_video_dir}" ]; then
    jepa_wrap="bash ${SELF_PATH} jepa ${RAW_MODEL_NAME} ${gt_video_dir}"
  else
    jepa_wrap="bash ${SELF_PATH} jepa ${RAW_MODEL_NAME}"
  fi

  JEPA_JOB_ID=$(sbatch --parsable \
    --dependency=afterok:${WORKER_DEP} \
    -J "jepa_${FORMAL_MODEL_NAME}" \
    -p "${GPU_PARTITION}" \
    -N 1 -n 1 -c 7 --gres=gpu:l40:1 \
    "${GPU_EXCLUDE_ARGS[@]}" \
    -o "${LOG_DIR}/jepa_%j.out" \
    --wrap="${jepa_wrap}")

  echo "[SUBMIT] jepa job id: ${JEPA_JOB_ID}"

  local -a AGG_MAIL_OPTS=()
  if [ -n "${mail_user}" ]; then
    AGG_MAIL_OPTS+=(--mail-type=END,FAIL --mail-user="${mail_user}")
  fi

  local agg_wrap
  if [ -n "${task_jsonl}" ]; then
    agg_wrap="bash ${SELF_PATH} aggregate ${RAW_MODEL_NAME} ${task_jsonl}"
  else
    agg_wrap="bash ${SELF_PATH} aggregate ${RAW_MODEL_NAME}"
  fi

  AGG_JOB_ID=$(sbatch --parsable \
    --dependency=afterok:${JEPA_JOB_ID} \
    -J "agg_${FORMAL_MODEL_NAME}" \
    -p "${CPU_PARTITION}" \
    -N 1 -n 1 -c 7 \
    "${AGG_MAIL_OPTS[@]}" \
    -o "${LOG_DIR}/aggregate_%j.out" \
    --wrap="${agg_wrap}")

  echo "[SUBMIT] aggregate job id: ${AGG_JOB_ID}"

  echo "=================================================="
  echo "[DONE] Pipeline submitted successfully."
  echo
  echo "Prepare job:   ${PREP_JOB_ID}"
  echo "Split job:     ${SPLIT_JOB_ID}"
  echo "Worker jobs:   ${WORKER_JOB_IDS[*]}"
  echo "JEPA job:      ${JEPA_JOB_ID}"
  echo "Aggregate job: ${AGG_JOB_ID}"
  if [ -n "${mail_user}" ]; then
    echo "Mail notify:   enabled (${mail_user})"
  else
    echo "Mail notify:   disabled"
  fi
  if [ -n "${exclude_nodes}" ]; then
    echo "Exclude nodes: ${exclude_nodes}"
  else
    echo "Exclude nodes: none"
  fi
  if [ -n "${gt_video_dir}" ]; then
    echo "GT video dir:  ${gt_video_dir}"
  else
    echo "GT video dir:  default"
  fi
  echo
  echo "Check status:"
  echo "  squeue -j ${PREP_JOB_ID},${SPLIT_JOB_ID},${JEPA_JOB_ID},${AGG_JOB_ID}$(printf ",%s" "${WORKER_JOB_IDS[@]}")"
  echo "Scancel Job:"
  echo "  scancel ${PREP_JOB_ID} ${SPLIT_JOB_ID} ${JEPA_JOB_ID} ${AGG_JOB_ID}$(printf " %s" "${WORKER_JOB_IDS[@]}")"
  echo
  echo "Logs:"
  echo "  ${LOG_DIR}"
  echo "=================================================="
}

MODE="submit"
if [ $# -gt 0 ]; then
  case "$1" in
    submit|prepare|split|worker|jepa|aggregate)
      MODE="$1"
      shift
      ;;
  esac
fi

case "${MODE}" in
  submit)
    [ $# -ge 2 ] || { usage; exit 1; }
    submit_pipeline "$@"
    ;;
  prepare)
    [ $# -ge 1 ] || { usage; exit 1; }
    run_prepare "$@"
    ;;
  split)
    [ $# -ge 2 ] || { usage; exit 1; }
    run_split "$@"
    ;;
  worker)
    [ $# -ge 1 ] || { usage; exit 1; }
    run_worker "$@"
    ;;
  jepa)
    [ $# -ge 1 ] || { usage; exit 1; }
    run_jepa "$@"
    ;;
  aggregate)
    [ $# -ge 1 ] || { usage; exit 1; }
    run_aggregate "$@"
    ;;
  *)
    usage
    exit 1
    ;;
esac