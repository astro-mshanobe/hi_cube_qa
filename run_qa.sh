#!/bin/bash
#SBATCH --job-name=hi_cube_qa
#SBATCH --partition=Main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=hi_cube_qa_%j.log
#SBATCH --error=hi_cube_qa_%j.err

# ─────────────────────────────────────────────────────────────────────
# hi_cube_qa -- Slurm job for per-session HI cube QA
#
# Usage:
#   cd ~/workspace/hi_cube_qa
#   sbatch run_qa.sh                       # uses qa_G23_all_sessions.yaml
#   sbatch run_qa.sh my_config.yaml        # or a custom config
#
# Notes:
#   - Cubes are auto-discovered under each session root in the config; one
#     HTML report is written per session into output.dir.
#   - Memory: cubes are loaded one unit (pointing/cube_N) at a time, so peak
#     RAM is ~ a few cubes, not the whole session. 64 GB is comfortable.
#   - Tip: run `python3 -m hi_cube_qa --config <cfg> --list` first to preview which
#     pointings/cubes/products will be assessed (no heavy compute).
# ─────────────────────────────────────────────────────────────────────

CONFIG="${1:-qa_G23_all_sessions.yaml}"
CONTAINER="/software/astro/containers/ASTRO-PY3.12-latest.sif"

echo "=========================================="
echo "  hi_cube_qa -- $(date)"
echo "  Job ID  : ${SLURM_JOB_ID}"
echo "  Node    : $(hostname)"
echo "  Config  : ${CONFIG}"
echo "=========================================="

cd ~/workspace/hi_cube_qa

export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"
singularity exec ${CONTAINER} python3 -m hi_cube_qa --config "${CONFIG}"

echo ""
echo "=========================================="
echo "  Finished: $(date)"
echo "=========================================="
