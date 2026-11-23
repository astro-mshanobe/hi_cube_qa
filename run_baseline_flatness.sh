#!/bin/bash
#SBATCH --job-name=baseline_flatness
#SBATCH --partition=Main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=baseline_flatness_%j.log
#SBATCH --error=baseline_flatness_%j.err

# ─────────────────────────────────────────────────────────────────────
# baseline_flatness -- mosaicking-prep check: is the per-channel mean
# (baseline) flat across the band for every continuum-subtracted cube?
#
# Usage:
#   cd ~/workspace/hi_cube_qa
#   sbatch run_baseline_flatness.sh                # uses qa_G23_all_sessions.yaml
#   sbatch run_baseline_flatness.sh my_config.yaml
# ─────────────────────────────────────────────────────────────────────

CONFIG="${1:-qa_G23_all_sessions.yaml}"
CONTAINER="/software/astro/containers/ASTRO-PY3.12-latest.sif"

echo "=========================================="
echo "  baseline_flatness -- $(date)"
echo "  Job ID  : ${SLURM_JOB_ID}"
echo "  Node    : $(hostname)"
echo "  Config  : ${CONFIG}"
echo "=========================================="

cd ~/workspace/hi_cube_qa

export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"
singularity exec ${CONTAINER} python3 -m hi_cube_qa.baseline_flatness --config "${CONFIG}"

echo ""
echo "=========================================="
echo "  Finished: $(date)"
echo "=========================================="
