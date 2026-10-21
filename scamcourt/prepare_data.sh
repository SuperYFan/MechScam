#!/usr/bin/env bash
# Builds the training files used by run_all.sh / run_glm.sh from the released data (CPU only, minutes).
# Requires the video frames under data/mechscam/samples/<collection>/<video>/frames/ (see README).
#   bash scamcourt/prepare_data.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p work scamcourt/data

# 1) Cold-start SFT data for the debater (4 roles) and the judge, training split only
python scripts/build_scamcourt_coldstart_sft.py \
  --split-file data/splits/scamcourt_v3_seed42.json --split train \
  --roles false_opening,false_rebuttal,true_opening,true_rebuttal,judge \
  --output-dir work/sft_coldstart

# 2) Two-fold cross-fitting split of the training set (judge-training debates)
python scamcourt/make_crossfit_folds.py

# 3) Same-backbone SFT classifier baseline data (same output format, no debate)
python scamcourt/build_judge_data.py --classifier-only --output scamcourt/data/classifier.jsonl

# 4) Adversarial test set (already released as data/adversarial_test.jsonl; rebuild only if missing)
[ -f data/adversarial_test.jsonl ] || python scamcourt/build_adversarial_testset.py
echo "done"
