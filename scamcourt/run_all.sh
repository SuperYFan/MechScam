#!/usr/bin/env bash
# ScamCourt with the Qwen3-VL-8B backbone: debater SFT, debates, judge SFT and evaluations on ONE GPU, in order. Safe to rerun after an interruption:
# finished trainings are skipped (adapter/ exists), debates and evaluations resume where they stopped.
# Stops at the first error. Usage (from the project root):
#   mkdir -p logs && nohup bash scamcourt/run_all.sh > logs/run_all.log 2>&1 &
#   tail -n 5 logs/run_all.log          # which stage is running
#   tail -n 3 logs/<stage>.log          # progress of that stage
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs scamcourt/data

SFT=work/sft_coldstart
DEBATER_DATA="$SFT/false_opening.jsonl $SFT/false_rebuttal.jsonl $SFT/true_opening.jsonl $SFT/true_rebuttal.jsonl"
FOLDS=data/splits/crossfit
R=scamcourt/runs

stage() { echo "[$(date '+%m-%d %H:%M')] $1"; }

train() {  # train <output-dir> <log-name> <args...>
  local out=$1 log=$2; shift 2
  if [ -d "$out/adapter" ]; then stage "skip $log ($out/adapter exists)"; return; fi
  stage "start $log"
  python scamcourt/train_sft.py "$@" --output-dir "$out" > "logs/$log.log" 2>&1
}

run() {  # run <log-name> <command...>   (debates / evaluations resume by themselves)
  local log=$1; shift
  stage "start $log"
  "$@" > "logs/$log.log" 2>&1
}

# --- data preparation (CPU, minutes)
if [ ! -f scamcourt/data/classifier.jsonl ] || [ ! -f $SFT/judge.jsonl ]; then
  stage "prepare SFT data / folds / classifier data"
  bash scamcourt/prepare_data.sh > logs/prepare_data.log 2>&1
fi

# --- step 4: debater SFT (both sides, openings and rebuttals, one shared LoRA adapter)
train $R/debater_sft debater_sft --data $DEBATER_DATA --epochs 1

# --- step 5a: test-set debates with the full debater (~9 h)
run debates_test python scamcourt/run_debate.py --split test --adapter $R/debater_sft/adapter --output-dir $R/debates_test

# --- step 5b: cross-fitted training-set debates (2 x (~7 h SFT + ~18 h debates))
for f in a b; do
  train $R/debater_fold_$f debater_fold_$f --split-file $FOLDS/fold_$f.json --data $DEBATER_DATA --epochs 1
  # in a fold file, "test" is the OTHER half of the training split (not the real test set)
  run debates_train_fold_$f python scamcourt/run_debate.py --split-file $FOLDS/fold_$f.json --split test \
    --adapter $R/debater_fold_$f/adapter --output-dir $R/debates_train_fold_$f
done

# --- step 6: judge (~20 h)
stage "build judge data"
python scamcourt/build_judge_data.py --debates $R/debates_train_fold_a/debates.jsonl $R/debates_train_fold_b/debates.jsonl \
  --output scamcourt/data/judge_rollouts.jsonl > logs/judge_data.log 2>&1
train $R/judge_sft judge_sft --data $SFT/judge.jsonl scamcourt/data/judge_rollouts.jsonl

# --- step 7: ScamCourt evaluations (main results first)
J=$R/judge_sft/adapter
run eval_judge_debate python scamcourt/evaluate.py --mode debate --adapter $J --debates $R/debates_test/debates.jsonl --output-dir $R/eval/judge_debate
run eval_judge_plain  python scamcourt/evaluate.py --mode no_debate --adapter $J --output-dir $R/eval/judge_plain
run eval_judge_adv    python scamcourt/evaluate.py --mode adversarial --adapter $J --output-dir $R/eval/judge_adv

# --- baselines: plain SFT classifier (~9 h) and zero-shot
train $R/classifier_sft classifier_sft --data scamcourt/data/classifier.jsonl
C=$R/classifier_sft/adapter
run eval_classifier_plain python scamcourt/evaluate.py --mode no_debate --adapter $C --output-dir $R/eval/classifier_plain
run eval_classifier_adv   python scamcourt/evaluate.py --mode adversarial --adapter $C --output-dir $R/eval/classifier_adv
run eval_zeroshot_plain   python scamcourt/evaluate.py --mode no_debate --no-adapter --output-dir $R/eval/zeroshot_plain
run eval_zeroshot_adv     python scamcourt/evaluate.py --mode adversarial --no-adapter --output-dir $R/eval/zeroshot_adv

# --- baseline: prompted self-debate with the untrained backbone (debaters keep their assigned side)
run tf_debates_test python scamcourt/run_debate.py --split test --no-adapter --role-reminder --output-dir $R/tf_debates_test
run eval_tf_selfdebate_full python scamcourt/evaluate.py --mode debate --no-adapter --debates $R/tf_debates_test/debates.jsonl \
  --debate-view full --output-dir $R/eval/tf_selfdebate_full

stage "all done"
