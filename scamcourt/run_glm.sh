#!/usr/bin/env bash
# Full ScamCourt pipeline with a second backbone (GLM-4.1V-9B) on THREE GPUs of the same machine.
# Same data, folds, prompts, hyperparameters and adversarial set as the Qwen3-VL-8B runs; everything is
# written to scamcourt/runs_glm and scamcourt/data_glm, so the Qwen results are never touched.
# The three lanes share this disk and wait for each other through DONE marker files.
#
#   export GLM=models/GLM-4.1V-9B-Thinking
#   mkdir -p logs
#   CUDA_VISIBLE_DEVICES=0 nohup bash scamcourt/run_glm.sh 0 > logs/glm_lane0.log 2>&1 &
#   CUDA_VISIBLE_DEVICES=1 nohup bash scamcourt/run_glm.sh 1 > logs/glm_lane1.log 2>&1 &
#   CUDA_VISIBLE_DEVICES=2 nohup bash scamcourt/run_glm.sh 2 > logs/glm_lane2.log 2>&1 &
#   bash scamcourt/run_glm.sh status
#
# lane 0: debater fold A -> debates on half B -> (wait fold B debates) judge data + judge SFT
#         -> (wait test debates) full-hearing evaluation
# lane 1: debater fold B -> debates on half A -> classifier SFT -> classifier evaluations
#         -> (wait judge SFT) judge-alone and adversarial evaluations
# lane 2: debater (full train) -> test debates -> zero-shot evaluations -> prompted self-debate baseline
# Safe to rerun a lane after an interruption: finished trainings are skipped, debates/evaluations resume.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${GLM:?set GLM to the local GLM-4.1V model directory}"
M=(--model-path "$GLM")
mkdir -p logs scamcourt/data_glm

SFT=work/sft_coldstart
DEBATER_DATA="$SFT/false_opening.jsonl $SFT/false_rebuttal.jsonl $SFT/true_opening.jsonl $SFT/true_rebuttal.jsonl"
FOLDS=data/splits/crossfit
R=scamcourt/runs_glm
J=$R/judge_sft/adapter
C=$R/classifier_sft/adapter
# Zero-shot rows only: answer prefix that skips the thinking block (set GLM_ZS_PREFIX= to disable)
ZS_PREFIX="${GLM_ZS_PREFIX-<think></think><answer>}"
# ... and the start of free generation for the same rows (e.g. "{"), so untrained models answer in JSON
ZS_GEN_PREFIX="${GLM_ZS_GEN_PREFIX-{}"

stage() { echo "[$(date '+%m-%d %H:%M')] $1"; }
train() {  # train <output-dir> <log-name> <args...>
  local out=$1 log=$2; shift 2
  if [ -d "$out/adapter" ]; then stage "skip $log ($out/adapter exists)"; touch "$out/DONE"; return; fi
  stage "start $log"
  python scamcourt/train_sft.py "${M[@]}" "$@" --output-dir "$out" > "logs/glm_$log.log" 2>&1
  touch "$out/DONE"
  stage "done $log"
}
run() {  # run <log-name> <command...>
  local log=$1; shift
  stage "start $log"
  "$@" > "logs/glm_$log.log" 2>&1
  stage "done $log"
}
wait_for() { stage "waiting for $2 ($1)"; until [ -e "$1" ]; do sleep 300; done; stage "found $2"; }

debates_fold() {
  local f=$1
  train $R/debater_fold_$f debater_fold_$f --split-file $FOLDS/fold_$f.json --data $DEBATER_DATA --epochs 1
  if [ ! -e $R/debates_train_fold_$f/DONE ]; then
    run debates_train_fold_$f python scamcourt/run_debate.py "${M[@]}" --split-file $FOLDS/fold_$f.json --split test \
      --adapter $R/debater_fold_$f/adapter --output-dir $R/debates_train_fold_$f
    touch $R/debates_train_fold_$f/DONE
  fi
}

for f in $FOLDS/fold_a.json $FOLDS/fold_b.json scamcourt/data/classifier.jsonl \
         data/adversarial_test.jsonl $SFT/judge.jsonl; do
  [ -f "$f" ] || { echo "ERROR: $f missing (it is shared with the Qwen runs)" >&2; exit 1; }
done

case "${1:-}" in
  0)
    debates_fold a
    wait_for $R/debates_train_fold_b/DONE "fold B debates (lane 1)"
    if [ ! -d "$J" ]; then
      stage "build judge data"
      python scamcourt/build_judge_data.py --debates $R/debates_train_fold_a/debates.jsonl \
        $R/debates_train_fold_b/debates.jsonl --output scamcourt/data_glm/judge_rollouts.jsonl > logs/glm_judge_data.log 2>&1
    fi
    train $R/judge_sft judge_sft --data $SFT/judge.jsonl scamcourt/data_glm/judge_rollouts.jsonl
    wait_for $R/debates_test/DONE "test debates (lane 2)"
    run eval_judge_debate python scamcourt/evaluate.py "${M[@]}" --mode debate --adapter $J \
      --debates $R/debates_test/debates.jsonl --output-dir $R/eval/judge_debate
    stage "lane 0 all done"
    ;;
  1)
    debates_fold b
    train $R/classifier_sft classifier_sft --data scamcourt/data/classifier.jsonl
    run eval_classifier_plain python scamcourt/evaluate.py "${M[@]}" --mode no_debate --adapter $C --output-dir $R/eval/classifier_plain
    run eval_classifier_adv python scamcourt/evaluate.py "${M[@]}" --mode adversarial --adapter $C --output-dir $R/eval/classifier_adv
    wait_for $R/judge_sft/DONE "judge SFT (lane 0)"
    run eval_judge_plain python scamcourt/evaluate.py "${M[@]}" --mode no_debate --adapter $J --output-dir $R/eval/judge_plain
    run eval_judge_adv python scamcourt/evaluate.py "${M[@]}" --mode adversarial --adapter $J --output-dir $R/eval/judge_adv
    stage "lane 1 all done"
    ;;
  2)
    train $R/debater_sft debater_sft --data $DEBATER_DATA --epochs 1
    if [ ! -e $R/debates_test/DONE ]; then
      run debates_test python scamcourt/run_debate.py "${M[@]}" --split test --adapter $R/debater_sft/adapter \
        --output-dir $R/debates_test
      touch $R/debates_test/DONE
    fi
    run eval_zeroshot_plain env SC_ANSWER_PREFIX="$ZS_PREFIX" SC_GEN_PREFIX="$ZS_GEN_PREFIX" python scamcourt/evaluate.py "${M[@]}" --mode no_debate \
      --no-adapter --output-dir $R/eval/zeroshot_plain
    run eval_zeroshot_adv env SC_ANSWER_PREFIX="$ZS_PREFIX" SC_GEN_PREFIX="$ZS_GEN_PREFIX" python scamcourt/evaluate.py "${M[@]}" --mode adversarial \
      --no-adapter --output-dir $R/eval/zeroshot_adv
    # prompted self-debate baseline: untrained debaters (with the keep-your-side reminder) and untrained judge
    if [ ! -e $R/tf_debates_test/DONE ]; then
      run tf_debates_test env SC_ANSWER_PREFIX="$ZS_PREFIX" SC_GEN_PREFIX="$ZS_GEN_PREFIX" python scamcourt/run_debate.py "${M[@]}" --split test \
        --no-adapter --role-reminder --output-dir $R/tf_debates_test
      touch $R/tf_debates_test/DONE
    fi
    run eval_tf_selfdebate_full env SC_ANSWER_PREFIX="$ZS_PREFIX" SC_GEN_PREFIX="$ZS_GEN_PREFIX" python scamcourt/evaluate.py "${M[@]}" --mode debate \
      --no-adapter --debates $R/tf_debates_test/debates.jsonl --debate-view full --output-dir $R/eval/tf_selfdebate_full
    stage "lane 2 all done"
    ;;
  status)
    for d in debater_fold_a debater_fold_b debater_sft judge_sft classifier_sft; do
      printf "%-16s %s\n" $d "$([ -d $R/$d/adapter ] && echo done || (tail -1 $R/$d/train_log.jsonl 2>/dev/null || echo -))"
    done
    for d in debates_train_fold_a debates_train_fold_b debates_test tf_debates_test; do
      printf "%-22s %s lines%s\n" $d "$(wc -l < $R/$d/debates.jsonl 2>/dev/null || echo 0)" "$([ -e $R/$d/DONE ] && echo ' (done)')"
    done
    for d in $R/eval/*/; do [ -d "$d" ] && printf "%-28s %s rows\n" "$(basename $d)" "$(wc -l < $d/predictions.jsonl 2>/dev/null || echo 0)"; done
    ;;
  *) echo "usage: bash scamcourt/run_glm.sh 0|1|2|status" >&2; exit 2 ;;
esac
