# MechScam & ScamCourt

Data and code for **"Beyond Risk Cues: A Mechanism-Level Benchmark and Verifiable Debate for Scam Video Detection on the Web"**.

- **MechScam**: 6,614 YouTube videos annotated with a binary label (Scam / Non-scam), one of twelve mechanism-level categories (six scam mechanisms, six non-scam categories), evidence-grounded rationales, and potential scam elements with their qualifying context.
- **Adversarial argument set**: three wrong-side arguments per video, built only from genuine video evidence, used to measure how often a judge is misled.
- **ScamCourt**: a verifiable-citation debate framework. A trained prosecution and defense argue over the same evidence, citations are checked against the sources, and a trained judge rules on the transcript.

## Repository layout

```
data/
  mechscam/
    manifest.jsonl            one row per video: sample_id, collection, video_id, label, category, sample path
    samples/<collection>/<video_id>/sample.json
    taxonomy.json             the 12 categories and their definitions
    policy_guide.json         research paraphrase of the platform spam/scam policy used in annotation
    dataset_info.json
  splits/
    scamcourt_v3_seed42.json  train (5,290) / test (1,324) split used in all experiments
    exclusions.json           5 training videos excluded from ScamCourt training data
  arguments/
    teacher_arguments/        per video: 3 wrong-side arguments (opening, rebuttal, reference-side reply)
    true_openings/            per video: reference-side opening in the advocate register
    qwen3_vl_8b_scores/       zero-shot Qwen3-VL-8B scores used to select the strongest argument
  adversarial_test.jsonl      3,972 test arguments (3 per test video; qwen_selected marks the one used)
scamcourt/                    ScamCourt training, debate, evaluation and API baselines
scripts/                      prompt formats, teacher-argument generation, SFT export, frame download
```

## Data

| | Videos | Scam | Non-scam |
|---|---|---|---|
| All | 6,614 | 2,184 | 4,430 |
| Train | 5,290 | 1,748 | 3,542 |
| Test | 1,324 | 436 | 888 |

Categories: S01 Reward Claims, S02 Trading Bots, S03 Platform Investments, S04 Mining and Nodes, S05 Token Investments, S06 Recruitment Commissions; N01 Scam Awareness and Investigation, N02 Financial News and Analysis, N03 Education and Tutorials, N04 Products and Services, N05 Rewards and Promotions, N06 General Content. Definitions are in `data/mechscam/taxonomy.json` (`paper_name` gives the names used in the paper; `name` and `definition` are the wording used in the prompts).

Each `sample.json` contains:

- `title`, `description`, `asr` (time-aligned transcript), and `text_sources`: the text split into addressable segments (`TITLE_0001`, `DESCRIPTION_0001`, `ASR_0003`, ...) that annotations and arguments cite.
- `frames`: up to 64 uniformly sampled frames (`frame_id`, file name, sha256); `frame_sets.label_evidence` lists the frames used in annotation, from which 12 are sampled uniformly as model input.
- `potential_elements`: candidate risk cues from text and frames, each with its evidence, surrounding context and limitations, plus context notes.
- `annotation`: `binary_label`, `primary_subtype` (category), `content_role`, `contextual_rationale` (binary decision), `fine_grained_rationale` (category), and the cited `evidence`.

Hard negatives are non-scam test videos with at least one potential scam element (543 videos).

E-mail addresses and phone numbers appearing in titles, descriptions and transcripts are replaced by `[EMAIL]` and `[PHONE]` throughout the release (sources, annotations and arguments alike, so every citation remains a verbatim excerpt).

Annotations are research descriptions of video content. They do not establish legal culpability or verified financial outcomes for any person, channel or project.

## Data access and copyright

The videos and their frames remain the property of their creators and are subject to YouTube's Terms of Service, so **this repository does not redistribute videos or frames**. We release the YouTube video IDs, our annotations, and the text needed to reproduce the experiments.

- **Rebuild the frames yourself.** `python scripts/download_frames.py --workers 4` downloads each video by its ID (`https://www.youtube.com/watch?v=<video_id>`) and extracts its frames; it is resumable and logs unavailable videos to `work/download_frames.jsonl`. Frames are expected at `data/mechscam/samples/<collection>/<video_id>/frames/000001.jpg, 000002.jpg, ...`, with the number of frames given by `frame_count` in the manifest, sampled uniformly over the video (frame `k` of `N` is decoded frame `int(k * (T - 1) / (N - 1))` of `T`). Some videos have been removed from YouTube since collection, and re-encoded downloads may differ slightly from the frames used in the paper.
- **Request the original frames.** The original frames are available to researchers for non-commercial research upon request to the authors. Please contact the authors from an institutional address with your name, affiliation, and intended use, and confirm that you will (1) use the data only for non-commercial research, (2) not redistribute the frames or any part of them, and (3) delete them on request.
- **Takedown.** If you are the owner of a video in MechScam and want it removed, please contact the authors with the video ID.

The two source collections are VidScam (`vidscamnet` in the data, 3,079 videos) and YouTube\_Scam (`youtube_scam`, 3,535 videos); please also cite them if you use MechScam. Transcripts were produced with Whisper large-v3, with non-English speech translated into English.

## Setup

```bash
pip install -r requirements.txt          # Python >= 3.10, CUDA GPU for training and local inference
# backbones (local directories)
huggingface-cli download Qwen/Qwen3-VL-8B-Instruct --local-dir models/Qwen3-VL-8B-Instruct
huggingface-cli download zai-org/GLM-4.1V-9B-Thinking --local-dir models/GLM-4.1V-9B-Thinking
```

Then place the frames as described in the data access section (`python scripts/download_frames.py --limit 5` to try a few videos first).

Offline tests (no GPU, no API; they build a small synthetic project):

```bash
python -m unittest discover -s scamcourt/tests
python -m unittest discover -s scripts -p "test_*.py"
```

## Reproducing the experiments

All commands run from the repository root. Long jobs can be interrupted and rerun; finished trainings are skipped and debates/evaluations resume.

**1. Training data** (CPU, minutes): cold-start SFT files for the debater and judge, the two cross-fitting folds, and the SFT-classifier baseline data.

```bash
bash scamcourt/prepare_data.sh
```

**2. ScamCourt with Qwen3-VL-8B** (one 32 GB GPU): debater SFT, test and cross-fitted training debates, judge SFT, SFT classifier, zero-shot and prompted self-debate baselines, and all evaluations (with and without debate, and under the adversarial arguments).

```bash
mkdir -p logs && nohup bash scamcourt/run_all.sh > logs/run_all.log 2>&1 &
```

Results are written to `scamcourt/runs/eval/<setting>/` (`predictions.jsonl`, `metrics.json`). The ablation variants correspond to `judge_plain` (without debate at inference), `classifier_plain` (without debate training) and `zeroshot_plain` (without any training); the adversarial runs are `*_adv`.

**3. ScamCourt with GLM-4.1V-9B** (three GPUs; same data, folds, prompts and hyperparameters):

```bash
export GLM=models/GLM-4.1V-9B-Thinking
for g in 0 1 2; do CUDA_VISIBLE_DEVICES=$g nohup bash scamcourt/run_glm.sh $g > logs/glm_lane$g.log 2>&1 & done
bash scamcourt/run_glm.sh status
```

**4. API models** (direct judgment, self-debate, one-sided advocacy, adversarial arguments) through any OpenAI-compatible endpoint:

```bash
read -rsp 'API key: ' OPENAI_API_KEY; export OPENAI_API_KEY; echo
python scamcourt/api_baselines.py --model <model> --probe
python scamcourt/api_baselines.py --model <model> --workers 8 --attack-arguments qwen_selected \
  --output-dir scamcourt/runs/api/<model>          # add --base-url for other providers
```

### Regenerating the teacher arguments (optional)

The released `data/arguments/` are the exact arguments used in the paper. To regenerate them with an API model:

```bash
python scripts/generate_scamcourt_false_arguments.py --all --execute --model <model>   # wrong-side arguments
python scripts/generate_scamcourt_true_openings.py --all --execute --model <model>     # reference-side openings
python scripts/score_scamcourt_false_arguments.py                         # zero-shot selector scores (GPU)
python scamcourt/build_adversarial_testset.py                             # data/adversarial_test.jsonl
```

## License

Code: MIT (see `LICENSE`). Annotations and arguments: CC BY-NC 4.0, for non-commercial research only. Video content belongs to its creators.
