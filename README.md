# Underage Moderation

`underage-moderation` is a public, inference-focused audio moderation repository for one narrow job: screening speech for underage risk before a human review or downstream enforcement step. The repository intentionally publishes only the runtime, policy logic, experiment tooling, and operating documentation needed to run the system safely in local environments or serverless execution.

Large weights, private review data, and training assets are intentionally excluded.

## Purpose

This repository is meant for moderation stacks that need a lightweight audio-first signal, not a full identity system.

Typical uses include:

- Pre-screening uploaded voice messages before they enter a moderator queue.
- Adding an underage-risk feature to an existing trust-and-safety pipeline.
- Running batch rechecks over previously stored audio.
- Returning a structured moderation response from a CLI, worker, or Lambda handler.

This project is intentionally scoped to **voice-based underage risk screening only**. It does not perform identity verification, speaker matching, or legal age verification.

## Public Release Boundary

Included in this public repository:

- Audio normalization and inference runtime code.
- A transparent decision policy with auditable thresholds.
- A CLI entrypoint for local execution.
- A Lambda entrypoint for file path, URL, or S3-triggered processing.
- A guarded offline experiment loop for threshold sweeps.
- A structured self-improvement logging utility for reviewer feedback and runtime failures.
- English-only technical documentation.

Intentionally excluded:

- Large model checkpoints and fine-tuned weights.
- Training sets and reviewer-labeled source media.
- Private evaluation reports and internal dashboards.
- Deployment bundles, private credentials, and environment-specific configuration.

## Core Runtime Flow

The runtime processes audio in five steps:

1. Normalize input audio to mono, 16 kHz WAV with `ffmpeg`.
2. Load a compatible speech checkpoint from a configured source.
3. Run inference to produce:
   - `age_score`
   - `child_probability`
   - `female_probability`
   - `male_probability`
4. Apply a small, auditable decision policy:
   - Flag when `age_score <= age_threshold`
   - Flag when `child_probability` exceeds the strongest adult probability by at least `child_probability_margin`
5. Return a structured payload containing input metadata, raw scores, policy hits, and runtime metadata.

The default `age_threshold` is `0.30`.

## Repository Layout

```text
.
├── cli.py
├── examples/
│   └── review_set.example.jsonl
├── lambda_function.py
├── model/
│   └── .gitignore
├── requirements.txt
├── scripts/
│   ├── log_learning.py
│   └── yolo_policy_sweep.py
├── tests/
│   ├── test_experiments.py
│   ├── test_learning.py
│   └── test_policy.py
└── underage_moderation/
    ├── __init__.py
    ├── audio.py
    ├── experiments.py
    ├── learning.py
    ├── model.py
    ├── pipeline.py
    └── policy.py
```

## Model Loading Strategy

The repository does not ship large weights. At runtime, model loading follows this order:

1. `UNDERAGE_MODEL_SOURCE`, if set.
2. The local `model/` directory, if compatible weights are present.
3. A default compatible remote checkpoint defined in `underage_moderation/model.py`.

For offline or tightly controlled deployments, place a compatible checkpoint in `model/`.

Expected local model files are typically:

- `config.json`
- `preprocessor_config.json`
- `vocab.json`
- `pytorch_model.bin` or `model.safetensors`

## Self-Improving Operations Loop

The runtime is intentionally small, so the improvement loop matters as much as the inference code.

The recommended operating pattern is:

1. Run moderation and capture structured output.
2. Compare flagged and allowed results against human review outcomes.
3. Log disagreements, model blind spots, and recurring runtime failures.
4. Promote repeated patterns into threshold changes, code changes, or reviewer guidance.
5. Re-evaluate changes offline before promoting them to production.

This repository includes `scripts/log_learning.py` to make that loop concrete.

Example:

```bash
python scripts/log_learning.py learning \
  --category correction \
  --area policy \
  --summary "Borderline juvenile voices were missed at the current threshold." \
  --details "Reviewers marked three clips as underage after the pipeline returned allow." \
  --suggested-action "Sweep higher age thresholds on the labeled holdout set." \
  --source reviewer \
  --tags threshold,false-negative
```

The script writes structured markdown entries to a local `.learnings/` directory. That directory is git-ignored on purpose so operational notes do not leak into the public repository by accident.

## YOLO Experiment Loop

In this repository, `YOLO` means **fast, reversible offline experimentation**. It does not mean blind production rollout, and it is not a reference to a computer-vision model family.

The goal of the YOLO loop is to make policy iteration cheap while keeping safety controls explicit:

1. Collect a labeled review set with human outcomes.
2. Reuse stored model scores instead of rerunning heavy inference for every threshold test.
3. Sweep candidate `age_threshold` and `child_probability_margin` values offline.
4. Rank policies by the metric that matches your risk posture, such as recall or false-positive rate.
5. Promote only reviewed winners into runtime configuration.

This repository includes `scripts/yolo_policy_sweep.py` for that workflow.

Example:

```bash
python scripts/yolo_policy_sweep.py examples/review_set.example.jsonl \
  --age-thresholds 0.26,0.28,0.30,0.32 \
  --child-margins 0.00,0.03,0.05 \
  --sort-by recall
```

The example review set is synthetic and exists only to document the file format.

Each JSONL row may be flat:

```json
{
  "id": "case-001",
  "label": true,
  "age_score": 0.24,
  "child_probability": 0.73,
  "female_probability": 0.17,
  "male_probability": 0.10
}
```

Or nested under `scores`:

```json
{
  "id": "case-002",
  "label": false,
  "scores": {
    "age_score": 0.68,
    "child_probability": 0.05,
    "female_probability": 0.44,
    "male_probability": 0.51
  }
}
```

## Local Setup

### 1. Install system dependencies

`ffmpeg` is required because every input is normalized before inference.

On macOS:

```bash
brew install ffmpeg
```

On Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
```

### 2. Create a Python environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Optional: pin a model source

```bash
export UNDERAGE_MODEL_SOURCE="/absolute/path/to/local-or-remote-model"
```

If `UNDERAGE_MODEL_SOURCE` is not set, the runtime falls back to `model/` and then to the default remote checkpoint.

## CLI Usage

Process one file:

```bash
python cli.py /path/to/audio.wav --json
```

Process a directory:

```bash
python cli.py /path/to/audio-directory --device cuda --max-duration 180
```

Tune the policy inline:

```bash
python cli.py /path/to/audio.wav \
  --age-threshold 0.28 \
  --child-margin 0.05 \
  --json
```

### CLI Output Shape

The JSON result has four blocks:

- `input`: source path, normalized path, sample rate, and duration.
- `scores`: raw model outputs used by the policy.
- `moderation`: decision, rule hits, and active thresholds.
- `runtime`: selected device and resolved model source.

## Lambda Usage

The Lambda entrypoint is `lambda_function.lambda_handler`.

Supported payload styles:

### Direct local path payload

```json
{
  "audio_path": "/tmp/sample.wav",
  "device": "cpu",
  "max_duration_seconds": 180
}
```

### URL payload

```json
{
  "audio_url": "https://example.com/sample.ogg",
  "age_threshold": 0.30,
  "child_probability_margin": 0.0
}
```

### S3-triggered execution

An S3 event record is also supported. The handler downloads the object to temporary storage, runs moderation, and returns the structured result.

### HTTP-style payload

If the event contains `body`, the handler treats it as an HTTP request and returns:

```json
{
  "statusCode": 200,
  "headers": {
    "Content-Type": "application/json"
  },
  "body": "{...json string...}"
}
```

## Configuration

### Environment variables

- `UNDERAGE_MODEL_SOURCE`: overrides the default model source resolution path.

### Policy inputs

- `age_threshold`: flag when predicted age score is at or below this threshold.
- `child_probability_margin`: require the child probability to exceed adult probabilities by this margin before that rule triggers.
- `max_duration_seconds`: truncate long files before inference to limit runtime and memory usage.
- `device`: choose `cpu`, `cuda`, or `mps`.

## Verification

Run the lightweight test suite:

```bash
python -m unittest discover -s tests -v
```

The tests cover:

- Rule-level moderation policy behavior.
- Review set loading and experiment ranking.
- Structured learning log generation.

## Operational Notes

- The first run may be slower because model assets can be downloaded or loaded at runtime.
- Audio normalization depends on `ffmpeg`, so any format that `ffmpeg` can decode can be screened.
- The policy is intentionally simple and auditable. Threshold tuning should be based on a labeled review set and explicit risk trade-offs.
- The public repository is inference-only. It does not publish training code, review data, or internal evaluation artifacts.

## Limitations

- Voice-based age estimation is probabilistic and should never be the sole enforcement signal.
- Audio quality, background noise, multiple speakers, and voice effects can materially shift model behavior.
- The output should be treated as a screening signal and reviewed alongside broader trust-and-safety context.

## Recommended Next Steps

- Add your preferred checkpoint through `UNDERAGE_MODEL_SOURCE` or the local `model/` directory.
- Build a reviewer-labeled holdout set and use it for threshold tuning.
- Use the self-improving log to capture false positives, false negatives, and recurring runtime issues.
- Use the YOLO sweep only offline, then promote reviewed winners into runtime configuration.
