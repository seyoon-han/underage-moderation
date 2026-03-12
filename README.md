# Underage Moderation

`underage-moderation` is a lightweight audio moderation repository focused on one task: flagging speech that appears to contain an underage voice signal. The public repository contains only the runtime code, the inference pipeline, and the operational documentation needed to run the system locally or in AWS Lambda. Large model weights are intentionally not committed.

## Purpose

This repository is designed for moderation workflows where audio should be screened before it reaches a downstream review or enforcement step.

Typical use cases:

- Pre-screening uploaded voice messages.
- Adding an underage-risk signal to an existing moderation pipeline.
- Running a lightweight batch review pass over stored audio.
- Returning a machine-readable moderation decision from a Lambda function.

This project is intentionally scoped to **voice-based underage risk detection only**. It does not perform identity verification, speaker linking, or legal age verification.

## Repository Scope

Included in this public repository:

- Reusable inference and moderation pipeline code.
- A command-line entrypoint for local processing.
- A Lambda entrypoint for file path, URL, or S3-triggered execution.
- A `model/` placeholder directory for optional local checkpoint assets.
- English-only technical documentation.

Intentionally excluded:

- Large model weight files.
- Training datasets.
- Experimental scripts from the source workspace.
- Packaging artifacts and deployment bundles.

## How It Works

The runtime performs the following steps:

1. Normalize input audio with `ffmpeg` into mono, 16 kHz WAV.
2. Load a compatible speech checkpoint either from a local `model/` directory or from a configured remote source.
3. Run inference to obtain:
   - `age_score`
   - `child_probability`
   - `female_probability`
   - `male_probability`
4. Apply a simple moderation rule:
   - Flag if `age_score <= age_threshold`
   - Flag if `child_probability` is greater than the adult gender probabilities by at least the configured margin
5. Return a structured result with raw scores, policy settings, runtime metadata, and the final moderation decision.

The default age threshold is `0.30`.

## Repository Layout

```text
.
├── cli.py
├── lambda_function.py
├── model/
│   └── .gitignore
├── requirements.txt
└── underage_moderation/
    ├── __init__.py
    ├── audio.py
    ├── model.py
    └── pipeline.py
```

## Model Loading Strategy

The repository does not ship large weights. At runtime the loader uses the following order:

1. `UNDERAGE_MODEL_SOURCE` environment variable, if set.
2. Local `model/` directory, if it contains compatible weight files.
3. The default compatible remote checkpoint defined in `underage_moderation/model.py`.

For offline or tightly controlled deployments, place a compatible checkpoint inside `model/`.

Expected local model files are typically:

- `config.json`
- `preprocessor_config.json`
- `vocab.json`
- `pytorch_model.bin` or `model.safetensors`

## Local Setup

### 1. Install system dependencies

`ffmpeg` is required because the pipeline normalizes every input before inference.

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

If this variable is not set, the runtime automatically falls back to the local `model/` directory and then to the default remote checkpoint.

## CLI Usage

Process a single file:

```bash
python cli.py /path/to/audio.wav --json
```

Process a directory:

```bash
python cli.py /path/to/audio-directory --device cuda --max-duration 180
```

Tune the moderation policy:

```bash
python cli.py /path/to/audio.wav \
  --age-threshold 0.28 \
  --child-margin 0.05 \
  --json
```

### CLI Output

The JSON output includes four sections:

- `input`: normalized input metadata
- `scores`: raw model output values used by the policy
- `moderation`: final decision, triggered rules, and active threshold settings
- `runtime`: device and model source information

## Lambda Usage

The Lambda handler is `lambda_function.lambda_handler`.

It accepts any of the following payload styles.

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

### HTTP payload

If the event contains `body`, the handler treats it as an HTTP-style request and returns:

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

- `UNDERAGE_MODEL_SOURCE`: override the default model source resolution order.

### CLI and Lambda policy inputs

- `age_threshold`: flag when predicted age score is at or below this value.
- `child_probability_margin`: require the child probability to exceed adult probabilities by this margin before that rule triggers.
- `max_duration_seconds`: truncate long files before inference to limit runtime and memory usage.
- `device`: choose `cpu`, `cuda`, or `mps`.

## Operational Notes

- The first run may be slower because model assets can be loaded or downloaded at runtime.
- Every file is normalized through `ffmpeg`, so non-WAV input formats are supported as long as `ffmpeg` can decode them.
- The moderation rule is intentionally simple and auditable. Threshold tuning should be based on your own review set and risk tolerance.
- The public repository does not include training code or evaluation reports. It is intended as an inference-only moderation runtime.

## Limitations

- Voice-based age estimation is probabilistic and should not be used as the sole enforcement signal.
- Audio quality, background noise, multiple speakers, and effects processing can materially change model behavior.
- The moderation output should be treated as a screening signal and reviewed alongside other platform context.

## Recommended Next Steps

- Add your preferred checkpoint through `UNDERAGE_MODEL_SOURCE` or the local `model/` directory.
- Validate thresholds on a representative internal review set.
- Wire the JSON output into your existing moderation queue, reviewer tooling, or storage layer.
