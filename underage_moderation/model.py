from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoFeatureExtractor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)

DEFAULT_MODEL_SOURCE = "audeering/wav2vec2-large-robust-6-ft-age-gender"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LOCAL_MODEL_DIR = REPOSITORY_ROOT / "model"


class ModelHead(nn.Module):
    def __init__(self, config, num_labels: int):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features, **kwargs):
        output = self.dropout(features)
        output = self.dense(output)
        output = torch.tanh(output)
        output = self.dropout(output)
        return self.out_proj(output)


class AgeGenderModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.age = ModelHead(config, 1)
        self.gender = ModelHead(config, 3)
        self.init_weights()

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0].mean(dim=1)
        age_logits = self.age(hidden_states)
        gender_probabilities = torch.softmax(self.gender(hidden_states), dim=1)
        return hidden_states, age_logits, gender_probabilities


@dataclass(frozen=True)
class RuntimeResources:
    model: AgeGenderModel
    feature_extractor: AutoFeatureExtractor
    device: torch.device
    model_source: str


def resolve_model_source() -> str:
    configured_source = os.getenv("UNDERAGE_MODEL_SOURCE")
    if configured_source:
        return configured_source

    has_local_weights = any(
        (LOCAL_MODEL_DIR / filename).exists()
        for filename in ("pytorch_model.bin", "model.safetensors")
    )
    if has_local_weights:
        return str(LOCAL_MODEL_DIR)

    return DEFAULT_MODEL_SOURCE


def select_device(preferred_device: str | None = None) -> torch.device:
    if preferred_device:
        device_name = preferred_device.lower()
        if device_name == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if (
            device_name == "mps"
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            return torch.device("mps")
        if device_name == "cpu":
            return torch.device("cpu")
        raise RuntimeError(f"Requested device is not available: {preferred_device}")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@lru_cache(maxsize=8)
def load_runtime(preferred_device: str | None = None) -> RuntimeResources:
    model_source = resolve_model_source()
    device = select_device(preferred_device)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_source)
    model = AgeGenderModel.from_pretrained(model_source).to(device)
    model.eval()
    return RuntimeResources(
        model=model,
        feature_extractor=feature_extractor,
        device=device,
        model_source=model_source,
    )


def predict_scores(
    audio: np.ndarray,
    sample_rate: int,
    preferred_device: str | None = None,
) -> tuple[dict[str, float], RuntimeResources]:
    runtime = load_runtime(preferred_device=preferred_device)
    inputs = runtime.feature_extractor(
        audio,
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding=True,
    )
    input_values = inputs["input_values"].to(runtime.device)

    with torch.inference_mode():
        _, age_logits, gender_probabilities = runtime.model(input_values)

    age_score = float(age_logits.detach().cpu().squeeze().item())
    gender_values = gender_probabilities.detach().cpu().squeeze(0).tolist()
    estimated_age_years = max(0.0, min(100.0, age_score * 100.0))
    scores = {
        "age_score": age_score,
        "estimated_age_years": estimated_age_years,
        "child_probability": float(gender_values[0]),
        "female_probability": float(gender_values[1]),
        "male_probability": float(gender_values[2]),
    }
    return scores, runtime
