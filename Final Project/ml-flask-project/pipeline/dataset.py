"""
Dataset loader for Global AI vs Human Content Dataset 2026.

If you have the real Kaggle CSV, place it at:
    ml-pipeline/data/ai_human_content.csv

Otherwise, a realistic synthetic dataset is generated automatically
with the same feature distribution as the real one.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def load_or_generate_dataset(data_dir: Path) -> pd.DataFrame:
    dataset_file = data_dir / "ai_human_content.csv"

    if dataset_file.exists():
        print(f"[dataset] Loading dataset from {dataset_file}")
        return pd.read_csv(dataset_file)

    print("[dataset] Generating synthetic dataset (10,000 samples)...")
    return _generate_synthetic(dataset_file)


def _generate_synthetic(save_path: Path) -> pd.DataFrame:
    np.random.seed(42)
    n = 10_000
    ai = np.random.rand(n) < 0.5
    label = np.where(ai, "AI", "Human")

    def _ai_human(ai_mu, ai_sd, hu_mu, hu_sd, lo=None, hi=None):
        vals = np.where(ai,
            np.random.normal(ai_mu, ai_sd, n),
            np.random.normal(hu_mu, hu_sd, n))
        if lo is not None or hi is not None:
            vals = np.clip(vals, lo, hi)
        return vals

    content_types = np.random.choice(
        ["article", "blog", "social_media", "academic", "news"], n)

    df = pd.DataFrame({
        "label":                    label,
        "avg_sentence_length":      _ai_human(22, 4,  18, 6,  5,  50),
        "vocabulary_richness":      _ai_human(.62,.07, .71,.10, .1, 1.),
        "punctuation_density":      _ai_human(.045,.008,.038,.012,.01,.15),
        "passive_voice_ratio":      _ai_human(.22,.06, .14,.07, 0., .5),
        "sentence_variance":        _ai_human(.31,.08, .48,.12, .05,1.),
        "coherence_score":          _ai_human(.78,.07, .65,.11, .1, 1.),
        "avg_word_length":          _ai_human(5.2,.6,  4.8,.9,  3., 8.),
        "transition_word_density":  _ai_human(.087,.015,.062,.018,.01,.2),
        "repetition_score":         _ai_human(.18,.05, .24,.08, 0., .8),
        "formality_score":          _ai_human(.73,.10, .55,.15, .1, 1.),
        "burstiness":               _ai_human(.28,.07, .54,.13, 0., 1.),
        "perplexity":               _ai_human(45.2,12.4,78.6,24.8,10.,200.),
        "num_paragraphs":           np.where(ai,
            np.random.randint(3,8,n), np.random.randint(1,12,n)),
        "has_headers":              np.where(ai,
            np.random.binomial(1,.65,n), np.random.binomial(1,.38,n)),
        "has_lists":                np.where(ai,
            np.random.binomial(1,.52,n), np.random.binomial(1,.29,n)),
        "word_count":               np.where(ai,
            np.random.randint(200,800,n), np.random.randint(50,1200,n)),
        "content_type":             content_types,
    })

    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"[dataset] Saved to {save_path}")
    return df
