import math
import torch
from experiments.exp01_adam_by_hand import run as run_adam
from experiments.exp02_bias_correction import correction_ratio, first_below
from src.schedules import cosine_factor, wsd_factor
from src.tiny_transformer import TinyTransformerLM, make_synthetic_language


def test_manual_adam_matches_torch():
    assert run_adam()['max_absolute_error'] < 1e-12


def test_bias_correction_converges_under_one_percent():
    t = first_below(.01, 20)
    assert t == 3916
    assert abs(correction_ratio(t) - 1) < .01


def test_wsd_has_stable_plateau_and_decay_endpoint():
    assert math.isclose(wsd_factor(19, 300, 20, 240), 1.0)
    assert math.isclose(wsd_factor(199, 300, 20, 240), 1.0)
    assert wsd_factor(299, 300, 20, 240) == 0.05


def test_cosine_endpoint():
    assert cosine_factor(299, 300, 20, 0.05) == 0.05


def test_cosine_and_wsd_share_same_linear_warmup():
    for zero_based_step in range(20):
        c = cosine_factor(zero_based_step, 300, 20, 0.05)
        w = wsd_factor(zero_based_step, 300, 20, 210, 0.05)
        assert math.isclose(c, w, rel_tol=0, abs_tol=1e-12)


def test_synthetic_language_dependency_is_exact():
    x = make_synthetic_language(n_sequences=16, seq_len=12)
    assert torch.equal(x[:, 1:], (x[:, :-1] + 1) % 32)


def test_tiny_transformer_output_shape():
    model = TinyTransformerLM()
    x = make_synthetic_language(n_sequences=4, seq_len=16)[:, :-1]
    logits = model(x)
    assert logits.shape == (4, 15, 32)


def test_tiny_transformer_is_causal_for_prefix_logits():
    model = TinyTransformerLM(dropout=0.0).eval()
    x = make_synthetic_language(n_sequences=1, seq_len=16)
    altered = x.clone()
    altered[:, 10:] = (altered[:, 10:] + 7) % 32
    with torch.no_grad():
        a = model(x[:, :-1])
        b = model(altered[:, :-1])
    # Future-token changes must not affect logits at positions strictly before the change.
    assert torch.allclose(a[:, :10], b[:, :10], atol=1e-6, rtol=1e-6)


def test_readme_artifact_links_resolve():
    import re
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    text = (root / "README.md").read_text(encoding="utf-8")
    refs = sorted(set(re.findall(r"\((artifacts/[^)]+)\)", text)))
    assert refs, "README should link to generated evidence artifacts"
    missing = [ref for ref in refs if not (root / ref).exists()]
    assert not missing, f"Missing README artifacts: {missing}"


def test_schedule_tuning_and_evaluation_seeds_are_disjoint():
    from experiments.exp04_cosine_vs_wsd import TUNE_SEEDS, EVAL_SEEDS
    assert len(TUNE_SEEDS) >= 3
    assert len(EVAL_SEEDS) >= 3
    assert set(TUNE_SEEDS).isdisjoint(EVAL_SEEDS)
