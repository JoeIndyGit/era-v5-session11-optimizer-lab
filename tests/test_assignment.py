import math
from experiments.exp01_adam_by_hand import run as run_adam
from experiments.exp02_bias_correction import correction_ratio, first_below
from src.schedules import cosine_factor, wsd_factor


def test_manual_adam_matches_torch():
    assert run_adam()['max_absolute_error'] < 1e-12


def test_bias_correction_converges():
    t=first_below(.01,20)
    assert t > 1000
    assert abs(correction_ratio(t)-1) < .01


def test_wsd_has_stable_plateau():
    assert math.isclose(wsd_factor(19,300,20,240),1.0)
    assert math.isclose(wsd_factor(199,300,20,240),1.0)
    assert wsd_factor(299,300,20,240) == 0.05


def test_cosine_endpoint():
    assert cosine_factor(299,300,20,0.05) == 0.05
