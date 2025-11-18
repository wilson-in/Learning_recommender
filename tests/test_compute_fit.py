# tests/test_compute_fit.py
import math
import os
import sys

# Ensure the project root (parent of tests/) is on sys.path so imports work.
TESTS_DIR = os.path.dirname(__file__)                # .../learning-recommender/tests
PROJECT_ROOT = os.path.abspath(os.path.join(TESTS_DIR, os.pardir))  # .../learning-recommender
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app_free import compute_fit

def test_compute_fit_bounds_zero():
    # when cosine sim is 0, no popularity, and huge prereq penalty -> fit near 0
    fit, skill = compute_fit(0.0, prereq_penalty=20.0, level_bonus=0.0, popularity=0.0)
    assert fit >= 0.0
    assert fit <= 100.0

def test_compute_fit_simple_case():
    # cosine 0.8, no prereq penalty, full level bonus, popularity 8 -> expect positive reasonable fit
    fit, skill = compute_fit(0.8, prereq_penalty=0.0, level_bonus=10.0, popularity=8.0)
    # skill_sim_score = 0.8 * 60 = 48.0
    assert math.isclose(skill, 48.0, rel_tol=1e-3)
    assert 60.0 <= fit <= 72.0

def test_compute_fit_clamp_top():
    # very high cosine should clamp at 100
    fit, skill = compute_fit(2.0, prereq_penalty=0.0, level_bonus=10.0, popularity=10.0)
    assert fit == 100.0
