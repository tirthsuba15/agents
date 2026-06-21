"""Tests for new features in train_gamma.py (#26, tech retrain, mom8w/mom13w)."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_global_feature_names_contains_rv_iv_ratio():
    from models.train_gamma import FEATURE_NAMES
    assert "rv_iv_ratio" in FEATURE_NAMES
