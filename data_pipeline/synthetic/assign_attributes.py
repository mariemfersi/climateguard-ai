"""
Assign construction attributes (class, roof type, year built) and total
insured value (TIV) to synthetic locations.

METHODOLOGY NOTE:
The distributional shape here reflects a well-documented, verifiable piece
of real Florida regulatory history: Hurricane Andrew (1992) exposed severe
weaknesses in South Florida construction, leading to major building-code
reform, culminating in the statewide Florida Building Code (effective
2002), which substantially raised wind-resistance requirements (hip roofs,
impact-rated openings, stronger connections). This module encodes that
real historical inflection point as a `year_built` breakpoint affecting the
construction_class / roof_type distributions.

The SPECIFIC percentage splits below (e.g. "60% masonry pre-2002") are
documented MODELING ASSUMPTIONS reflecting general domain knowledge of
Florida residential construction, not figures pulled from a specific cited
statistical source. Before treating this project's backtest results as
representative of real market loss ratios, these splits should be
validated/replaced against an actual cited source (e.g. a state-level
building-stock survey) — flagged here explicitly rather than silently
presented as precise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Florida Building Code took effect in 2002, substantially raising wind
# design requirements statewide following Hurricane Andrew (1992).
_CODE_REFORM_YEAR = 2002

_YEAR_BUILT_BINS = [
    (1900, 1959, 0.05),
    (1960, 1979, 0.15),
    (1980, 1994, 0.20),
    (1995, 2001, 0.15),
    (2002, 2010, 0.25),
    (2011, 2024, 0.20),
]

_CONSTRUCTION_CLASSES = ["frame", "masonry_cbs", "reinforced_concrete", "masonry_veneer"]
_PRE_CODE_CONSTRUCTION_WEIGHTS = [0.35, 0.45, 0.10, 0.10]
_POST_CODE_CONSTRUCTION_WEIGHTS = [0.15, 0.55, 0.20, 0.10]

_ROOF_TYPES = ["hip", "gable", "flat"]
_PRE_CODE_ROOF_WEIGHTS = [0.30, 0.60, 0.10]
_POST_CODE_ROOF_WEIGHTS = [0.65, 0.30, 0.05]


def _sample_year_built(n: int, rng: np.random.Generator) -> np.ndarray:
    bin_weights = np.array([b[2] for b in _YEAR_BUILT_BINS])
    bin_probs = bin_weights / bin_weights.sum()
    bin_idx = rng.choice(len(_YEAR_BUILT_BINS), size=n, p=bin_probs)

    years = np.empty(n, dtype=int)
    for i, idx in enumerate(bin_idx):
        start, end, _ = _YEAR_BUILT_BINS[idx]
        years[i] = rng.integers(start, end + 1)
    return years


def assign_attributes(locations: pd.DataFrame, seed: int = 43) -> pd.DataFrame:
    """
    Assign construction_class, roof_type, year_built, and tiv_usd to each
    location, with construction_class/roof_type distributions shifting at
    the 2002 Florida Building Code reform breakpoint.

    Args:
        locations: output of generate_locations() — must have a
            'location_id' column; row count determines n.

    Returns:
        locations with four new columns added.
    """
    n = len(locations)
    rng = np.random.default_rng(seed)

    year_built = _sample_year_built(n, rng)
    is_post_code = year_built >= _CODE_REFORM_YEAR

    construction_class = np.empty(n, dtype=object)
    roof_type = np.empty(n, dtype=object)

    post_idx = np.where(is_post_code)[0]
    pre_idx = np.where(~is_post_code)[0]

    if len(post_idx) > 0:
        construction_class[post_idx] = rng.choice(
            _CONSTRUCTION_CLASSES, size=len(post_idx), p=_POST_CODE_CONSTRUCTION_WEIGHTS
        )
        roof_type[post_idx] = rng.choice(
            _ROOF_TYPES, size=len(post_idx), p=_POST_CODE_ROOF_WEIGHTS
        )
    if len(pre_idx) > 0:
        construction_class[pre_idx] = rng.choice(
            _CONSTRUCTION_CLASSES, size=len(pre_idx), p=_PRE_CODE_CONSTRUCTION_WEIGHTS
        )
        roof_type[pre_idx] = rng.choice(_ROOF_TYPES, size=len(pre_idx), p=_PRE_CODE_ROOF_WEIGHTS)

    # TIV: lognormal, right-skewed (many modest homes, a long tail of high-
    # value properties), truncated to a plausible residential range.
    tiv_raw = rng.lognormal(mean=np.log(300_000), sigma=0.55, size=n)
    tiv_usd = np.clip(tiv_raw, 75_000, 5_000_000).round(-3)  # round to nearest $1,000

    result = locations.copy()
    result["year_built"] = year_built
    result["construction_class"] = construction_class
    result["roof_type"] = roof_type
    result["tiv_usd"] = tiv_usd
    return result