"""
Generate one active policy per synthetic location.

METHODOLOGY NOTE — Florida-specific hurricane deductible convention:
Florida homeowners policies are required to disclose a separate hurricane
deductible, expressed as a PERCENTAGE of dwelling coverage (commonly 2%,
5%, or 10%), rather than a flat dollar amount as with standard perils. This
is real, verifiable Florida insurance-market practice, not a modeling
assumption — see Fla. Stat. § 627.701. The specific weight distribution
across {2%, 5%, 10%} below IS a modeling assumption (documented as such),
but the percentage-of-dwelling-coverage structure itself is factually
grounded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_HURRICANE_DEDUCTIBLE_PCTS = [0.02, 0.05, 0.10]
_HURRICANE_DEDUCTIBLE_WEIGHTS = [0.30, 0.45, 0.25]

_POLICY_YEAR_RANGE = (2018, 2023)  # effective dates sampled within this range


def generate_policies(locations: pd.DataFrame, seed: int = 44) -> pd.DataFrame:
    """
    Generate one policy per row in `locations` (which must already have
    tiv_usd from assign_attributes()).

    Returns:
        DataFrame with columns [policy_id, location_id, effective_date,
        expiry_date, limit_usd, deductible_pct, deductible_usd,
        peril_coverage].
    """
    if "tiv_usd" not in locations.columns:
        raise ValueError(
            "locations must have a 'tiv_usd' column — run assign_attributes() first."
        )

    n = len(locations)
    rng = np.random.default_rng(seed)

    # Standard homeowners practice: insure to (approximately) full
    # replacement cost, with minor variation reflecting real-world
    # underinsurance.
    coverage_ratio = np.clip(rng.normal(loc=0.97, scale=0.05, size=n), 0.75, 1.0)
    limit_usd = (locations["tiv_usd"].to_numpy() * coverage_ratio).round(-3)

    deductible_pct = rng.choice(
        _HURRICANE_DEDUCTIBLE_PCTS, size=n, p=_HURRICANE_DEDUCTIBLE_WEIGHTS
    )
    deductible_usd = (limit_usd * deductible_pct).round(-2)

    start_year, end_year = _POLICY_YEAR_RANGE
    eff_years = rng.integers(start_year, end_year + 1, size=n)
    eff_months = rng.integers(1, 13, size=n)
    eff_days = rng.integers(1, 28, size=n)  # avoid month-length edge cases
    effective_dates = pd.to_datetime(
        {"year": eff_years, "month": eff_months, "day": eff_days}
    )
    expiry_dates = effective_dates + pd.DateOffset(years=1)

    return pd.DataFrame(
        {
            "policy_id": [f"POL{i:07d}" for i in range(1, n + 1)],
            "location_id": locations["location_id"].to_numpy(),
            "effective_date": effective_dates,
            "expiry_date": expiry_dates,
            "limit_usd": limit_usd,
            "deductible_pct": deductible_pct,
            "deductible_usd": deductible_usd,
            "peril_coverage": [["wind"]] * n,
        }
    )