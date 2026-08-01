"""
Milestone 2.1 entrypoint: generate the full synthetic book of business
(locations + attributes + policies) and write to the Silver layer.

Note: this writes to `data_pipeline/silver/`, not Bronze — this is
generated/derived data, not raw external source data, so it belongs at the
Silver stage of the medallion architecture per the roadmap.

Usage:
    python -m data_pipeline.synthetic.run_generation --n 20000
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from data_pipeline.synthetic.assign_attributes import assign_attributes
from data_pipeline.synthetic.generate_locations import generate_locations
from data_pipeline.synthetic.generate_policies import generate_policies

logger = logging.getLogger(__name__)

SILVER_ROOT = Path("data_pipeline/silver")


def run(n: int = 20_000, seed: int = 42) -> None:
    logger.info("Generating %d synthetic locations...", n)
    locations = generate_locations(n, seed=seed)

    logger.info("Assigning construction attributes and TIV...")
    locations = assign_attributes(locations, seed=seed + 1)

    logger.info("Generating policies...")
    policies = generate_policies(locations, seed=seed + 2)

    SILVER_ROOT.mkdir(parents=True, exist_ok=True)
    locations_path = SILVER_ROOT / "locations.parquet"
    policies_path = SILVER_ROOT / "policies.parquet"

    locations.to_parquet(locations_path, index=False)
    # peril_coverage is a list column — parquet handles this fine via arrow,
    # but flag it in case a downstream reader expects a flat schema.
    policies.to_parquet(policies_path, index=False)

    logger.info("Wrote %d locations to %s", len(locations), locations_path)
    logger.info("Wrote %d policies to %s", len(policies), policies_path)

    logger.info(
        "Summary — TIV: median=$%.0f, total=$%.0fM | metro distribution:\n%s",
        locations["tiv_usd"].median(),
        locations["tiv_usd"].sum() / 1e6,
        locations["metro_center"].value_counts().to_string(),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20_000, help="Number of synthetic locations")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(n=args.n, seed=args.seed)