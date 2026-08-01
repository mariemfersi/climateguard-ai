"""
Milestone 2.2 entrypoint: load real HURDAT2 tracks + the synthetic book,
generate claims, write to Silver, and print the Hurricane Ian sanity check
required by the roadmap's Milestone 2.2 validation checklist.

Usage:
    python -m data_pipeline.synthetic.run_claims_generation
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from data_pipeline.synthetic.claims_generator import generate_claims

logger = logging.getLogger(__name__)

BRONZE_HURDAT2_PATH = Path("data_pipeline/bronze/hurdat2/tracks.parquet")
SILVER_LOCATIONS_PATH = Path("data_pipeline/silver/locations.parquet")
SILVER_POLICIES_PATH = Path("data_pipeline/silver/policies.parquet")
SILVER_CLAIMS_PATH = Path("data_pipeline/silver/claims.parquet")


def _print_hurricane_ian_sanity_check(claims: pd.DataFrame) -> None:
    """
    Hurricane Ian (2022) is one of the costliest hurricanes in US history
    (real insured losses widely reported in the tens of billions of
    dollars). This is the order-of-magnitude sanity check the roadmap's
    Milestone 2.2 validation checklist calls for: our synthetic book is a
    small (thousands of locations) slice of the real Florida market, so we
    should NOT expect our simulated Ian loss to match the real market total
    — but it should be a plausible, non-trivial fraction of our synthetic
    book's total insured value, not near-zero and not exceeding total TIV.
    """
    ian_claims = claims[claims["storm_name"].str.upper() == "IAN"]
    if ian_claims.empty:
        print(
            "\nNo 'IAN' storm found in the filtered claims — this can happen if "
            "your HURDAT2 file's coverage range doesn't include 2022, or if "
            "your locations.parquet was generated before this run. Not "
            "necessarily an error, but worth checking manually."
        )
        return

    total_incurred = ian_claims["incurred_loss_usd"].sum()
    total_paid = ian_claims["paid_loss_usd"].sum()
    n_locations_hit = ian_claims["location_id"].nunique()
    max_wind = ian_claims["max_wind_experienced_kt"].max()

    print("\n--- Hurricane Ian (2022) sanity check ---")
    print(f"Locations with a simulated Ian claim: {n_locations_hit:,}")
    print(f"Max wind experienced by any location: {max_wind:.0f} kt")
    print(f"Total simulated incurred loss:         ${total_incurred:,.0f}")
    print(f"Total simulated paid loss (post-ded.):  ${total_paid:,.0f}")
    print(
        "Sanity expectation: this should be a real, non-trivial dollar figure "
        "(not near-zero), concentrated in Southwest Florida locations (Lee/"
        "Collier counties) if Ian's real track is reflected correctly — Ian's "
        "real landfall was near Fort Myers/Lee County."
    )


def run() -> None:
    if not BRONZE_HURDAT2_PATH.exists():
        raise FileNotFoundError(
            f"{BRONZE_HURDAT2_PATH} not found — run "
            f"`python -m data_pipeline.ingestion.hurdat2.run_ingestion` first (Milestone 1.1)."
        )
    if not SILVER_LOCATIONS_PATH.exists() or not SILVER_POLICIES_PATH.exists():
        raise FileNotFoundError(
            "Silver locations/policies not found — run "
            "`python -m data_pipeline.synthetic.run_generation` first (Milestone 2.1)."
        )

    logger.info("Loading HURDAT2 tracks, locations, and policies...")
    hurdat2_tracks = pd.read_parquet(BRONZE_HURDAT2_PATH)
    locations = pd.read_parquet(SILVER_LOCATIONS_PATH)
    policies = pd.read_parquet(SILVER_POLICIES_PATH)

    logger.info(
        "Loaded %d storm-track records, %d locations, %d policies",
        len(hurdat2_tracks),
        len(locations),
        len(policies),
    )

    claims = generate_claims(locations, policies, hurdat2_tracks)

    SILVER_CLAIMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    claims.to_parquet(SILVER_CLAIMS_PATH, index=False)
    logger.info("Wrote %d claims to %s", len(claims), SILVER_CLAIMS_PATH)

    if claims.empty:
        print("\nWARNING: zero claims generated — check thresholds and data inputs.")
        return

    total_tiv = locations["tiv_usd"].sum()
    total_incurred = claims["incurred_loss_usd"].sum()
    print("\n--- Overall summary ---")
    print(f"Total claim records: {len(claims):,}")
    print(f"Distinct storms producing a claim: {claims['storm_id'].nunique():,}")
    print(f"Total synthetic book TIV: ${total_tiv:,.0f}")
    print(f"Total simulated incurred loss (all storms, all years): ${total_incurred:,.0f}")

    print("\nTop 10 storms by total incurred loss:")
    top_storms = (
        claims.groupby(["storm_id", "storm_name"])["incurred_loss_usd"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )
    print(top_storms.to_string())

    _print_hurricane_ian_sanity_check(claims)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()