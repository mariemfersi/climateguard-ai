"""
Great Expectations validation suites for the Silver-layer tables produced
by Phase 2 (locations, policies, claims).

Uses the Great Expectations 1.x "Fluent" API (ephemeral, in-memory context
— no persisted GE project config needed for this vertical slice). Each
`build_*_suite()` function returns an unregistered ExpectationSuite;
`validate_dataframe()` handles registering it with a fresh ephemeral
context and running it.

Usage:
    python -m data_pipeline.great_expectations.validate_silver
"""

from __future__ import annotations

import logging

import great_expectations as gx
import great_expectations.expectations as gxe
import pandas as pd

logger = logging.getLogger(__name__)

# Matches data_pipeline/synthetic/generate_locations.py's FLORIDA_BBOX.
_FLORIDA_LAT_RANGE = (24.3, 31.1)
_FLORIDA_LON_RANGE = (-87.7, -79.7)
_VALID_CONSTRUCTION_CLASSES = [
    "frame",
    "masonry_cbs",
    "reinforced_concrete",
    "masonry_veneer",
]
_VALID_ROOF_TYPES = ["hip", "gable", "flat"]
_VALID_DEDUCTIBLE_PCTS = [0.02, 0.05, 0.10]


def build_locations_suite() -> gx.ExpectationSuite:
    suite = gx.ExpectationSuite(name="locations_suite")
    suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column="location_id"))
    suite.add_expectation(gxe.ExpectColumnValuesToBeUnique(column="location_id"))
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeBetween(
            column="lat",
            min_value=_FLORIDA_LAT_RANGE[0],
            max_value=_FLORIDA_LAT_RANGE[1],
        )
    )
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeBetween(
            column="lon",
            min_value=_FLORIDA_LON_RANGE[0],
            max_value=_FLORIDA_LON_RANGE[1],
        )
    )
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeInSet(
            column="construction_class", value_set=_VALID_CONSTRUCTION_CLASSES
        )
    )
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeInSet(column="roof_type", value_set=_VALID_ROOF_TYPES)
    )
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeBetween(
            column="year_built", min_value=1900, max_value=2024
        )
    )
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeBetween(
            column="tiv_usd", min_value=75_000, max_value=5_000_000
        )
    )
    return suite


def build_policies_suite() -> gx.ExpectationSuite:
    suite = gx.ExpectationSuite(name="policies_suite")
    suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column="policy_id"))
    suite.add_expectation(gxe.ExpectColumnValuesToBeUnique(column="policy_id"))
    suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column="location_id"))
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeInSet(
            column="deductible_pct", value_set=_VALID_DEDUCTIBLE_PCTS
        )
    )
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeBetween(column="limit_usd", min_value=1)
    )
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeBetween(column="deductible_usd", min_value=0)
    )
    suite.add_expectation(
        gxe.ExpectColumnPairValuesAToBeGreaterThanB(
            column_A="expiry_date", column_B="effective_date"
        )
    )
    return suite


def build_claims_suite() -> gx.ExpectationSuite:
    suite = gx.ExpectationSuite(name="claims_suite")
    suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column="claim_id"))
    suite.add_expectation(gxe.ExpectColumnValuesToBeUnique(column="claim_id"))
    suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column="location_id"))
    suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column="policy_id"))
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeBetween(
            column="damage_ratio", min_value=0.0, max_value=1.0
        )
    )
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeBetween(column="incurred_loss_usd", min_value=0)
    )
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeBetween(column="paid_loss_usd", min_value=0)
    )
    # incurred must always be >= paid (deductible/limit only ever reduce, never increase, payout).
    suite.add_expectation(
        gxe.ExpectColumnPairValuesAToBeGreaterThanB(
            column_A="incurred_loss_usd", column_B="paid_loss_usd", or_equal=True
        )
    )
    return suite


def validate_dataframe(df: pd.DataFrame, suite_builder, asset_name: str) -> tuple[bool, object]:
    """
    Build and run a validation suite against `df` using a fresh ephemeral
    GX context.

    IMPORTANT: `suite_builder` is a CALLABLE (e.g. `build_locations_suite`),
    not an already-constructed ExpectationSuite. Great Expectations 1.x
    requires an active context to exist (via a global project-manager
    singleton) BEFORE any ExpectationSuite/Expectation objects are
    constructed — constructing the suite before the context exists raises
    `DataContextRequiredError`. Accepting a builder function guarantees the
    correct order (context first, suite second) every time.

    Returns:
        (success: bool, full_result_object)
    """
    context = gx.get_context(mode="ephemeral")
    suite = suite_builder()

    data_source = context.data_sources.add_pandas(f"{asset_name}_source")
    data_asset = data_source.add_dataframe_asset(name=asset_name)
    batch_definition = data_asset.add_batch_definition_whole_dataframe(
        f"{asset_name}_batch_def"
    )

    registered_suite = context.suites.add(suite)
    validation_definition = gx.ValidationDefinition(
        data=batch_definition, suite=registered_suite, name=f"{asset_name}_validation"
    )
    validation_definition = context.validation_definitions.add(validation_definition)

    result = validation_definition.run(batch_parameters={"dataframe": df})
    return result.success, result


def _summarize_failures(result) -> list[str]:
    """Extract a short, readable list of which expectations failed."""
    failures = []
    for r in result.results:
        if not r.success:
            exp_type = r.expectation_config.type
            col = r.expectation_config.kwargs.get("column", "?")
            failures.append(f"{exp_type} on column '{col}'")
    return failures


def run_all_validations(
    locations: pd.DataFrame, policies: pd.DataFrame, claims: pd.DataFrame
) -> bool:
    """
    Run all three Silver-layer suites. Logs a clear pass/fail summary per
    table. Returns True only if all three pass.
    """
    all_passed = True

    for name, df, suite_builder in [
        ("locations", locations, build_locations_suite),
        ("policies", policies, build_policies_suite),
        ("claims", claims, build_claims_suite),
    ]:
        success, result = validate_dataframe(df, suite_builder, name)
        if success:
            logger.info("[%s] PASSED (%d rows)", name, len(df))
        else:
            all_passed = False
            failures = _summarize_failures(result)
            logger.error(
                "[%s] FAILED on %d rows. Failed expectations: %s",
                name,
                len(df),
                failures,
            )

    return all_passed


if __name__ == "__main__":
    from pathlib import Path

    logging.basicConfig(level=logging.INFO)

    silver_root = Path("data_pipeline/silver")
    locations_df = pd.read_parquet(silver_root / "locations.parquet")
    policies_df = pd.read_parquet(silver_root / "policies.parquet")
    claims_df = pd.read_parquet(silver_root / "claims.parquet")

    passed = run_all_validations(locations_df, policies_df, claims_df)
    if passed:
        print("\nAll Silver-layer validation suites PASSED.")
    else:
        print("\nOne or more Silver-layer validation suites FAILED — see log above.")
        raise SystemExit(1)