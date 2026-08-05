"""
Automated validation gate for model registration.

Blocks model promotion to registry if metrics fall outside defined thresholds.
Thresholds are based on achievable performance ranges from Milestone 4.1,
making this a regression-catcher rather than an impossible target.
"""
import logging
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)


@dataclass
class ValidationThresholds:
    """Performance thresholds for model validation."""
    
    # Frequency model thresholds
    frequency_auc_min: float = 0.50  # Floor - near-random is expected ceiling
    frequency_auc_max: float = 0.95  # Catch leakage (0.92 was event-feature leakage)
    
    # Severity model thresholds
    severity_r2_min: float = 0.0  # Floor - negative R² is worse than predicting mean
    severity_r2_max: float = 0.90  # Catch overfitting to deterministic generator
    
    # Generalization thresholds
    train_test_r2_gap_max: float = 0.30  # Catch severe overfitting
    train_test_auc_gap_max: float = 0.15  # Catch severe overfitting


@dataclass
class ValidationResult:
    """Result of model validation."""
    passed: bool
    model_type: Literal["frequency", "severity", "ensemble"]
    metrics: dict
    failures: list[str]
    
    def __str__(self) -> str:
        if self.passed:
            return f"✓ {self.model_type} validation passed"
        else:
            failures_str = "; ".join(self.failures)
            return f"✗ {self.model_type} validation failed: {failures_str}"


def validate_frequency_model(
    train_auc: float,
    test_auc: float,
    thresholds: ValidationThresholds | None = None,
) -> ValidationResult:
    """
    Validate frequency model metrics against thresholds.
    
    Args:
        train_auc: Training AUC
        test_auc: Test AUC
        thresholds: Validation thresholds (uses defaults if None)
    
    Returns:
        ValidationResult with pass/fail status and failure reasons
    """
    if thresholds is None:
        thresholds = ValidationThresholds()
    
    failures = []
    metrics = {"train_auc": train_auc, "test_auc": test_auc}
    
    # Check minimum performance (regression catcher)
    if test_auc < thresholds.frequency_auc_min:
        failures.append(
            f"test_auc {test_auc:.3f} below minimum {thresholds.frequency_auc_min}"
        )
    
    # Check for leakage (suspiciously high AUC)
    if test_auc > thresholds.frequency_auc_max:
        failures.append(
            f"test_auc {test_auc:.3f} exceeds maximum {thresholds.frequency_auc_max} "
            "(possible data leakage)"
        )
    
    # Check train-test gap (overfitting)
    auc_gap = train_auc - test_auc
    if auc_gap > thresholds.train_test_auc_gap_max:
        failures.append(
            f"train-test AUC gap {auc_gap:.3f} exceeds maximum "
            f"{thresholds.train_test_auc_gap_max} (severe overfitting)"
        )
    
    passed = len(failures) == 0
    return ValidationResult(
        passed=passed,
        model_type="frequency",
        metrics=metrics,
        failures=failures,
    )


def validate_severity_model(
    train_r2: float,
    test_r2: float,
    thresholds: ValidationThresholds | None = None,
) -> ValidationResult:
    """
    Validate severity model metrics against thresholds.
    
    Args:
        train_r2: Training R²
        test_r2: Test R²
        thresholds: Validation thresholds (uses defaults if None)
    
    Returns:
        ValidationResult with pass/fail status and failure reasons
    """
    if thresholds is None:
        thresholds = ValidationThresholds()
    
    failures = []
    metrics = {"train_r2": train_r2, "test_r2": test_r2}
    
    # Check minimum performance (regression catcher)
    if test_r2 < thresholds.severity_r2_min:
        failures.append(
            f"test_r2 {test_r2:.3f} below minimum {thresholds.severity_r2_min} "
            "(worse than predicting mean)"
        )
    
    # Check for leakage (suspiciously high R²)
    if test_r2 > thresholds.severity_r2_max:
        failures.append(
            f"test_r2 {test_r2:.3f} exceeds maximum {thresholds.severity_r2_max} "
            "(possible data leakage or overfit to deterministic generator)"
        )
    
    # Check train-test gap (overfitting)
    r2_gap = train_r2 - test_r2
    if r2_gap > thresholds.train_test_r2_gap_max:
        failures.append(
            f"train-test R² gap {r2_gap:.3f} exceeds maximum "
            f"{thresholds.train_test_r2_gap_max} (severe overfitting)"
        )
    
    passed = len(failures) == 0
    return ValidationResult(
        passed=passed,
        model_type="severity",
        metrics=metrics,
        failures=failures,
    )


def validate_ensemble(
    xgb_test_auc: float,
    catboost_test_auc: float,
    blended_test_auc: float,
    thresholds: ValidationThresholds | None = None,
) -> ValidationResult:
    """
    Validate ensemble model metrics against thresholds.
    
    Args:
        xgb_test_auc: XGBoost test AUC
        catboost_test_auc: CatBoost test AUC
        blended_test_auc: Blended ensemble test AUC
        thresholds: Validation thresholds (uses defaults if None)
    
    Returns:
        ValidationResult with pass/fail status and failure reasons
    """
    if thresholds is None:
        thresholds = ValidationThresholds()
    
    failures = []
    metrics = {
        "xgb_test_auc": xgb_test_auc,
        "catboost_test_auc": catboost_test_auc,
        "blended_test_auc": blended_test_auc,
    }
    
    # Check minimum performance (regression catcher)
    if blended_test_auc < thresholds.frequency_auc_min:
        failures.append(
            f"blended_auc {blended_test_auc:.3f} below minimum {thresholds.frequency_auc_min}"
        )
    
    # Check that ensemble improves or matches individual models
    if blended_test_auc < min(xgb_test_auc, catboost_test_auc) - 0.01:
        failures.append(
            f"blended_auc {blended_test_auc:.3f} worse than both individual models "
            f"(XGB: {xgb_test_auc:.3f}, CatBoost: {catboost_test_auc:.3f})"
        )
    
    passed = len(failures) == 0
    return ValidationResult(
        passed=passed,
        model_type="ensemble",
        metrics=metrics,
        failures=failures,
    )


def run_validation_gate(
    frequency_result: ValidationResult,
    severity_result: ValidationResult,
    ensemble_result: ValidationResult | None = None,
    enforce: bool = True,
) -> bool:
    """
    Run the full validation gate for model registration.
    
    Args:
        frequency_result: Frequency model validation result
        severity_result: Severity model validation result
        ensemble_result: Optional ensemble validation result
        enforce: If True, raise ValueError on failure (blocks registration).
                 If False, log warnings and return False (for development).
    
    Returns:
        True if all validations pass, False otherwise
    
    Raises:
        ValueError: If any validation fails and enforce=True
    """
    results = [frequency_result, severity_result]
    if ensemble_result is not None:
        results.append(ensemble_result)
    
    logger.info("=" * 60)
    logger.info("Running validation gate for model registration")
    logger.info("=" * 60)
    
    for result in results:
        logger.info(str(result))
        if not result.passed:
            logger.error(f"  Metrics: {result.metrics}")
            logger.error(f"  Failures: {result.failures}")
    
    all_passed = all(r.passed for r in results)
    
    if all_passed:
        logger.info("=" * 60)
        logger.info("✓ All validations passed - model registration approved")
        logger.info("=" * 60)
    else:
        logger.error("=" * 60)
        logger.error("✗ Validation failed")
        if enforce:
            logger.error("Model registration blocked")
            logger.error("=" * 60)
            raise ValueError(
                "Validation gate failed: fix metric regressions before registration"
            )
        else:
            logger.warning("Validation gate not enforced (development mode)")
            logger.warning("Models would be blocked in production")
            logger.info("=" * 60)
    
    return all_passed
