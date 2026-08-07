"""
Integration tests for the multi-agent LLM system
"""

import unittest
import json
from agents.orchestrator.orchestrator import Orchestrator, Intent
from agents.numeric_fidelity_validator import NumericFidelityValidator


class TestOrchestrator(unittest.TestCase):
    """Test orchestrator intent classification"""
    
    def setUp(self):
        self.orchestrator = Orchestrator(use_llm_classifier=False)
    
    def test_pricing_intent(self):
        """Test pricing intent classification"""
        queries = [
            "What is the price for a $10M Florida hurricane treaty?",
            "Calculate rate-on-line for this layer",
            "Draft a pricing memo for the Florida wind book"
        ]
        
        for query in queries:
            result = self.orchestrator.classify_intent(query)
            self.assertEqual(result.intent, Intent.PRICING)
            self.assertGreater(result.confidence, 0.5)
    
    def test_regulatory_intent(self):
        """Test regulatory intent classification"""
        queries = [
            "What does Article 105 say about SCR?",
            "Explain the Solvency II capital requirement",
            "Is this ORSA compliant?"
        ]
        
        for query in queries:
            result = self.orchestrator.classify_intent(query)
            self.assertEqual(result.intent, Intent.REGULATORY)
            self.assertGreater(result.confidence, 0.5)
    
    def test_scenario_intent(self):
        """Test scenario intent classification"""
        queries = [
            "What if hurricane frequency increases 20%?",
            "Run a stress test with 8% inflation",
            "Scenario: Category 5 landfall in Miami"
        ]
        
        for query in queries:
            result = self.orchestrator.classify_intent(query)
            self.assertEqual(result.intent, Intent.SCENARIO)
            self.assertGreater(result.confidence, 0.5)
    
    def test_report_intent(self):
        """Test report intent classification"""
        queries = [
            "Generate a full pricing report",
            "Create a Solvency II narrative",
            "Draft the ORSA report"
        ]
        
        for query in queries:
            result = self.orchestrator.classify_intent(query)
            self.assertEqual(result.intent, Intent.REPORT)
            self.assertGreater(result.confidence, 0.5)


class TestNumericFidelityValidator(unittest.TestCase):
    """Test numeric fidelity validator"""
    
    def setUp(self):
        self.validator = NumericFidelityValidator()
        
        # Register sample tool outputs
        self.validator.register_tool_output("predict", {
            "expected_loss": 5000000,
            "var_995": 15000000,
            "tvar_995": 22000000,
            "locations": [
                {"id": "loc1", "expected_loss": 100000},
                {"id": "loc2", "expected_loss": 200000}
            ]
        })
        
        self.validator.register_tool_output("simulate", {
            "scenario_expected_loss": 6000000,
            "scenario_var": 18000000
        })
    
    def test_valid_numbers_pass(self):
        """Test that valid numbers pass validation"""
        report = """
        The expected loss is $5,000,000.
        The VaR 99.5% is $15,000,000.
        The TVaR is $22,000,000.
        """
        
        is_valid, violations = self.validator.validate_report(report)
        self.assertTrue(is_valid)
        self.assertEqual(len(violations), 0)
    
    def test_invalid_numbers_fail(self):
        """Test that invalid numbers are flagged"""
        report = """
        The expected loss is $7,500,000.  # This number is not in tool outputs
        The VaR 99.5% is $15,000,000.   # This one is valid
        """
        
        is_valid, violations = self.validator.validate_report(report)
        self.assertFalse(is_valid)
        self.assertGreater(len(violations), 0)
    
    def test_percentage_numbers_pass(self):
        """Test that percentage representation passes"""
        report = """
        The expected loss is $5,000,000, which represents 50% of the total TIV.
        """
        
        # 50% of $10M TIV = $5M (matches the tool output)
        self.validator.register_tool_output("tiv", {"total": 10000000})
        
        is_valid, violations = self.validator.validate_report(report)
        self.assertTrue(is_valid)
    
    def test_hallucination_detection(self):
        """Test that hallucinations are detected"""
        report = """
        The model predicts losses of $5,000,000. 
        Our analysis shows an additional $2,500,000 in risk load,
        bringing the total to $7,500,000.
        """
        
        is_valid, violations = self.validator.validate_report(report)
        self.assertFalse(is_valid)
        
        # The $2,500,000 and $7,500,000 should be flagged
        self.assertGreaterEqual(len(violations), 1)


if __name__ == "__main__":
    unittest.main()