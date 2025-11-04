"""Rulebook testing framework."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from rulesmith.dag.graph import Rulebook
from rulesmith.io.decision_result import DecisionResult


@dataclass
class TestCase:
    """A single test case for a rulebook."""
    
    name: str
    inputs: Dict[str, Any]
    expected: Dict[str, Any]
    description: Optional[str] = None
    expected_metrics: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "inputs": self.inputs,
            "expected": self.expected,
            "description": self.description,
            "expected_metrics": self.expected_metrics,
        }


@dataclass
class TestResult:
    """Result of running a test case."""
    
    test_case: TestCase
    passed: bool
    actual: Optional[Dict[str, Any]] = None
    decision_result: Optional[DecisionResult] = None
    error: Optional[str] = None
    differences: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "test_case": self.test_case.to_dict(),
            "passed": self.passed,
            "actual": self.actual,
            "error": self.error,
            "differences": self.differences,
        }


@dataclass
class TestSuite:
    """Collection of test cases and results."""
    
    rulebook: Rulebook
    test_cases: List[TestCase] = field(default_factory=list)
    results: List[TestResult] = field(default_factory=list)
    
    def add_test_case(
        self,
        name: str,
        inputs: Dict[str, Any],
        expected: Dict[str, Any],
        description: Optional[str] = None,
        expected_metrics: Optional[Dict[str, float]] = None,
    ) -> "TestSuite":
        """
        Add a test case to the suite.
        
        Args:
            name: Test case name
            inputs: Input payload for the rulebook
            expected: Expected output dictionary
            description: Optional description
            expected_metrics: Optional expected metric values
        
        Returns:
            Self for chaining
        """
        test_case = TestCase(
            name=name,
            inputs=inputs,
            expected=expected,
            description=description,
            expected_metrics=expected_metrics,
        )
        self.test_cases.append(test_case)
        return self
    
    def run_all(self) -> "TestSuite":
        """
        Run all test cases.
        
        Returns:
            Self with results populated
        """
        self.results = []
        
        for test_case in self.test_cases:
            try:
                # Run rulebook
                result = self.rulebook.run(test_case.inputs, enable_mlflow=False)
                
                # Extract actual output
                if isinstance(result, DecisionResult):
                    actual = result.value
                    decision_result = result
                else:
                    actual = result
                    decision_result = None
                
                # Check if passed
                passed, differences = self._compare_outputs(test_case.expected, actual)
                
                # Check metrics if expected
                if test_case.expected_metrics and decision_result:
                    metric_passed, metric_diffs = self._compare_metrics(
                        test_case.expected_metrics, decision_result.metrics
                    )
                    passed = passed and metric_passed
                    differences["metrics"] = metric_diffs
                
                test_result = TestResult(
                    test_case=test_case,
                    passed=passed,
                    actual=actual,
                    decision_result=decision_result,
                    differences=differences,
                )
                
            except Exception as e:
                test_result = TestResult(
                    test_case=test_case,
                    passed=False,
                    error=str(e),
                )
            
            self.results.append(test_result)
        
        return self
    
    def _compare_outputs(
        self, expected: Dict[str, Any], actual: Dict[str, Any]
    ) -> tuple[bool, Dict[str, Any]]:
        """Compare expected and actual outputs."""
        differences = {}
        passed = True
        
        # Check all expected keys
        for key, expected_value in expected.items():
            if key not in actual:
                differences[key] = {"expected": expected_value, "actual": None}
                passed = False
            elif actual[key] != expected_value:
                differences[key] = {"expected": expected_value, "actual": actual[key]}
                passed = False
        
        # Check for unexpected keys (optional - might want to allow extra keys)
        # For now, we only check expected keys
        
        return passed, differences
    
    def _compare_metrics(
        self, expected: Dict[str, float], actual: Dict[str, float]
    ) -> tuple[bool, Dict[str, Any]]:
        """Compare expected and actual metrics."""
        differences = {}
        passed = True
        
        for metric_name, expected_value in expected.items():
            if metric_name not in actual:
                differences[metric_name] = {"expected": expected_value, "actual": None}
                passed = False
            elif abs(actual[metric_name] - expected_value) > 1e-6:
                differences[metric_name] = {
                    "expected": expected_value,
                    "actual": actual[metric_name],
                    "delta": actual[metric_name] - expected_value,
                }
                passed = False
        
        return passed, differences
    
    def summary(self) -> str:
        """Generate a summary of test results."""
        if not self.results:
            return "No tests run yet. Call run_all() first."
        
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed
        
        lines = [
            f"Test Suite Summary for {self.rulebook.name}",
            f"=" * 50,
            f"Total: {total}",
            f"Passed: {passed}",
            f"Failed: {failed}",
            "",
        ]
        
        if failed > 0:
            lines.append("Failed Tests:")
            for result in self.results:
                if not result.passed:
                    lines.append(f"  - {result.test_case.name}")
                    if result.error:
                        lines.append(f"    Error: {result.error}")
                    if result.differences:
                        lines.append(f"    Differences: {result.differences}")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rulebook_name": self.rulebook.name,
            "rulebook_version": self.rulebook.version,
            "test_cases": [tc.to_dict() for tc in self.test_cases],
            "results": [r.to_dict() for r in self.results],
            "summary": {
                "total": len(self.results),
                "passed": sum(1 for r in self.results if r.passed),
                "failed": sum(1 for r in self.results if not r.passed),
            },
        }


class RulebookTester:
    """
    Testing framework for rulebooks.
    
    Provides a simple way to write and run test suites for rulebooks.
    """
    
    def __init__(self, rulebook: Rulebook):
        """
        Initialize tester.
        
        Args:
            rulebook: Rulebook to test
        """
        self.rulebook = rulebook
        self.suite = TestSuite(rulebook=rulebook)
    
    def add_test_case(
        self,
        name: str,
        inputs: Dict[str, Any],
        expected: Dict[str, Any],
        description: Optional[str] = None,
        expected_metrics: Optional[Dict[str, float]] = None,
    ) -> "RulebookTester":
        """
        Add a test case.
        
        Args:
            name: Test case name
            inputs: Input payload
            expected: Expected output
            description: Optional description
            expected_metrics: Optional expected metrics
        
        Returns:
            Self for chaining
        """
        self.suite.add_test_case(name, inputs, expected, description, expected_metrics)
        return self
    
    def run_all(self) -> TestSuite:
        """
        Run all test cases.
        
        Returns:
            TestSuite with results
        """
        return self.suite.run_all()
    
    def summary(self) -> str:
        """Get summary of test results."""
        return self.suite.summary()

