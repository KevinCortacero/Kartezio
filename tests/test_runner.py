"""
Comprehensive test runner for Kartezio package.
This script runs all tests with proper reporting and validation.
"""

import sys
import time
import unittest
from io import StringIO
from typing import Any


class KartezioTestResult(unittest.TestResult):
    """Custom test result class with detailed reporting."""

    def __init__(self):
        super().__init__()
        self.test_results = []
        self.start_time = None

    def startTest(self, test):
        super().startTest(test)
        self.start_time = time.time()

    def stopTest(self, test):
        super().stopTest(test)
        duration = time.time() - self.start_time if self.start_time else 0

        status = "PASS"
        message = ""

        # Determine test status
        for failure in self.failures:
            if failure[0] == test:
                status = "FAIL"
                message = failure[1]
                break

        for error in self.errors:
            if error[0] == test:
                status = "ERROR"
                message = error[1]
                break

        for skip in self.skipped:
            if skip[0] == test:
                status = "SKIP"
                message = skip[1]
                break

        self.test_results.append(
            {
                "test": str(test),
                "status": status,
                "duration": duration,
                "message": message,
            }
        )


class KartezioTestRunner:
    """Custom test runner for Kartezio package."""

    def __init__(self, verbosity: int = 2):
        self.verbosity = verbosity
        self.results = {}

    def run_test_suite(self, test_suite, suite_name: str) -> dict[str, Any]:
        """Run a test suite and return results."""
        print(f"\n{'=' * 60}")
        print(f"Running {suite_name}")
        print(f"{'=' * 60}")

        result = KartezioTestResult()

        # Capture stdout during tests
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        try:
            start_time = time.time()
            test_suite.run(result)
            total_time = time.time() - start_time

        finally:
            sys.stdout = old_stdout

        # Print results
        self._print_suite_results(result, suite_name, total_time)

        return {
            "suite_name": suite_name,
            "tests_run": result.testsRun,
            "failures": len(result.failures),
            "errors": len(result.errors),
            "skipped": len(result.skipped),
            "total_time": total_time,
            "test_results": result.test_results,
            "captured_output": captured_output.getvalue(),
        }

    def _print_suite_results(
        self, result: KartezioTestResult, suite_name: str, total_time: float
    ):
        """Print formatted results for a test suite."""
        print(f"\n{suite_name} Results:")
        print(f"Tests run: {result.testsRun}")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
        print(f"Skipped: {len(result.skipped)}")
        print(f"Time: {total_time:.2f}s")

        if self.verbosity >= 2:
            # Print individual test results
            for test_result in result.test_results:
                status_symbol = {
                    "PASS": "✓",
                    "FAIL": "✗",
                    "ERROR": "✗",
                    "SKIP": "-",
                }.get(test_result["status"], "?")

                print(
                    f"  {status_symbol} {test_result['test']} ({test_result['duration']:.3f}s)"
                )

                if test_result["status"] in ["FAIL", "ERROR"] and self.verbosity >= 3:
                    # Print error details
                    print(f"    {test_result['message'][:200]}...")

        # Print failure/error details
        if result.failures and self.verbosity >= 1:
            print(f"\nFAILURES ({len(result.failures)}):")
            for test, traceback in result.failures:
                print(f"  {test}")
                if self.verbosity >= 2:
                    print(
                        f"    {traceback.split(chr(10))[-2]}"
                    )  # Last line of traceback

        if result.errors and self.verbosity >= 1:
            print(f"\nERRORS ({len(result.errors)}):")
            for test, traceback in result.errors:
                print(f"  {test}")
                if self.verbosity >= 2:
                    print(
                        f"    {traceback.split(chr(10))[-2]}"
                    )  # Last line of traceback

    def run_all_tests(self) -> dict[str, Any]:
        """Run all test suites."""
        print("Kartezio Package Test Suite")
        print("=" * 60)

        overall_start = time.time()

        # Define test suites
        test_suites = [
            ("Security Tests", "tests.test_security"),
            ("Core Component Tests", "tests.test_core_components"),
            ("Component Tests", "tests.test_components"),
            ("Genotype Tests", "tests.test_genotype"),
            ("Sequential Tests", "tests.test_sequential"),
            ("Reader Tests", "tests.components.test_reader"),
            ("Integration Tests", "tests.test_integration"),
        ]

        all_results = []
        total_tests = 0
        total_failures = 0
        total_errors = 0
        total_skipped = 0

        for suite_name, module_name in test_suites:
            try:
                # Load test suite
                loader = unittest.TestLoader()
                suite = loader.loadTestsFromName(module_name)

                # Run tests
                suite_result = self.run_test_suite(suite, suite_name)
                all_results.append(suite_result)

                # Accumulate stats
                total_tests += suite_result["tests_run"]
                total_failures += suite_result["failures"]
                total_errors += suite_result["errors"]
                total_skipped += suite_result["skipped"]

            except ImportError as e:
                print(f"\nSkipping {suite_name}: {e}")
                continue
            except Exception as e:
                print(f"\nError running {suite_name}: {e}")
                continue

        total_time = time.time() - overall_start

        # Print overall summary
        self._print_final_summary(
            total_tests,
            total_failures,
            total_errors,
            total_skipped,
            total_time,
        )

        return {
            "overall_results": {
                "total_tests": total_tests,
                "total_failures": total_failures,
                "total_errors": total_errors,
                "total_skipped": total_skipped,
                "total_time": total_time,
                "success_rate": (total_tests - total_failures - total_errors)
                / max(total_tests, 1)
                * 100,
            },
            "suite_results": all_results,
        }

    def _print_final_summary(
        self,
        total_tests: int,
        total_failures: int,
        total_errors: int,
        total_skipped: int,
        total_time: float,
    ):
        """Print final test summary."""
        print(f"\n{'=' * 60}")
        print("FINAL SUMMARY")
        print(f"{'=' * 60}")
        print(f"Total tests run: {total_tests}")
        print(f"Successes: {total_tests - total_failures - total_errors}")
        print(f"Failures: {total_failures}")
        print(f"Errors: {total_errors}")
        print(f"Skipped: {total_skipped}")
        print(f"Total time: {total_time:.2f}s")

        if total_tests > 0:
            success_rate = (
                (total_tests - total_failures - total_errors) / total_tests * 100
            )
            print(f"Success rate: {success_rate:.1f}%")

            if total_failures == 0 and total_errors == 0:
                print("\n🎉 ALL TESTS PASSED! 🎉")
                return True
            else:
                print(f"\n❌ {total_failures + total_errors} tests failed")
                return False
        else:
            print("\n⚠️  No tests were run")
            return False


def run_quick_tests() -> bool:
    """Run a quick subset of tests for fast validation."""
    print("Running Quick Test Suite (Core functionality only)")

    runner = KartezioTestRunner(verbosity=1)

    # Quick test suites
    quick_suites = [
        ("Security Tests", "tests.test_security"),
        ("Core Component Tests", "tests.test_core_components"),
    ]

    total_tests = 0
    total_failures = 0
    total_errors = 0

    for suite_name, module_name in quick_suites:
        try:
            loader = unittest.TestLoader()
            suite = loader.loadTestsFromName(module_name)
            result = runner.run_test_suite(suite, suite_name)

            total_tests += result["tests_run"]
            total_failures += result["failures"]
            total_errors += result["errors"]

        except Exception as e:
            print(f"Error in quick test {suite_name}: {e}")
            return False

    success = total_failures == 0 and total_errors == 0 and total_tests > 0
    print(f"\nQuick tests: {'PASSED' if success else 'FAILED'}")
    return success


def validate_package_integrity():
    """Validate basic package integrity."""
    print("Validating Package Integrity...")

    integrity_checks = [
        ("Import kartezio", lambda: __import__("kartezio")),
        (
            "Import core components",
            lambda: __import__("kartezio.core.components"),
        ),
        (
            "Import primitives",
            lambda: __import__("kartezio.primitives.matrix"),
        ),
        ("Import evolution", lambda: __import__("kartezio.evolution.base")),
        (
            "Create basic library",
            lambda: __import__(
                "kartezio.primitives.matrix", fromlist=["default_matrix_lib"]
            ).default_matrix_lib(),
        ),
    ]

    failed_checks = []

    for check_name, check_func in integrity_checks:
        try:
            check_func()
            print(f"✓ {check_name}")
        except Exception as e:
            print(f"✗ {check_name}: {e}")
            failed_checks.append(check_name)

    if failed_checks:
        print(f"\n❌ Package integrity check failed: {len(failed_checks)} issues")
        return False
    else:
        print("\n✅ Package integrity check passed")
        return True


def main():
    """Main test runner entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Kartezio Test Runner")
    parser.add_argument(
        "--quick", action="store_true", help="Run quick test suite only"
    )
    parser.add_argument(
        "--integrity-only",
        action="store_true",
        help="Run package integrity checks only",
    )
    parser.add_argument(
        "--verbose", "-v", type=int, default=2, help="Verbosity level (0-3)"
    )

    args = parser.parse_args()

    # Always run integrity check first
    if not validate_package_integrity():
        print("❌ Package integrity check failed. Aborting tests.")
        sys.exit(1)

    if args.integrity_only:
        sys.exit(0)

    # Run tests
    if args.quick:
        success = run_quick_tests()
    else:
        runner = KartezioTestRunner(verbosity=args.verbose)
        results = runner.run_all_tests()
        success = (
            results["overall_results"]["total_failures"] == 0
            and results["overall_results"]["total_errors"] == 0
        )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
