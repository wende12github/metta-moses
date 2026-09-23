import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Test runner for LLM-MOSES (Seeding & Semantic Mutation).
from llm_moses.tests.test_syntax_validator import (
    test_balanced_parentheses,
    test_ast_conversions,
    test_syntax_validation,
    test_validate_metta_sexpr,
)
from llm_moses.tests.test_evaluator import (
    test_ast_evaluation,
    test_count_nodes,
    test_truth_table_evaluation,
    test_tictactoe_state,
)
from llm_moses.tests.test_gemini_connection import test_gemini_connection
from llm_moses.tests.test_seeder import test_parity3_seeding, test_majority3_seeding
from llm_moses.tests.test_mutator import test_error_profiler, test_targeted_semantic_mutation
from llm_moses.tests.test_crossover_and_refactor import (
    test_semantic_crossover,
    test_anti_bloat_refactoring,
)


def run_all_tests():
    print("=" * 70)
    print("🚀 RUNNING COMPLETE LLM-MOSES TEST SUITE")
    print("=" * 70)

    suites = [
        ("Common: Syntax Validator", [
            ("test_balanced_parentheses", test_balanced_parentheses),
            ("test_ast_conversions", test_ast_conversions),
            ("test_syntax_validation", test_syntax_validation),
            ("test_validate_metta_sexpr", test_validate_metta_sexpr),
        ]),
        ("Common: Evaluator & Data Loader", [
            ("test_ast_evaluation", test_ast_evaluation),
            ("test_count_nodes", test_count_nodes),
            ("test_truth_table_evaluation", test_truth_table_evaluation),
            ("test_tictactoe_state", test_tictactoe_state),
        ]),
        ("Config & Gemini Connectivity", [
            ("test_gemini_connection", test_gemini_connection),
        ]),
        ("Seeder Integration (Parity3 & Majority3)", [
            ("test_parity3_seeding", test_parity3_seeding),
            ("test_majority3_seeding", test_majority3_seeding),
        ]),
        ("Mutator & Error Profiler", [
            ("test_error_profiler", test_error_profiler),
            ("test_targeted_semantic_mutation", test_targeted_semantic_mutation),
        ]),
        ("Crossover & Anti-Bloat Refactoring", [
            ("test_semantic_crossover", test_semantic_crossover),
            ("test_anti_bloat_refactoring", test_anti_bloat_refactoring),
        ]),
    ]

    total_tests = 0
    passed_tests = 0
    start_time = time.time()

    for suite_name, tests in suites:
        print(f"\n Suite: {suite_name}")
        print("-" * 50)
        for test_name, test_fn in tests:
            total_tests += 1
            t0 = time.time()
            try:
                test_fn()
                elapsed = time.time() - t0
                print(f"  ✅ {test_name:<40} PASSED ({elapsed:.2f}s)")
                passed_tests += 1
            except Exception as e:
                elapsed = time.time() - t0
                print(f"  ❌ {test_name:<40} FAILED ({elapsed:.2f}s): {e}")

    total_elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f" SUMMARY: {passed_tests}/{total_tests} tests passed in {total_elapsed:.2f}s")
    print("=" * 70)

    if passed_tests == total_tests:
        print("✅ ALL TESTS PASSED SUCCESSFULLY!")
        return 0
    else:
        print(f"⚠️  {total_tests - passed_tests} test(s) failed.")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
