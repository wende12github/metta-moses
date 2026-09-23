import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from llm_moses.mutator.error_profiler import ErrorProfiler
from llm_moses.mutator.semantic_mutator import SemanticMutator
from llm_moses.mutator.mutator_evaluator import VariationEvaluator
from llm_moses.common.data_loader import DEMO_TRUTH_TABLES


# Unit & Integration tests for SemanticMutator.
def test_error_profiler():
    prob = DEMO_TRUTH_TABLES["parity3"]
    # Flawed candidate missing minterms:
    flawed_expr = "(OR (AND (NOT X1) (NOT X2) X3) (AND (NOT X1) X2 (NOT X3)))"
    profile = ErrorProfiler.profile_truth_table(flawed_expr, prob["labels"], prob["rows"])

    assert profile.total == 8
    assert profile.failed_count == 2
    assert profile.passed_count == 6
    assert profile.accuracy == 0.75
    assert not profile.is_perfect
    print("Error summary:\n", profile.format_failure_summary())


def test_targeted_semantic_mutation():
    prob = DEMO_TRUTH_TABLES["parity3"]
    flawed_expr = "(OR (AND (NOT X1) (NOT X2) X3) (AND (NOT X1) X2 (NOT X3)) (AND X1 (NOT X2) (NOT X3)))"
    profile = ErrorProfiler.profile_truth_table(flawed_expr, prob["labels"], prob["rows"])

    mutator = SemanticMutator()
    mutants = mutator.mutate(
        parent_sexpr=flawed_expr,
        error_profile=profile,
        allowed_leaves=prob["labels"][:-1],
        domain="boolean",
        n_mutants=2,
    )
    print("Generated Mutants:", mutants)
    assert len(mutants) >= 1

    # Check evaluation
    eval_res = VariationEvaluator.evaluate_mutation(
        mutant_sexpr=mutants[0],
        parent_sexpr=flawed_expr,
        labels=prob["labels"],
        rows=prob["rows"],
    )
    print("Evaluation result:", eval_res)
    assert "parent_accuracy" in eval_res


if __name__ == "__main__":
    test_error_profiler()
    test_targeted_semantic_mutation()
    print("All mutator tests passed!")
