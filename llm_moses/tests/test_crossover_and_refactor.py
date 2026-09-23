import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from llm_moses.mutator.error_profiler import ErrorProfiler
from llm_moses.mutator.semantic_crossover import SemanticCrossover
from llm_moses.mutator.semantic_mutator import SemanticMutator
from llm_moses.common.data_loader import DEMO_TRUTH_TABLES
from llm_moses.common.simple_evaluator import evaluate_expression_on_truth_table, count_ast_nodes
from llm_moses.common.syntax_validator import sexpr_to_ast


# Integration tests for SemanticCrossover and Refactoring.
def test_semantic_crossover():
    prob = DEMO_TRUTH_TABLES["parity3"]
    # Parent A solves minterms (0,0,1) and (0,1,0)
    parent_a = "(OR (AND (NOT X1) (NOT X2) X3) (AND (NOT X1) X2 (NOT X3)))"
    # Parent B solves minterms (1,0,0) and (1,1,1)
    parent_b = "(OR (AND X1 (NOT X2) (NOT X3)) (AND X1 X2 X3))"

    prof_a = ErrorProfiler.profile_truth_table(parent_a, prob["labels"], prob["rows"])
    prof_b = ErrorProfiler.profile_truth_table(parent_b, prob["labels"], prob["rows"])

    crossover_op = SemanticCrossover()
    children = crossover_op.crossover(
        parent_a_sexpr=parent_a,
        parent_b_sexpr=parent_b,
        profile_a=prof_a,
        profile_b=prof_b,
        allowed_leaves=prob["labels"][:-1],
        domain="boolean",
        n_children=2,
    )
    print("Crossover children:", children)
    assert len(children) >= 1

    scores = [
        evaluate_expression_on_truth_table(c, prob["labels"], prob["rows"])[0]
        for c in children
    ]
    max_acc = max(scores)
    print(f"Max crossover offspring accuracy: {max_acc * 100:.1f}%")
    assert max_acc >= 0.5


def test_anti_bloat_refactoring():
    prob = DEMO_TRUTH_TABLES["demorgan"]
    # Bloated expression: (OR (NOT A) (NOT B) (AND (NOT A) (NOT B))) -> simplifies to (OR (NOT A) (NOT B))
    bloated = "(OR (NOT A) (NOT B) (AND (NOT A) (NOT B)))"

    mutator = SemanticMutator()
    refactors = mutator.refactor(
        parent_sexpr=bloated,
        allowed_leaves=["A", "B"],
        domain="boolean",
    )
    print("Refactored variants:", refactors)
    assert len(refactors) >= 1

    # Check semantic equivalence
    orig_acc, _, _ = evaluate_expression_on_truth_table(bloated, prob["labels"], prob["rows"])
    ref_acc, _, _ = evaluate_expression_on_truth_table(refactors[0], prob["labels"], prob["rows"])
    assert orig_acc == ref_acc

    orig_nodes = count_ast_nodes(sexpr_to_ast(bloated))
    ref_nodes = count_ast_nodes(sexpr_to_ast(refactors[0]))
    print(f"Nodes reduced from {orig_nodes} to {ref_nodes}")


if __name__ == "__main__":
    test_semantic_crossover()
    test_anti_bloat_refactoring()
    print("All crossover and refactoring tests passed!")
