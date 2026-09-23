import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from llm_moses.common.simple_evaluator import (
    evaluate_ast,
    count_ast_nodes,
    evaluate_expression_on_truth_table,
    evaluate_tictactoe_state,
)
from llm_moses.common.data_loader import DEMO_TRUTH_TABLES
from llm_moses.common.syntax_validator import sexpr_to_ast


# Unit tests for Simple Evaluator and Data Loader.
def test_ast_evaluation():
    ast = sexpr_to_ast("(OR (AND X1 (NOT X2)) X3)")
    assert evaluate_ast(ast, {"X1": True, "X2": False, "X3": False}) is True
    assert evaluate_ast(ast, {"X1": True, "X2": True, "X3": False}) is False
    assert evaluate_ast(ast, {"X1": False, "X2": False, "X3": True}) is True


def test_count_nodes():
    ast = sexpr_to_ast("(OR (AND X1 X2) (NOT X3))")
    # OR (1) + AND (1) + X1 (1) + X2 (1) + NOT (1) + X3 (1) = 6 nodes
    assert count_ast_nodes(ast) == 6


def test_truth_table_evaluation():
    prob = DEMO_TRUTH_TABLES["xor2"]
    # XOR formula: (OR (AND A (NOT B)) (AND (NOT A) B))
    xor_expr = "(OR (AND A (NOT B)) (AND (NOT A) B))"
    acc, correct, total = evaluate_expression_on_truth_table(
        expr=xor_expr,
        labels=prob["labels"],
        rows=prob["rows"],
    )
    assert acc == 1.0
    assert correct == 4
    assert total == 4


def test_tictactoe_state():
    board = ["X", "X", "X", ".", ".", ".", ".", ".", "."]
    win_expr = "(AND Pos1 Pos2 Pos3)"
    assert evaluate_tictactoe_state(win_expr, board) is True


if __name__ == "__main__":
    test_ast_evaluation()
    test_count_nodes()
    test_truth_table_evaluation()
    test_tictactoe_state()
    print("All evaluator tests passed!")
