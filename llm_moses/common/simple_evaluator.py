import csv
import logging
from pathlib import Path
from typing import Union

from llm_moses.common.syntax_validator import sexpr_to_ast

logger = logging.getLogger(__name__)


def count_ast_nodes(ast: Union[str, list]) -> int:
    # Counts total AST nodes (operators and variable leaves) as a proxy for tree complexity.
    if isinstance(ast, list):
        return sum(count_ast_nodes(child) for child in ast)
    return 1


def evaluate_ast(ast: Union[str, list], var_assignment: dict[str, bool]) -> bool:
    # Evaluates an AST against a single boolean variable assignment.
    if isinstance(ast, str):
        if ast.lower() == "true":
            return True
        elif ast.lower() == "false":
            return False
        # Case-insensitive variable lookup
        for k, v in var_assignment.items():
            if k.lower() == ast.lower():
                return bool(v)
        raise KeyError(f"Variable '{ast}' not found in assignment: {list(var_assignment.keys())}")

    if isinstance(ast, list):
        if not ast:
            raise ValueError("Cannot evaluate empty AST node")

        # Constant wrapped in list, e.g. ["true"]
        if len(ast) == 1 and isinstance(ast[0], str):
            return evaluate_ast(ast[0], var_assignment)

        op = str(ast[0]).upper()
        args = ast[1:]

        if op == "NOT":
            if len(args) != 1:
                raise ValueError(f"NOT expects exactly 1 argument, got {len(args)}")
            return not evaluate_ast(args[0], var_assignment)

        elif op == "AND":
            if not args:
                raise ValueError("AND expects at least 1 argument")
            return all(evaluate_ast(arg, var_assignment) for arg in args)

        elif op in {"OR", "PRIORITIZED-OR"}:
            if not args:
                raise ValueError(f"{op} expects at least 1 argument")
            return any(evaluate_ast(arg, var_assignment) for arg in args)

        else:
            raise ValueError(f"Unknown operator '{op}'")

    raise TypeError(f"Invalid AST node type: {type(ast)}")


def evaluate_expression_on_truth_table(
    expr: str,
    labels: list[str],
    rows: list[list[bool]],
) -> tuple[float, int, int]:
    """Evaluates a candidate expression against a truth table.

    Args:
        expr: MeTTa expression string, e.g. "(OR (AND X1 X2) (NOT X3))"
        labels: Column names, where the last label is the target output.
        rows: List of boolean rows.

    Returns:
        tuple of (accuracy_float, correct_count, total_count)
    """
    try:
        ast = sexpr_to_ast(expr)
    except Exception as e:
        logger.debug(f"Failed to parse expression '{expr}': {e}")
        return 0.0, 0, len(rows)

    feature_names = labels[:-1]
    correct = 0
    total = len(rows)

    for row in rows:
        var_assignment = {feature_names[i]: row[i] for i in range(len(feature_names))}
        expected = row[-1]
        try:
            actual = evaluate_ast(ast, var_assignment)
            if actual == expected:
                correct += 1
        except Exception:
            # On evaluation error, count as incorrect
            pass

    accuracy = correct / total if total > 0 else 0.0
    # [DEBUG PRINT] print(f"[DEBUG Evaluator] Evaluated '{expr}' -> Accuracy: {accuracy*100:.1f}% ({correct}/{total})")
    return accuracy, correct, total


def evaluate_expression_on_csv(
    expr: str,
    csv_path: Union[str, Path],
    target_feature: str = "",
) -> tuple[float, int, int]:
    """Evaluates a candidate expression against a CSV dataset.

    Args:
        expr: Candidate MeTTa expression string.
        csv_path: Path to CSV dataset.
        target_feature: Name of target column (defaults to last column).

    Returns:
        tuple of (accuracy_float, correct_count, total_count)
    """
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        all_rows = list(reader)

    if not all_rows:
        return 0.0, 0, 0

    header = [h.strip() for h in all_rows[0]]
    if not target_feature:
        target_col = header[-1]
        feature_cols = header[:-1]
    else:
        target_col = target_feature.strip()
        feature_cols = [h for h in header if h != target_col]

    target_idx = header.index(target_col)
    feature_indices = [header.index(fc) for fc in feature_cols]

    def parse_bool(val: str) -> bool:
        v = val.strip().lower()
        return v in {"1", "true", "t", "yes", "y"}

    try:
        ast = sexpr_to_ast(expr)
    except Exception:
        return 0.0, 0, len(all_rows) - 1

    correct = 0
    total = 0

    for row in all_rows[1:]:
        if not row or len(row) < len(header):
            continue
        total += 1
        var_assignment = {
            header[idx]: parse_bool(row[idx]) for idx in feature_indices
        }
        expected = parse_bool(row[target_idx])
        try:
            actual = evaluate_ast(ast, var_assignment)
            if actual == expected:
                correct += 1
        except Exception:
            pass

    accuracy = correct / total if total > 0 else 0.0
    # [DEBUG PRINT] print(f"[DEBUG Evaluator] CSV Evaluated '{expr}' -> Accuracy: {accuracy*100:.1f}% ({correct}/{total})")
    return accuracy, correct, total


def evaluate_tictactoe_state(expr: str, board: list[str]) -> bool:
    """Evaluates a candidate MeTTa heuristic expression on a Tic-Tac-Toe board state.

    Args:
        expr: MeTTa expression with variables Pos1..Pos9.
        board: 9-element list where each element is 'X', 'O', or '.'.
    """
    var_assignment = {f"Pos{i+1}": (board[i] == "X") for i in range(9)}
    try:
        ast = sexpr_to_ast(expr)
        return evaluate_ast(ast, var_assignment)
    except Exception:
        return False
