import logging
import os
import sys
from pathlib import Path
from typing import Optional

# Ensure project root is in sys.path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from llm_moses.config.settings import DEFAULT_MODEL, DEFAULT_N_MUTANTS
from llm_moses.common.syntax_validator import validate_metta_sexpr
from llm_moses.common.data_loader import load_problem_data
from llm_moses.mutator.error_profiler import ErrorProfiler
from llm_moses.mutator.semantic_mutator import SemanticMutator
from llm_moses.mutator.semantic_crossover import SemanticCrossover

logger = logging.getLogger(__name__)


"""
LLM Semantic Mutator & Variation Bridge for MeTTa-MOSES.

Exposes functions called from MeTTa via `py-call` to perform targeted logic repairs,
semantic crossover, and anti-bloat refactoring using Google Gemini.
"""
def _to_sexpr_string(val) -> str:
    """Converts a string, list, or tuple into a canonical MeTTa S-expression string."""
    if isinstance(val, (list, tuple)):
        return "(" + " ".join(_to_sexpr_string(x) for x in val) + ")"
    return str(val).strip()


def mutate_expression(
    parent_expr_str: str,
    problem_name: str = "parity3",
    input_file: str = "",
    target_feature: str = "",
    n_mutants: int = DEFAULT_N_MUTANTS,
    model_name: str = DEFAULT_MODEL,
) -> str:
    """Performs targeted semantic mutation on a parent expression using Gemini."""
    norm_parent = _to_sexpr_string(parent_expr_str)
    # [DEBUG PRINT] print(f"[DEBUG MeTTa Bridge: mutate_expression] Invoked with parent='{norm_parent}', problem='{problem_name}'")
    try:
        labels, rows = load_problem_data(
            problem_name=problem_name,
            input_file=input_file,
            target_feature=target_feature,
        )
        allowed = set(labels[:-1])

        clean_parent = validate_metta_sexpr(norm_parent, allowed)
        if not clean_parent:
            logger.warning(f"Parent expression invalid: {norm_parent}")
            # [DEBUG PRINT] print(f"[DEBUG MeTTa Bridge: mutate_expression] Parent invalid. Returning '({norm_parent})'")
            return f"({norm_parent})"

        profile = ErrorProfiler.profile_truth_table(clean_parent, labels, rows)
        if profile.is_perfect:
            # Already perfect, no mutation needed
            # [DEBUG PRINT] print(f"[DEBUG MeTTa Bridge: mutate_expression] Parent is already perfect (100%). Returning '({clean_parent})'")
            return f"({clean_parent})"

        mutator = SemanticMutator(model_name=str(model_name))
        mutants = mutator.mutate(
            parent_sexpr=clean_parent,
            error_profile=profile,
            allowed_leaves=labels[:-1],
            domain="boolean",
            n_mutants=int(n_mutants),
        )

        if not mutants:
            # [DEBUG PRINT] print(f"[DEBUG MeTTa Bridge: mutate_expression] No mutants generated. Returning '({clean_parent})'")
            return f"({clean_parent})"

        mutants_list_str = "(" + " ".join(mutants) + ")"
        # [DEBUG PRINT] print(f"[DEBUG MeTTa Bridge: mutate_expression] Returning mutants list: {mutants_list_str}")
        return mutants_list_str

    except Exception as e:
        logger.error(f"Error in mutate_expression: {e}")
        # [DEBUG PRINT] print(f"[DEBUG MeTTa Bridge: mutate_expression] Exception: {e}. Returning '({norm_parent})'")
        return f"({norm_parent})"


def crossover_expressions(
    parent_a_str: str,
    parent_b_str: str,
    problem_name: str = "parity3",
    input_file: str = "",
    target_feature: str = "",
    n_children: int = DEFAULT_N_MUTANTS,
    model_name: str = DEFAULT_MODEL,
) -> str:
    """Performs semantic crossover between two parent expressions."""
    norm_a = _to_sexpr_string(parent_a_str)
    norm_b = _to_sexpr_string(parent_b_str)
    # [DEBUG PRINT] print(f"[DEBUG MeTTa Bridge: crossover_expressions] Crossing '{norm_a}' and '{norm_b}'")
    try:
        labels, rows = load_problem_data(
            problem_name=problem_name,
            input_file=input_file,
            target_feature=target_feature,
        )
        allowed = set(labels[:-1])

        clean_a = validate_metta_sexpr(norm_a, allowed) or norm_a
        clean_b = validate_metta_sexpr(norm_b, allowed) or norm_b

        profile_a = ErrorProfiler.profile_truth_table(clean_a, labels, rows)
        profile_b = ErrorProfiler.profile_truth_table(clean_b, labels, rows)

        crossover_op = SemanticCrossover(model_name=str(model_name))
        children = crossover_op.crossover(
            parent_a_sexpr=clean_a,
            parent_b_sexpr=clean_b,
            profile_a=profile_a,
            profile_b=profile_b,
            allowed_leaves=labels[:-1],
            domain="boolean",
            n_children=int(n_children),
        )

        if not children:
            # [DEBUG PRINT] print(f"[DEBUG MeTTa Bridge: crossover_expressions] Fallback: Returning '({clean_a} {clean_b})'")
            return f"({clean_a} {clean_b})"

        children_list_str = "(" + " ".join(children) + ")"
        # [DEBUG PRINT] print(f"[DEBUG MeTTa Bridge: crossover_expressions] Returning crossover children: {children_list_str}")
        return children_list_str

    except Exception as e:
        logger.error(f"Error in crossover_expressions: {e}")
        # [DEBUG PRINT] print(f"[DEBUG MeTTa Bridge: crossover_expressions] Exception: {e}. Returning '({norm_a} {norm_b})'")
        return f"({norm_a} {norm_b})"


def refactor_expression(
    parent_expr_str: str,
    problem_name: str = "parity3",
    input_file: str = "",
    target_feature: str = "",
    model_name: str = DEFAULT_MODEL,
) -> str:
    """Refactors a bloated parent expression."""
    norm_parent = _to_sexpr_string(parent_expr_str)
    # [DEBUG PRINT] print(f"[DEBUG MeTTa Bridge: refactor_expression] Refactoring '{norm_parent}'")
    try:
        labels, rows = load_problem_data(
            problem_name=problem_name,
            input_file=input_file,
            target_feature=target_feature,
        )
        allowed = set(labels[:-1])
        clean_parent = validate_metta_sexpr(norm_parent, allowed) or norm_parent

        mutator = SemanticMutator(model_name=str(model_name))
        refactored = mutator.refactor(
            parent_sexpr=clean_parent,
            allowed_leaves=labels[:-1],
            domain="boolean",
        )
        if not refactored:
            # [DEBUG PRINT] print(f"[DEBUG MeTTa Bridge: refactor_expression] Fallback: Returning '({clean_parent})'")
            return f"({clean_parent})"
        refactored_str = "(" + " ".join(refactored) + ")"
        # [DEBUG PRINT] print(f"[DEBUG MeTTa Bridge: refactor_expression] Returning refactored list: {refactored_str}")
        return refactored_str
    except Exception as e:
        logger.error(f"Error in refactor_expression: {e}")
        # [DEBUG PRINT] print(f"[DEBUG MeTTa Bridge: refactor_expression] Exception: {e}. Returning '({norm_parent})'")
        return f"({norm_parent})"


if __name__ == "__main__":
    test_parent = "(OR (AND (NOT X1) (NOT X2) X3) (AND (NOT X1) X2 (NOT X3)) (AND X1 (NOT X2) (NOT X3)))"
    print("Testing mutate_expression on parity3:")
    res = mutate_expression(test_parent, problem_name="parity3", n_mutants=2)
    print("Mutated candidates:", res)
