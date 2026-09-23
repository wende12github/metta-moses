from typing import Any
from llm_moses.common.syntax_validator import sexpr_to_ast
from llm_moses.common.simple_evaluator import (
    evaluate_expression_on_truth_table,
    count_ast_nodes,
)
from llm_moses.mutator.error_profiler import ErrorProfiler, ErrorProfile


# Evaluator for comparing mutant candidates against parent programs.
class VariationEvaluator:
    # Evaluates variations and reports metric changes relative to parent programs.

    @staticmethod
    def evaluate_mutation(
        mutant_sexpr: str,
        parent_sexpr: str,
        labels: list[str],
        rows: list[list[bool]],
    ) -> dict[str, Any]:
        # Evaluates a mutant against its parent on a truth table or CSV dataset.
        p_acc, p_cor, p_tot = evaluate_expression_on_truth_table(parent_sexpr, labels, rows)
        m_acc, m_cor, m_tot = evaluate_expression_on_truth_table(mutant_sexpr, labels, rows)

        try:
            p_size = count_ast_nodes(sexpr_to_ast(parent_sexpr))
        except Exception:
            p_size = 999
        try:
            m_size = count_ast_nodes(sexpr_to_ast(mutant_sexpr))
        except Exception:
            m_size = 999

        p_profile = ErrorProfiler.profile_truth_table(parent_sexpr, labels, rows)
        m_profile = ErrorProfiler.profile_truth_table(mutant_sexpr, labels, rows)

        acc_delta = m_acc - p_acc
        size_delta = m_size - p_size

        return {
            "parent_accuracy": p_acc,
            "mutant_accuracy": m_acc,
            "accuracy_delta": acc_delta,
            "parent_size": p_size,
            "mutant_size": m_size,
            "size_delta": size_delta,
            "parent_profile": p_profile,
            "mutant_profile": m_profile,
            "improved": acc_delta > 0 or (acc_delta == 0 and size_delta < 0),
        }
