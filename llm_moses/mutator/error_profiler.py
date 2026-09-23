from dataclasses import dataclass
from typing import Any, Union

from llm_moses.common.syntax_validator import sexpr_to_ast
from llm_moses.common.simple_evaluator import evaluate_ast


# Profiles candidate AST execution against problem datasets to isolate failing cases (error signatures).
@dataclass
class ErrorProfile:
    """Encapsulates the performance breakdown and failure cases of a candidate program."""
    total: int
    passed_count: int
    failed_count: int
    accuracy: float
    error_rate: float
    failed_cases: list[dict[str, Any]]
    passed_cases: list[dict[str, Any]]

    @property
    def is_perfect(self) -> bool:
        """Returns True if the program has no failed cases."""
        return self.failed_count == 0

    def format_failure_summary(self) -> str:
        """Formats the top failure cases into a readable string for LLM prompts."""
        if not self.failed_cases:
            return "No failing cases (100% accuracy)."

        lines = []
        for i, c in enumerate(self.failed_cases, start=1):
            inputs_str = ", ".join(
                f"{k}={int(v) if isinstance(v, bool) else v}" for k, v in c["inputs"].items()
            )
            lines.append(
                f"- Case {i}: Inputs [{inputs_str}] -> Expected {int(c['expected'])}, but Program Output was {int(c['predicted']) if c['predicted'] is not None else 'ERROR'}"
            )
        return "\n".join(lines)


class ErrorProfiler:
    """Profiles candidate expressions to identify failing conditions and behavioral discrepancies."""

    @staticmethod
    def profile_truth_table(
        expr_or_ast: Union[str, list],
        labels: list[str],
        rows: list[list[bool]],
        max_samples: int = 6,
    ) -> ErrorProfile:
        """Profiles a boolean AST/S-expression against truth table rows."""
        # [DEBUG PRINT] print(f"[DEBUG ErrorProfiler] Profiling expression: '{expr_or_ast}' on {len(rows)} rows")

        inputs = labels[:-1]
        passed_cases = []
        failed_cases = []

        if isinstance(expr_or_ast, str):
            try:
                ast = sexpr_to_ast(expr_or_ast)
            except Exception:
                ast = None
        else:
            ast = expr_or_ast

        for r in rows:
            row_dict = {inputs[i]: r[i] for i in range(len(inputs))}
            expected = r[-1]
            if ast is not None:
                try:
                    predicted = evaluate_ast(ast, row_dict)
                except Exception:
                    predicted = None
            else:
                predicted = None

            case_info = {"inputs": row_dict, "expected": expected, "predicted": predicted}
            if predicted == expected:
                passed_cases.append(case_info)
            else:
                failed_cases.append(case_info)

        total = len(rows)
        passed_cnt = len(passed_cases)
        failed_cnt = len(failed_cases)
        accuracy = passed_cnt / total if total > 0 else 0.0
        error_rate = failed_cnt / total if total > 0 else 0.0

        # return ErrorProfile(
        #     total=total,
        #     passed_count=passed_cnt,
        #     failed_count=failed_cnt,
        #     accuracy=accuracy,
        #     error_rate=error_rate,
        #     failed_cases=failed_cases[:max_samples],
        #     passed_cases=passed_cases[:max_samples],
        # )
        profile = ErrorProfile(
                    total=total,
                    passed_count=passed_cnt,
                    failed_count=failed_cnt,
                    accuracy=accuracy,
                    error_rate=error_rate,
                    failed_cases=failed_cases[:max_samples],
                    passed_cases=passed_cases[:max_samples],
                )
        # [DEBUG PRINT] print(f"[DEBUG ErrorProfiler] Result: Accuracy={accuracy*100:.1f}% ({passed_cnt}/{total}), Failed Cases={failed_cnt}")
        # [DEBUG PRINT] if failed_cnt > 0: print(f"[DEBUG ErrorProfiler] Summary:\n{profile.format_failure_summary()}")
        return profile
