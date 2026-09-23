import logging
import os
import sys
from pathlib import Path
from typing import Optional

# Ensure project root is in sys.path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from llm_moses.config.gemini_client import GeminiClient
from llm_moses.config.settings import DEFAULT_MODEL
from llm_moses.common.data_loader import DEMO_TRUTH_TABLES
from llm_moses.seeder.seeder_main import LLMSeeder

logger = logging.getLogger(__name__)


"""
LLM Initial Population Seeder Bridge for MeTTa-MOSES.

Exposes functions called from MeTTa via `py-call` to synthesize candidate programs
using Google Gemini (default: gemini-3.5-flash).
"""
def generate_seed_expressions(
    problem_name: str = "parity3",
    input_file: str = "",
    target_feature: str = "",
    n_seeds: int = 5,
    model_name: str = DEFAULT_MODEL,
) -> str:
    """Synthesizes initial population candidates using Gemini and returns a MeTTa expression list string.

    Args:
        problem_name: Problem identifier (e.g. "parity3", "majority3", "mux3", "tic-tac-toe").
        input_file: Path to a CSV file (if using tabular data).
        target_feature: Target feature column name for CSV (optional).
        n_seeds: Number of diverse candidates to synthesize.
        model_name: Gemini model (defaults to gemini-3.5-flash).

    Returns:
        A string containing a valid MeTTa expression list of parsed candidate expressions,
        e.g. "((OR (AND X1 X2) (AND (NOT X1) (NOT X2))) (AND X1 (NOT X2)))".
        On failure or error, returns a fallback expression list "((true))".
    """
    # [DEBUG PRINT] print(f"[DEBUG MeTTa Bridge: generate_seed_expressions] Invoked with problem='{problem_name}', file='{input_file}', n_seeds={n_seeds}, model='{model_name}'")
    try:
        seeder = LLMSeeder(model_name=str(model_name))
        candidates = []

        if input_file and str(input_file).strip() != "" and str(input_file).strip() != "()":
            csv_path = Path(str(input_file).strip())
            if csv_path.is_file():
                candidates = seeder.seed_csv(
                    csv_path=csv_path,
                    target_feature=str(target_feature).strip() if target_feature and target_feature != "()" else "",
                    n_seeds=int(n_seeds),
                )
        elif problem_name and str(problem_name).strip() != "" and str(problem_name).strip() != "()":
            prob_str = str(problem_name).strip().lower()
            candidates = seeder.seed_by_problem_name(problem_name=prob_str, n_seeds=int(n_seeds))

        if not candidates:
            # [DEBUG PRINT] print("[DEBUG MeTTa Bridge: generate_seed_expressions] No candidates returned. Yielding fallback '(true)'")
            return "(true)"

        # Return expression list: (expr1 expr2 expr3 ...)
        formatted_cands = [c if isinstance(c, str) else getattr(c, "sexpr", str(c)) for c in candidates]
        expr_list_str = "(" + " ".join(formatted_cands) + ")"
        # [DEBUG PRINT] print(f"[DEBUG MeTTa Bridge: generate_seed_expressions] Returning MeTTa exemplar list: {expr_list_str}")
        return expr_list_str

    except Exception as e:
        logger.error(f"Error in LLM seeding bridge: {e}")
        # Graceful fallback to default constant true
        # [DEBUG PRINT] print(f"[DEBUG MeTTa Bridge: generate_seed_expressions] Exception: {e}. Falling back to '(true)'")
        return "(true)"


if __name__ == "__main__":
    print("Testing generate_seed_expressions on parity3:")
    res = generate_seed_expressions(problem_name="parity3", n_seeds=3)
    print("Result:", res)
