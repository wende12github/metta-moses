import logging
from pathlib import Path
from typing import Optional, Union

from llm_moses.config.gemini_client import GeminiClient
from llm_moses.config.settings import DEFAULT_MODEL, DEFAULT_N_SEEDS, DEFAULT_SEED_TEMPERATURE
from llm_moses.common.syntax_validator import validate_metta_sexpr
from llm_moses.common.data_loader import DEMO_TRUTH_TABLES, load_csv_data
from llm_moses.seeder.prompt_builder import (
    build_truth_table_prompt,
    build_csv_prompt,
    build_game_prompt,
)

logger = logging.getLogger(__name__)


# LLM Initial Population Seeder Engine for MeTTa-MOSES.
class LLMSeeder:
    """Orchestrates initial population seeding using Gemini LLM models."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        client: Optional[GeminiClient] = None,
        api_key: Optional[str] = None,
    ):
        self.model_name = model_name
        self.client = client or GeminiClient(model_name=self.model_name, api_key=api_key)

    def seed_truth_table(
        self,
        problem_name: str,
        labels: list[str],
        rows: list[list[bool]],
        n_seeds: int = DEFAULT_N_SEEDS,
        temperature: float = DEFAULT_SEED_TEMPERATURE,
    ) -> list[str]:
        """Synthesizes candidate programs for a boolean truth table problem."""
        prompt = build_truth_table_prompt(
            problem_name=problem_name,
            labels=labels,
            rows=rows,
            n_candidates=n_seeds,
        )
        # [DEBUG PRINT] print(f"[DEBUG LLMSeeder] Seeding truth table for '{problem_name}' ({len(rows)} rows, target seeds={n_seeds})")
        raw_candidates = self.client.generate_candidates(prompt=prompt, temperature=temperature)
        allowed_vars = set(labels[:-1])

        validated = []
        for cand in raw_candidates:
            clean = validate_metta_sexpr(cand, allowed_variables=allowed_vars)
            if clean and clean not in validated:
                validated.append(clean)

        # [DEBUG PRINT] print(f"[DEBUG LLMSeeder] Raw candidate strings received: {len(raw_candidates)}")
        return validated

    def seed_csv(
        self,
        csv_path: Union[str, Path],
        target_feature: str = "",
        n_seeds: int = DEFAULT_N_SEEDS,
        temperature: float = DEFAULT_SEED_TEMPERATURE,
    ) -> list[str]:
        """Synthesizes candidate programs for a CSV tabular dataset."""
        labels, bool_rows = load_csv_data(csv_path=csv_path, target_feature=target_feature)
        if not labels or not bool_rows:
            return []

        prompt = build_csv_prompt(
            labels=labels,
            rows=bool_rows,
            target_name=labels[-1],
            n_candidates=n_seeds,
        )
        raw_candidates = self.client.generate_candidates(prompt=prompt, temperature=temperature)
        allowed_vars = set(labels[:-1])

        validated = []
        for cand in raw_candidates:
            clean = validate_metta_sexpr(cand, allowed_variables=allowed_vars)
            if clean and clean not in validated:
                validated.append(clean)

        return validated

    def seed_game_domain(
        self,
        domain_name: str,
        primitives: list[str],
        rules_description: str,
        n_seeds: int = DEFAULT_N_SEEDS,
        temperature: float = DEFAULT_SEED_TEMPERATURE,
    ) -> list[str]:
        """Synthesizes agent strategy heuristics for game/control environments."""
        prompt = build_game_prompt(
            domain_name=domain_name,
            primitives=primitives,
            rules_description=rules_description,
            n_candidates=n_seeds,
        )
        raw_candidates = self.client.generate_candidates(prompt=prompt, temperature=temperature)
        allowed_vars = set(primitives)

        validated = []
        for cand in raw_candidates:
            clean = validate_metta_sexpr(cand, allowed_variables=allowed_vars)
            if clean and clean not in validated:
                validated.append(clean)

        return validated

    def seed_by_problem_name(
        self,
        problem_name: str,
        n_seeds: int = DEFAULT_N_SEEDS,
        temperature: float = DEFAULT_SEED_TEMPERATURE,
    ) -> list[str]:
        """Convenience dispatcher to seed standard demo benchmark problems."""
        prob_key = problem_name.lower().strip()
        if prob_key in DEMO_TRUTH_TABLES:
            prob = DEMO_TRUTH_TABLES[prob_key]
            return self.seed_truth_table(
                problem_name=prob_key,
                labels=prob["labels"],
                rows=prob["rows"],
                n_seeds=n_seeds,
                temperature=temperature,
            )
        elif "tictactoe" in prob_key or "tic-tac-toe" in prob_key:
            return self.seed_game_domain(
                domain_name="Tic-Tac-Toe",
                primitives=["playwin", "playblock", "playcenter", "playcorner", "playside"],
                rules_description="Prioritize moves to maximize winning or defensive blocking chances.",
                n_seeds=n_seeds,
                temperature=temperature,
            )
        else:
            logger.warning(f"Unknown problem name: {problem_name}. Defaulting to parity3.")
            prob = DEMO_TRUTH_TABLES["parity3"]
            return self.seed_truth_table(
                problem_name="parity3",
                labels=prob["labels"],
                rows=prob["rows"],
                n_seeds=n_seeds,
                temperature=temperature,
            )
        
# [DEBUG PRINT] print(f"[DEBUG LLMSeeder] Validated seed: {clean}")59

# [DEBUG PRINT] print(f"[DEBUG LLMSeeder] Total valid distinct seeds: {len(validated)}")61
# [DEBUG PRINT] print(f"[DEBUG LLMSeeder] Seeding from CSV '{csv_path}' (target={target_feature})")72
# [DEBUG PRINT] print(f"[DEBUG LLMSeeder] Validated CSV seed: {clean}")91
# [DEBUG PRINT] print(f"[DEBUG LLMSeeder] Seeding game domain '{domain_name}' with primitives {primitives}")104
# [DEBUG PRINT] print(f"[DEBUG LLMSeeder] Validated game strategy seed: {clean}")119
# [DEBUG PRINT] print(f"[DEBUG LLMSeeder] Dispatching problem name '{prob_key}'")131