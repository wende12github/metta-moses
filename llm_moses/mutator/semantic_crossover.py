import logging
from typing import Optional

from llm_moses.config.gemini_client import GeminiClient
from llm_moses.config.settings import DEFAULT_MODEL, DEFAULT_N_MUTANTS, DEFAULT_MUTATE_TEMPERATURE
from llm_moses.common.syntax_validator import validate_metta_sexpr
from llm_moses.mutator.error_profiler import ErrorProfile
from llm_moses.mutator.prompt_templates import VariationPromptBuilder

logger = logging.getLogger(__name__)

# Semantic Crossover for Synthesizing Unified Programs from Complementary Parents.
class SemanticCrossover:
    """Combines complementary candidate programs through semantic LLM reasoning."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        client: Optional[GeminiClient] = None,
        api_key: Optional[str] = None,
    ):
        self.model_name = model_name
        self.client = client or GeminiClient(model_name=self.model_name, api_key=api_key)

    def crossover(
        self,
        parent_a_sexpr: str,
        parent_b_sexpr: str,
        profile_a: ErrorProfile,
        profile_b: ErrorProfile,
        allowed_leaves: list[str],
        domain: str = "boolean",
        n_children: int = DEFAULT_N_MUTANTS,
        temperature: float = DEFAULT_MUTATE_TEMPERATURE,
    ) -> list[str]:
        """Performs semantic crossover between two parents."""
        # [DEBUG PRINT] print(f"[DEBUG SemanticCrossover] Crossing Parent A ({profile_a.accuracy*100:.1f}%) and Parent B ({profile_b.accuracy*100:.1f}%)")
        # [DEBUG PRINT] print(f"[DEBUG SemanticCrossover] Parent A: '{parent_a_sexpr}'")
        # [DEBUG PRINT] print(f"[DEBUG SemanticCrossover] Parent B: '{parent_b_sexpr}'")

        prompt = VariationPromptBuilder.build_crossover_prompt(
            parent_a_sexpr=parent_a_sexpr,
            parent_b_sexpr=parent_b_sexpr,
            profile_a=profile_a,
            profile_b=profile_b,
            allowed_leaves=allowed_leaves,
            domain=domain,
            n_children=n_children,
        )

        raw_candidates = self.client.generate_candidates(
            prompt=prompt,
            temperature=temperature,
        )
        # [DEBUG PRINT] print(f"[DEBUG SemanticCrossover] Raw crossover children received: {len(raw_candidates)}")

        allowed_vars = set(allowed_leaves)
        valid_children = []
        seen = {parent_a_sexpr.strip(), parent_b_sexpr.strip()}

        for cand in raw_candidates:
            clean = validate_metta_sexpr(cand, allowed_variables=allowed_vars)
            if not clean:
                logger.warning(f"Discarding invalid crossover child: {cand}")
                # [DEBUG PRINT] print(f"[DEBUG SemanticCrossover] Discarded invalid child: '{cand}'")
                continue
            if clean in seen:
                # [DEBUG PRINT] print(f"[DEBUG SemanticCrossover] Skipped duplicate/parent child: '{clean}'")
                continue
            seen.add(clean)
            valid_children.append(clean)
            # [DEBUG PRINT] print(f"[DEBUG SemanticCrossover] Accepted valid crossover child: '{clean}'")

        # [DEBUG PRINT] print(f"[DEBUG SemanticCrossover] Total valid crossover children: {len(valid_children)}")
        return valid_children