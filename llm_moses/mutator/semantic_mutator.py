import logging
from typing import Optional

from llm_moses.config.gemini_client import GeminiClient
from llm_moses.config.settings import DEFAULT_MODEL, DEFAULT_N_MUTANTS, DEFAULT_MUTATE_TEMPERATURE
from llm_moses.common.syntax_validator import validate_metta_sexpr
from llm_moses.mutator.error_profiler import ErrorProfile
from llm_moses.mutator.prompt_templates import VariationPromptBuilder

logger = logging.getLogger(__name__)

# Semantic Mutator for Targeted Hypothesis-Driven Program Modification.
# Uses Google Gemini to diagnose failure cases and repair candidate AST logic.
class SemanticMutator:
    """Performs targeted semantic mutations and anti-bloat refactorings."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        client: Optional[GeminiClient] = None,
        api_key: Optional[str] = None,
    ):
        self.model_name = model_name
        self.client = client or GeminiClient(model_name=self.model_name, api_key=api_key)

    def mutate(
        self,
        parent_sexpr: str,
        error_profile: ErrorProfile,
        allowed_leaves: list[str],
        domain: str = "boolean",
        n_mutants: int = DEFAULT_N_MUTANTS,
        temperature: float = DEFAULT_MUTATE_TEMPERATURE,
    ) -> list[str]:
        """Generates targeted mutations repairing failing test cases."""
        # [DEBUG PRINT] print(f"[DEBUG SemanticMutator] Mutating parent: '{parent_sexpr}' (Accuracy={error_profile.accuracy*100:.1f}%, Failing={error_profile.failed_count})")
        prompt = VariationPromptBuilder.build_mutation_prompt(
            parent_sexpr=parent_sexpr,
            error_profile=error_profile,
            allowed_leaves=allowed_leaves,
            domain=domain,
            n_mutants=n_mutants,
        )

        raw_candidates = self.client.generate_candidates(
            prompt=prompt,
            temperature=temperature,
        )
        # [DEBUG PRINT] print(f"[DEBUG SemanticMutator] Raw mutant strings received: {len(raw_candidates)}")

        allowed_vars = set(allowed_leaves)
        valid_mutants = []
        seen = {parent_sexpr.strip()}

        for cand in raw_candidates:
            clean = validate_metta_sexpr(cand, allowed_variables=allowed_vars)
            if not clean:
                logger.warning(f"Discarding invalid mutant: {cand}")
                # [DEBUG PRINT] print(f"[DEBUG SemanticMutator] Discarded invalid mutant: '{cand}'")
                continue
            if clean in seen:
                # [DEBUG PRINT] print(f"[DEBUG SemanticMutator] Skipped duplicate mutant: '{clean}'")
                continue
            seen.add(clean)
            valid_mutants.append(clean)
            # [DEBUG PRINT] print(f"[DEBUG SemanticMutator] Accepted valid mutant: '{clean}'")

        # [DEBUG PRINT] print(f"[DEBUG SemanticMutator] Total valid mutants generated: {len(valid_mutants)}")
        return valid_mutants

    def refactor(
        self,
        parent_sexpr: str,
        allowed_leaves: list[str],
        domain: str = "boolean",
        temperature: float = 0.2,
    ) -> list[str]:
        """Refactors bloated expressions into minimal equivalent forms."""
        # [DEBUG PRINT] print(f"[DEBUG SemanticMutator] Refactoring bloated parent: '{parent_sexpr}'")
        prompt = VariationPromptBuilder.build_refactoring_prompt(
            parent_sexpr=parent_sexpr,
            allowed_leaves=allowed_leaves,
            domain=domain,
            n_variants=2,
        )

        raw_candidates = self.client.generate_candidates(
            prompt=prompt,
            temperature=temperature,
        )

        allowed_vars = set(allowed_leaves)
        valid_refactors = []
        seen = {parent_sexpr.strip()}

        for cand in raw_candidates:
            clean = validate_metta_sexpr(cand, allowed_variables=allowed_vars)
            if not clean or clean in seen:
                continue
            seen.add(clean)
            valid_refactors.append(clean)
            # [DEBUG PRINT] print(f"[DEBUG SemanticMutator] Accepted refactored candidate: '{clean}'")
            
        # [DEBUG PRINT] print(f"[DEBUG SemanticMutator] Total refactored variants: {len(valid_refactors)}")
        return valid_refactors
