from llm_moses.mutator.error_profiler import ErrorProfile


# Prompt Templates for LLM Semantic Mutation, Semantic Crossover, and Anti-Bloat Refactoring.
class VariationPromptBuilder:
    """Builds targeted prompts for semantic program modification."""

    @staticmethod
    def build_mutation_prompt(
        parent_sexpr: str,
        error_profile: ErrorProfile,
        allowed_leaves: list[str],
        domain: str = "boolean",
        n_mutants: int = 3,
    ) -> str:
        """Constructs a prompt for targeted, error-driven semantic mutation."""
        failure_summary = error_profile.format_failure_summary()

        prompt = f"""You are an expert Genetic Programming Semantic Mutation Operator for the MeTTa programming language.
            Your task is to repair and mutate the following Parent Program to fix its specific logic errors.

            ### Parent Program (MeTTa S-Expression):
            {parent_sexpr}

            ### Performance Breakdown:
            - Accuracy: {error_profile.accuracy * 100:.1f}% ({error_profile.passed_count}/{error_profile.total} test cases correct)
            - Errors: {error_profile.failed_count} failing cases

            ### Specific Failing Cases to Fix:
            {failure_summary}

            ### Allowed Variable Leaves:
            {allowed_leaves}

            ### Grammar & Syntax Rules (CRITICAL):
            1. Prefix MeTTa S-expression syntax:
            - `(AND <expr1> <expr2> ...)`
            - `(OR <expr1> <expr2> ...)`
            - `(NOT <expr>)` (Must have exactly 1 argument)
            - Only use variables from {allowed_leaves}, or constants `True`/`False`.
            2. Strictly balanced parentheses.
            3. Output MUST be a JSON array of {n_mutants} mutated S-expression strings.

            ### Task:
            Diagnose the logical flaw in the parent program. Apply targeted surgical repairs (flipping terms, swapping sub-clauses, adding missing guard conditions) so that the failing cases are resolved without breaking the passed cases.

            Return ONLY a JSON array of {n_mutants} strings.
        """
        return prompt

    @staticmethod
    def build_crossover_prompt(
        parent_a_sexpr: str,
        parent_b_sexpr: str,
        profile_a: ErrorProfile,
        profile_b: ErrorProfile,
        allowed_leaves: list[str],
        domain: str = "boolean",
        n_children: int = 3,
    ) -> str:
        """Constructs a prompt for semantic crossover between two complementary parents."""
        summary_a = profile_a.format_failure_summary()
        summary_b = profile_b.format_failure_summary()

        prompt = f"""You are an expert Genetic Programming Semantic Crossover Operator.
            Your task is to perform semantic crossover by combining the complementary strengths of two parent programs.

            ### Parent A (Accuracy: {profile_a.accuracy * 100:.1f}%):
            {parent_a_sexpr}
            Failures of Parent A:
            {summary_a}

            ### Parent B (Accuracy: {profile_b.accuracy * 100:.1f}%):
            {parent_b_sexpr}
            Failures of Parent B:
            {summary_b}

            ### Allowed Variable Leaves:
            {allowed_leaves}

            ### Grammar & Syntax Rules (CRITICAL):
            1. Valid MeTTa S-expression syntax using `(AND ...)`, `(OR ...)`, `(NOT ...)`, `True`, `False`.
            2. Only use variables from {allowed_leaves}.
            3. Every open parenthesis must be closed.

            ### Task:
            Identify which sub-expressions in Parent A correctly solve parts of the problem that Parent B fails on (and vice-versa).
            Synthesize {n_children} unified candidate programs that blend these complementary logical structures into a single superior program.

            Return ONLY a valid JSON array of {n_children} S-expression strings.
        """
        return prompt

    @staticmethod
    def build_refactoring_prompt(
        parent_sexpr: str,
        allowed_leaves: list[str],
        domain: str = "boolean",
        n_variants: int = 2,
    ) -> str:
        """Constructs a prompt for anti-bloat structural refactoring."""
        prompt = f"""You are an expert Program Synthesizer and Boolean Minimizer.
            Your task is to simplify and refactor a bloated MeTTa S-expression into its minimal, canonical equivalent form without changing its boolean truth value on any input.

            ### Bloated Program:
            {parent_sexpr}

            ### Allowed Variable Leaves:
            {allowed_leaves}

            ### Grammar & Rules:
            1. Valid MeTTa prefix syntax: `(AND ...)`, `(OR ...)`, `(NOT ...)`.
            2. Strictly eliminate redundant terms, subsumed clauses, and repetitive sub-trees.
            3. Preserve 100% exact semantic equivalence.

            Output MUST be a JSON array of {n_variants} simplified S-expression strings.
        """
        return prompt
