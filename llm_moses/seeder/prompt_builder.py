from typing import Optional


# Prompt Builder for LLM Initial Population Seeding.
# Constructs few-shot, grammar-constrained prompts for truth tables, CSVs, and game policies.
def build_truth_table_prompt(
    problem_name: str,
    labels: list[str],
    rows: list[list[bool]],
    n_candidates: int = 5,
) -> str:
    """Constructs a prompt for boolean truth table program synthesis."""
    inputs = labels[:-1]
    target = labels[-1]

    table_rows_str = []
    table_rows_str.append(f"| {' | '.join(inputs)} | {target} |")
    table_rows_str.append(f"|{'---|' * len(inputs)}---|")

    for r in rows:
        formatted_row = [("1" if val else "0") for val in r]
        table_rows_str.append(f"| {' | '.join(formatted_row[:-1])} | {formatted_row[-1]} |")

    markdown_table = "\n".join(table_rows_str)

    prompt = f"""You are an expert Genetic Programming and MeTTa language program synthesizer.
        Your task is to synthesize {n_candidates} diverse, syntactically valid initial candidate programs in MeTTa S-expression syntax to solve the boolean problem: '{problem_name}'.

        ### Problem Specification:
        - Input Variables: {inputs}
        - Target Output: {target}

        ### Truth Table:
        {markdown_table}

        ### Grammar & Syntax Rules (CRITICAL):
        1. Use standard MeTTa S-expression syntax with prefix operators:
        - `(AND <expr1> <expr2> ...)` : Logical conjunction (2 or more arguments)
        - `(OR <expr1> <expr2> ...)`  : Logical disjunction (2 or more arguments)
        - `(NOT <expr>)`             : Logical negation (EXACTLY 1 argument)
        - Variable leaves: ONLY use variables from {inputs}, or constant literals `True` / `False`.
        2. Every open parenthesis `(` must have a matching closing parenthesis `)`.
        3. Do NOT use infix notation (e.g. do NOT use `A AND B`). Always use prefix notation: `(AND A B)`.
        4. Do NOT include markdown code blocks or explanations, return ONLY a JSON array of {n_candidates} string expressions.

        ### Strategy & Diversity:
        Generate {n_candidates} structurally different candidate hypotheses:
        - Candidate 1: Disjunctive Normal Form (DNF / sum-of-products) covering minterms.
        - Candidate 2: Conjunctive Normal Form (CNF / product-of-sums) covering maxterms.
        - Candidate 3: Compact hierarchical/nested boolean expression.
        - Candidate 4 & 5: Alternative heuristic groupings or parity/majority logic approximations.

        Output MUST be a valid JSON array of {n_candidates} strings. Example:
        [
        "(OR (AND {inputs[0]} {inputs[1]}) (AND (NOT {inputs[0]}) (NOT {inputs[1]})))",
        "(AND (OR {inputs[0]} {inputs[1]}) (NOT (AND {inputs[0]} {inputs[1]})))"
        ]
    """
    return prompt


def build_csv_prompt(
    labels: list[str],
    rows: list[list[bool]],
    target_name: Optional[str] = None,
    n_candidates: int = 5,
    max_sample_rows: int = 16,
) -> str:
    """Constructs a prompt for CSV tabular classification problems."""
    if not target_name:
        target_name = labels[-1]
        inputs = labels[:-1]
    else:
        target_idx = labels.index(target_name)
        inputs = [l for i, l in enumerate(labels) if i != target_idx]

    total_rows = len(rows)
    pos_rows = [r for r in rows if r[-1] is True]
    neg_rows = [r for r in rows if r[-1] is False]

    sample_rows = rows[:max_sample_rows]

    table_rows_str = []
    table_rows_str.append(f"| {' | '.join(inputs)} | {target_name} |")
    table_rows_str.append(f"|{'---|' * len(inputs)}---|")
    for r in sample_rows:
        formatted_row = [("1" if val else "0") for val in r]
        table_rows_str.append(f"| {' | '.join(formatted_row[:-1])} | {formatted_row[-1]} |")

    markdown_table = "\n".join(table_rows_str)

    prompt = f"""You are an expert Genetic Programming program synthesizer.
        Synthesize {n_candidates} diverse MeTTa S-expression boolean classifier programs for the following tabular dataset.

        ### Dataset Overview:
        - Total Samples: {total_rows} ({len(pos_rows)} Positive, {len(neg_rows)} Negative)
        - Features: {inputs}
        - Target Feature: {target_name}

        ### Sample Data:
        {markdown_table}

        ### Grammar & Syntax Rules (CRITICAL):
        1. MeTTa Prefix S-expression syntax:
        - `(AND <expr1> <expr2> ...)`
        - `(OR <expr1> <expr2> ...)`
        - `(NOT <expr>)` (Must have exactly 1 child)
        - Leaves: ONLY feature names from {inputs}, or `True`/`False`.
        2. Strictly balanced parentheses.
        3. Generate {n_candidates} distinct structural candidate hypotheses with varying feature combinations.

        Output MUST be a JSON array of {n_candidates} S-expression strings.
    """
    return prompt


def build_game_prompt(
    domain_name: str,
    primitives: list[str],
    rules_description: str,
    n_candidates: int = 5,
) -> str:
    """Constructs a prompt for game policy / agent strategy synthesis."""
    primitives_str = ", ".join(primitives)

    prompt = f"""You are an expert Game AI and Genetic Programming synthesizer.
        Your task is to synthesize {n_candidates} high-performing agent strategy expressions for: '{domain_name}'.

        ### Domain Description & Rules:
        {rules_description}

        ### Available Primitive Actions:
        The only allowed action leaves are:
        {primitives_str}

        ### Strategy Combinator Grammar (CRITICAL):
        Strategies are prioritized decision cascades evaluated with `PRIORITIZED-OR`:
        - `(PRIORITIZED-OR <action1> <action2> ...)`
        - Evaluation semantics: Tries `<action1>`. If it returns False (action unavailable or invalid), tries `<action2>`, and so forth until an action succeeds.

        ### Syntax Rules:
        1. Every expression MUST be enclosed in `(PRIORITIZED-OR ...)` containing 2 or more primitive actions from {primitives}.
        2. Balanced parentheses are strictly required.
        3. Provide {n_candidates} diverse tactical prioritizations (e.g. aggressive win-first, defensive block-first, center-control variants, etc.).

        Output MUST be a JSON array of {n_candidates} strings. Example:
        [
            "(PRIORITIZED-OR playwin playblock playcenter playcorner playside)",
            "(PRIORITIZED-OR playblock playwin playcenter playcorner playside)",
            "(PRIORITIZED-OR playwin playcenter playblock playcorner playside)"
        ]
    """
    return prompt
