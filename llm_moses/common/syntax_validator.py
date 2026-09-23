import logging
import re
from typing import Any, Union

logger = logging.getLogger(__name__)

# Standard MeTTa MOSES Boolean grammar keywords
ALLOWED_OPERATORS = {"AND", "OR", "NOT", "PRIORITIZED-OR"}
BOOLEAN_LITERALS = {"true", "false", "True", "False"}


def is_balanced_parens(s: str) -> bool:
    """Checks if parentheses in a string are well-balanced."""
    depth = 0
    s = s.strip()
    if not s:
        return False
    if "(" in s or ")" in s:
        if not (s.startswith("(") and s.endswith(")")):
            # [DEBUG PRINT] print(f"[DEBUG SyntaxValidator] Parens not enclosing string: '{s}'")
            return False
    for char in s:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                # [DEBUG PRINT] print(f"[DEBUG SyntaxValidator] Closing paren before open: '{s}'")
                return False
    # [DEBUG PRINT] print(f"[DEBUG SyntaxValidator] Parens balanced check for '{s}': {depth == 0}")
    return depth == 0


def sexpr_to_ast(s: str) -> Union[str, list]:
    """Parses an S-expression string into nested Python lists/strings."""
    tokens = re.findall(r"\(|\)|[^\s()]+", s.strip())
    if not tokens:
        raise ValueError("Empty S-expression")

    def parse(toks):
        if not toks:
            raise ValueError("Unexpected end of tokens")
        token = toks.pop(0)
        if token == "(":
            ast = []
            while toks and toks[0] != ")":
                ast.append(parse(toks))
            if not toks:
                raise ValueError("Missing closing parenthesis")
            toks.pop(0)  # remove ')'
            return ast
        elif token == ")":
            raise ValueError("Unexpected ')'")
        else:
            return token

    result = parse(tokens)
    if tokens:
        raise ValueError("Extra trailing tokens after S-expression")
    # [DEBUG PRINT] print(f"[DEBUG SyntaxValidator] Parsed sexpr '{s}' -> AST: {result}")
    return result


def ast_to_sexpr(ast: Any) -> str:
    """Converts a parsed AST back to a canonical MeTTa S-expression string."""
    if isinstance(ast, list):
        return "(" + " ".join(ast_to_sexpr(child) for child in ast) + ")"
    elif isinstance(ast, str):
        if ast.lower() == "true":
            return "true"
        elif ast.lower() == "false":
            return "false"
        return ast
    return str(ast)


def is_valid_syntax(
    s: str,
    allowed_variables: set[str] | None = None,
    max_depth: int = 10,
) -> bool:
    """Validates that an S-expression is syntactically well-formed for MeTTa MOSES."""
    if not is_balanced_parens(s):
        # [DEBUG PRINT] print(f"[DEBUG SyntaxValidator] Rejected '{s}' -> Unbalanced parens")
        return False

    try:
        ast = sexpr_to_ast(s)
    except Exception as e:
        # [DEBUG PRINT] print(f"[DEBUG SyntaxValidator] Rejected '{s}' -> AST parse error: {e}")
        return False

    def check_node(node: Any, current_depth: int) -> bool:
        if current_depth > max_depth:
            # [DEBUG PRINT] print(f"[DEBUG SyntaxValidator] Rejected node -> Exceeded max depth {max_depth}")
            return False

        if isinstance(node, str):
            if node in BOOLEAN_LITERALS or node.lower() in {"true", "false"}:
                return True
            if allowed_variables is not None:
                is_allowed = node in allowed_variables or node.upper() in allowed_variables
                if not is_allowed:
                    # [DEBUG PRINT] print(f"[DEBUG SyntaxValidator] Rejected leaf '{node}' -> Not in allowed vars {allowed_variables}")
                    pass
                return is_allowed
            return bool(re.match(r"^[a-zA-Z0-9_\-]+$", node))

        if isinstance(node, list):
            if not node:
                return False
            if len(node) == 1 and isinstance(node[0], str):
                return check_node(node[0], current_depth + 1)

            head = node[0]
            if not isinstance(head, str):
                return False

            op = head.upper()
            if op not in ALLOWED_OPERATORS:
                # [DEBUG PRINT] print(f"[DEBUG SyntaxValidator] Rejected operator '{op}' -> Not in ALLOWED_OPERATORS {ALLOWED_OPERATORS}")
                return False

            args = node[1:]
            if op == "NOT":
                if len(args) != 1:
                    # [DEBUG PRINT] print(f"[DEBUG SyntaxValidator] Rejected NOT with {len(args)} args (expected 1)")
                    return False
                return check_node(args[0], current_depth + 1)
            elif op in {"AND", "OR", "PRIORITIZED-OR"}:
                if not args:
                    # [DEBUG PRINT] print(f"[DEBUG SyntaxValidator] Rejected {op} with 0 args")
                    return False
                return all(check_node(arg, current_depth + 1) for arg in args)

            return False

        return False

    valid = check_node(ast, 1)
    # [DEBUG PRINT] print(f"[DEBUG SyntaxValidator] is_valid_syntax for '{s}' -> {valid}")
    return valid


def validate_metta_sexpr(
    expr: str,
    allowed_variables: set[str] | None = None,
) -> str | None:
    """Validates and canonicalizes a MeTTa candidate expression."""
    expr = expr.strip()
    if not is_valid_syntax(expr, allowed_variables):
        # [DEBUG PRINT] print(f"[DEBUG SyntaxValidator] validate_metta_sexpr: '{expr}' REJECTED")
        return None
    try:
        ast = sexpr_to_ast(expr)
        canonical = ast_to_sexpr(ast)
        # [DEBUG PRINT] print(f"[DEBUG SyntaxValidator] validate_metta_sexpr: '{expr}' -> VALID CANONICAL: '{canonical}'")
        return canonical
    except Exception:
        return None
