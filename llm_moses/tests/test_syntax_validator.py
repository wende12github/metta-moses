import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from llm_moses.common.syntax_validator import (
    is_balanced_parens,
    is_valid_syntax,
    validate_metta_sexpr,
    sexpr_to_ast,
    ast_to_sexpr,
)


# Unit tests for Syntax Validator and S-expression parser.
def test_balanced_parentheses():
    assert is_balanced_parens("(AND X1 X2)")
    assert is_balanced_parens("(OR (AND X1 X2) (NOT X3))")
    assert not is_balanced_parens("(AND X1 X2")
    assert not is_balanced_parens("AND X1 X2)")
    assert not is_balanced_parens("")
    assert not is_balanced_parens("(AND (OR X1) ) )")


def test_ast_conversions():
    s = "(OR (AND X1 (NOT X2)) X3)"
    ast = sexpr_to_ast(s)
    assert ast == ["OR", ["AND", "X1", ["NOT", "X2"]], "X3"]
    reconstructed = ast_to_sexpr(ast)
    assert reconstructed == s


def test_syntax_validation():
    allowed = {"X1", "X2", "X3"}

    # Valid prefix S-expressions
    assert is_valid_syntax("(AND X1 X2)", allowed_variables=allowed)
    assert is_valid_syntax("(OR (AND X1 X2) (NOT X3))", allowed_variables=allowed)
    assert is_valid_syntax("(PRIORITIZED-OR X1 X2)", allowed_variables=allowed)
    assert is_valid_syntax("true", allowed_variables=allowed)
    assert is_valid_syntax("(true)", allowed_variables=allowed)

    # Invalid cases
    assert not is_valid_syntax("(NAND X1 X2)", allowed_variables=allowed)  # Unknown operator
    assert not is_valid_syntax("(AND X1 Unknown)", allowed_variables=allowed)  # Unknown var
    assert not is_valid_syntax("(X1 AND X2)", allowed_variables=allowed)  # Infix
    assert not is_valid_syntax("(NOT X1 X2)", allowed_variables=allowed)  # NOT takes 1 arg
    assert not is_valid_syntax("(AND)", allowed_variables=allowed)  # Empty args


def test_validate_metta_sexpr():
    allowed = {"X1", "X2"}
    assert validate_metta_sexpr("(AND X1 X2)", allowed) == "(AND X1 X2)"
    assert validate_metta_sexpr("(INVALID X1)", allowed) is None


if __name__ == "__main__":
    test_balanced_parentheses()
    test_ast_conversions()
    test_syntax_validation()
    test_validate_metta_sexpr()
    print("All syntax validator tests passed!")
