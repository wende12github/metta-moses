import csv
from pathlib import Path
from typing import Union

# Common dataset and problem loaders for LLM-MOSES benchmarks.
# Standard truth tables for benchmark problems
DEMO_TRUTH_TABLES: dict[str, dict] = {
    "parity3": {
        "labels": ["X1", "X2", "X3", "Output"],
        "rows": [
            [False, False, False, False],
            [False, False, True, True],
            [False, True, False, True],
            [False, True, True, False],
            [True, False, False, True],
            [True, False, True, False],
            [True, True, False, False],
            [True, True, True, True],
        ],
        "description": "3-bit Odd Parity: Output is true if an odd number of inputs are true.",
    },
    "majority3": {
        "labels": ["X1", "X2", "X3", "Output"],
        "rows": [
            [False, False, False, False],
            [False, False, True, False],
            [False, True, False, False],
            [False, True, True, True],
            [True, False, False, False],
            [True, False, True, True],
            [True, True, False, True],
            [True, True, True, True],
        ],
        "description": "3-bit Majority: Output is true if at least 2 of the 3 inputs are true.",
    },
    "mux3": {
        "labels": ["A", "D0", "D1", "Output"],
        "rows": [
            [False, False, False, False],
            [False, False, True, False],
            [False, True, False, True],
            [False, True, True, True],
            [True, False, False, False],
            [True, False, True, True],
            [True, True, False, False],
            [True, True, True, True],
        ],
        "description": "3-bit Multiplexer (Mux-3): If address A is false output D0, else output D1.",
    },
    "xor2": {
        "labels": ["A", "B", "Output"],
        "rows": [
            [False, False, False],
            [False, True, True],
            [True, False, True],
            [True, True, False],
        ],
        "description": "2-bit XOR: Output is true if exactly one input is true.",
    },
    "demorgan": {
        "labels": ["A", "B", "Output"],
        "rows": [
            [False, False, True],
            [False, True, True],
            [True, False, True],
            [True, True, False],
        ],
        "description": "De Morgan NAND: Output is true if NOT (A AND B).",
    },
}


def _to_bool_val(val: str) -> bool:
    # Helper to convert string/numeric values to boolean.
    v = str(val).strip().lower()
    return v in {"1", "true", "t", "yes", "y"}


def load_csv_data(
    csv_path: Union[str, Path],
    target_feature: str = "",
) -> tuple[list[str], list[list[bool]]]:
    """Loads a CSV dataset and converts rows into boolean values with target column last.

    Returns:
        tuple of (reordered_labels, bool_rows)
    """
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with open(path, "r", encoding="utf-8") as f:
        reader = list(csv.reader(f))

    if not reader:
        return [], []

    labels = [h.strip() for h in reader[0]]
    body = reader[1:]

    if not target_feature or str(target_feature).strip() in {"", "()"}:
        target_idx = len(labels) - 1
        target_name = labels[target_idx]
    else:
        target_name = str(target_feature).strip()
        target_idx = labels.index(target_name)

    bool_rows = []
    for r_num, row in enumerate(body, start=2):
        if not row or len(row) < len(labels):
            continue
        b_row = [_to_bool_val(row[i]) for i in range(len(labels))]
        reordered = [b for i, b in enumerate(b_row) if i != target_idx]
        reordered.append(b_row[target_idx])
        bool_rows.append(reordered)

    reordered_labels = [l for i, l in enumerate(labels) if i != target_idx]
    reordered_labels.append(target_name)
    return reordered_labels, bool_rows


def load_problem_data(
    problem_name: str = "parity3",
    input_file: str = "",
    target_feature: str = "",
) -> tuple[list[str], list[list[bool]]]:
    # Unified problem loader supporting truth tables and CSV datasets.
    if input_file and str(input_file).strip() not in {"", "()"}:
        return load_csv_data(csv_path=str(input_file).strip(), target_feature=target_feature)

    prob_key = str(problem_name).strip().lower()
    if prob_key in DEMO_TRUTH_TABLES:
        return DEMO_TRUTH_TABLES[prob_key]["labels"], DEMO_TRUTH_TABLES[prob_key]["rows"]

    # Fallback to parity3 if unknown
    return DEMO_TRUTH_TABLES["parity3"]["labels"], DEMO_TRUTH_TABLES["parity3"]["rows"]
