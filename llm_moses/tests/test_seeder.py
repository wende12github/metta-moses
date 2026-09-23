import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from llm_moses.seeder.seeder_main import LLMSeeder
from llm_moses.common.simple_evaluator import evaluate_expression_on_truth_table
from llm_moses.common.data_loader import DEMO_TRUTH_TABLES


# End-to-end integration tests for LLMSeeder.
def test_parity3_seeding():
    seeder = LLMSeeder()
    cands = seeder.seed_by_problem_name("parity3", n_seeds=3)
    assert len(cands) >= 1
    print(f"Parity-3 Candidates: {cands}")

    prob = DEMO_TRUTH_TABLES["parity3"]
    scores = [
        evaluate_expression_on_truth_table(c, prob["labels"], prob["rows"])[0]
        for c in cands
    ]
    max_acc = max(scores)
    print(f"Max candidate accuracy for Parity-3: {max_acc * 100:.1f}%")
    assert max_acc >= 0.5


def test_majority3_seeding():
    seeder = LLMSeeder()
    cands = seeder.seed_by_problem_name("majority3", n_seeds=3)
    assert len(cands) >= 1
    print(f"Majority-3 Candidates: {cands}")

    prob = DEMO_TRUTH_TABLES["majority3"]
    scores = [
        evaluate_expression_on_truth_table(c, prob["labels"], prob["rows"])[0]
        for c in cands
    ]
    max_acc = max(scores)
    print(f"Max candidate accuracy for Majority-3: {max_acc * 100:.1f}%")
    assert max_acc >= 0.7


if __name__ == "__main__":
    test_parity3_seeding()
    test_majority3_seeding()
    print("All seeder tests passed!")
