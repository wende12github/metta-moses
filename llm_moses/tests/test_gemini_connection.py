import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from llm_moses.config.gemini_client import GeminiClient
from llm_moses.config.settings import DEFAULT_MODEL


# Test live connectivity to Google Gemini using GeminiClient.
def test_gemini_connection():
    client = GeminiClient(model_name=DEFAULT_MODEL)
    assert client.api_key is not None

    prompt = """Synthesize 2 MeTTa S-expression boolean programs solving 2-bit XOR for inputs [A, B].
Return ONLY a JSON array of 2 strings:
[
  "(OR (AND A (NOT B)) (AND (NOT A) B))",
  "(AND (OR A B) (NOT (AND A B)))"
]
"""
    candidates = client.generate_candidates(prompt=prompt, temperature=0.2)
    assert len(candidates) >= 1
    print(f"Connection test succeeded! Received {len(candidates)} candidates: {candidates}")


if __name__ == "__main__":
    test_gemini_connection()
