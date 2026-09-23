# Global configuration and default settings for LLM-MOSES.

DEFAULT_MODEL: str = "gemini-3.1-flash"
FALLBACK_MODEL: str = "gemini-3.7-flash"

# Seeding hyperparameters (Macro-exploration)
DEFAULT_N_SEEDS: int = 5
DEFAULT_SEED_TEMPERATURE: float = 0.7

# Mutation & Variation hyperparameters (Micro-refinement)
DEFAULT_N_MUTANTS: int = 3
DEFAULT_MUTATE_TEMPERATURE: float = 0.4

# API Rate-limiting and Retry parameters
DEFAULT_MAX_RETRIES: int = 4
DEFAULT_INITIAL_BACKOFF_SECONDS: float = 3.0
DEFAULT_BACKOFF_MULTIPLIER: float = 2.0
