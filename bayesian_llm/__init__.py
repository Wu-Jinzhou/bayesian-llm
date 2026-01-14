"""Utilities for Bayesian-cognition-in-LLMs experiments."""

from .bayes import (
    beta_bernoulli_posterior_predictive,
    discrete_posterior_predictive,
    two_generator_posterior_predictive,
)
from .data import (
    make_sequence,
    permute_sequence,
    set_seed,
)
from .llm import (
    load_hf_causal_lm,
    normalized_next_token_prob,
    next_token_distribution,
)
from .metrics import (
    mae,
    pearson_r,
)

__all__ = [
    "beta_bernoulli_posterior_predictive",
    "discrete_posterior_predictive",
    "two_generator_posterior_predictive",
    "make_sequence",
    "permute_sequence",
    "set_seed",
    "load_hf_causal_lm",
    "normalized_next_token_prob",
    "next_token_distribution",
    "mae",
    "pearson_r",
]

