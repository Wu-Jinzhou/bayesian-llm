from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


def _as_np(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def beta_bernoulli_posterior_predictive(
    *,
    alpha: float,
    beta: float,
    n_success: int,
    n_total: int,
) -> float:
    """Posterior predictive mean for Bernoulli with Beta(alpha,beta) prior."""
    if n_total < 0 or n_success < 0 or n_success > n_total:
        raise ValueError("Require 0 <= n_success <= n_total.")
    return (alpha + n_success) / (alpha + beta + n_total)


@dataclass(frozen=True)
class DiscreteHypothesis:
    name: str
    p_success: float


def discrete_posterior_predictive(
    *,
    n_success: int,
    n_total: int,
    hypotheses: Sequence[DiscreteHypothesis],
    priors: Sequence[float] | None = None,
) -> tuple[float, dict[str, float]]:
    """
    Posterior predictive under a discrete latent hypothesis H with known p(success|H).

    Returns:
      p_next_success, posterior_probs_by_name
    """
    if n_total < 0 or n_success < 0 or n_success > n_total:
        raise ValueError("Require 0 <= n_success <= n_total.")
    if len(hypotheses) == 0:
        raise ValueError("Need at least one hypothesis.")

    p = _as_np([h.p_success for h in hypotheses])
    if np.any((p <= 0.0) | (p >= 1.0)):
        raise ValueError("Each hypothesis p_success must be in (0,1).")

    if priors is None:
        pri = np.full(len(hypotheses), 1.0 / len(hypotheses), dtype=np.float64)
    else:
        pri = _as_np(priors)
        if pri.shape != (len(hypotheses),):
            raise ValueError("Priors shape mismatch with hypotheses.")
        if np.any(pri < 0.0) or pri.sum() == 0.0:
            raise ValueError("Priors must be non-negative and not all zero.")
        pri = pri / pri.sum()

    n_fail = n_total - n_success
    # Work in log space for numerical stability.
    log_lik = n_success * np.log(p) + n_fail * np.log1p(-p)
    log_post_unnorm = np.log(pri) + log_lik
    log_post_unnorm -= log_post_unnorm.max()
    post = np.exp(log_post_unnorm)
    post = post / post.sum()

    p_next = float((p * post).sum())
    post_dict = {h.name: float(post[i]) for i, h in enumerate(hypotheses)}
    return p_next, post_dict


def two_generator_posterior_predictive(*, n_x: int, n_total: int) -> float:
    """
    Convenience for the canonical task:
      Generator A: P(X)=0.50, Generator B: P(X)=0.75, uniform prior over {A,B}.

    Returns P(next token is X | evidence).
    """
    hypotheses = [
        DiscreteHypothesis(name="A", p_success=0.50),
        DiscreteHypothesis(name="B", p_success=0.75),
    ]
    p_next, _ = discrete_posterior_predictive(
        n_success=n_x, n_total=n_total, hypotheses=hypotheses, priors=[0.5, 0.5]
    )
    return p_next

