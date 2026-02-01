# bayesian-llm

Normative Bayesian baselines and mechanistic experiments for **Bayesian cognition in LLMs**.

This repo is built around a simple workflow:
1. Define a **normative Bayesian posterior predictive** for a toy inference problem (e.g., urn-style sequential evidence).
2. Measure an LLM’s **next-token probabilities** on the same task.
3. Quantify alignment/deviation from the normative baseline.
4. Localize and test causal mechanisms (e.g., induction / order sensitivity) using **TransformerLens** hooks and ablations.

## What’s inside

- `bayesian_llm/` — small Python utilities:
  - `bayes.py`: posterior predictive helpers (Beta–Bernoulli, discrete hypotheses, two-generator urn task).
  - `llm.py`: HuggingFace causal LM loading + next-token probability helpers (incl. tokenizer-variant handling).
  - `data.py`: simple sequence constructors/permutations.
  - `metrics.py`: lightweight metrics (MAE, Pearson correlation).
- `urn_task.ipynb` — tests whether a model behaves like a Bayesian observer on a two-urn sequential inference task.
- `notebooks/01_behavioral_tests.ipynb` — behavioral evaluation vs normative Bayes; compares deviations to alternative cognitive models.
- `notebooks/02_localization.ipynb` — localization analysis.
- `notebooks/03_causal_interventions.ipynb` — causal interventions/ablations.
- `notebooks/toy_induction.ipynb` — minimal induction circuit / order-sensitivity reproduction.
- `results_urn_task/` — saved results (probe weights/posteriors, summary CSVs).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For some HF models (e.g., Llama 3.x), you may need:
- an HF account with accepted license terms
- `HF_TOKEN` set in your environment

## Quick usage

### 1) Compute a normative posterior predictive

```python
from bayesian_llm.bayes import two_generator_posterior_predictive

# canonical urn task: generator A has P(X)=0.50, generator B has P(X)=0.75
p_next_x = two_generator_posterior_predictive(n_x=12, n_total=20)
print(p_next_x)
```

### 2) Query an LLM’s next-token probability of X vs Y

```python
from bayesian_llm.llm import load_hf_causal_lm, normalized_next_token_prob

m = load_hf_causal_lm("gpt2")

p_x = normalized_next_token_prob(
    m.model,
    m.tokenizer,
    prompt="Sequence so far: X Y X X Y ...\nNext token:",
    a_variants=["X", " X"],
    b_variants=["Y", " Y"],
)
print(p_x)
```

## Notes

- Many analyses assume next-token variants are **single tokens** (tokenization matters).
- For interpretability experiments, notebooks use **TransformerLens**.

## License

Not specified (inherit repository defaults).
