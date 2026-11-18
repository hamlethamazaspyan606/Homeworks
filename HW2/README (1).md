# DS 223 Marketing Analytics — Homework 2

## A/B Testing with Multi-Armed Bandits

This project compares two bandit algorithms for four advertisement options with true reward means:

```python
Bandit_Reward = [1, 2, 3, 4]
NumberOfTrials = 20000
```

Implemented algorithms:

- Epsilon-Greedy with decaying `epsilon = 1/t`
- Thompson Sampling for Gaussian rewards with known precision

## How to run

```bash
pip install -r requirements.txt
python Bandit.py
```

Running the script creates:

- `output/bandit_rewards.csv` with the required columns: `Bandit`, `Reward`, `Algorithm`
- Individual algorithm CSV files
- `output/summary.csv`
- Learning-process plots on linear and logarithmic scales
- Cumulative reward comparison plot
- Cumulative regret comparison plot

## Model choices

### Epsilon-Greedy

At trial `t`, exploration probability is:

```text
epsilon = 1/t
```

With probability epsilon, a random arm is selected. Otherwise, the arm with the highest estimated sample mean is selected.

### Thompson Sampling

Rewards are modeled as Gaussian with known precision `tau = 1`. Each unknown bandit mean has a Normal prior:

```text
prior mean m0 = 0
prior precision lambda0 = 1
```

At each trial, one value is sampled from each posterior and the arm with the largest sample is selected. The posterior mean and precision are updated after observing the reward.

## Regret

The project uses pseudo-regret:

```text
regret_t = best_true_mean - selected_bandit_true_mean
```

Cumulative regret is the running sum of this value.

## Bonus — Better implementation plan

A stronger production implementation could:

1. Separate arm objects from algorithm/controller objects so each bandit owns its own reward distribution and state.
2. Use a shared experiment interface so additional algorithms such as UCB1 can be added without changing reporting code.
3. Run many independent simulations rather than one seeded run, then compare mean cumulative reward/regret with confidence intervals.
4. Store experiment configuration in a dedicated config object or file instead of global constants.
5. Add automated tests for posterior updates, epsilon decay, CSV schema, and reproducibility.
6. Add command-line arguments for number of trials, seed, precision, prior parameters, and output directory.
7. Keep generated data and plots outside the source-code directory and document reproducibility in the README.
