"""A/B testing homework: Epsilon-Greedy vs Thompson Sampling.

Scenario
--------
Four advertisement options are modeled as Gaussian bandits with true means
[1, 2, 3, 4]. We compare:

1. Epsilon-Greedy with decaying epsilon = 1 / t.
2. Thompson Sampling for Gaussian rewards with known precision.

The script runs both algorithms for 20,000 trials, saves the required reward
CSV, creates learning/comparison plots, and logs cumulative/average reward and
regret.
"""

from abc import ABC, abstractmethod
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger


# ---------------------------------------------------------------------------
# Global experiment settings from the homework
# ---------------------------------------------------------------------------
BANDIT_REWARD = [1, 2, 3, 4]
NUMBER_OF_TRIALS = 20_000
SEED = 223
OUTPUT_DIR = Path("output")
IMG_DIR = Path("img")


class Bandit(ABC):
    ##==== DO NOT REMOVE ANYTHING FROM THIS CLASS ====##

    @abstractmethod
    def __init__(self, p):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def pull(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def experiment(self):
        pass

    @abstractmethod
    def report(self):
        # store data in csv
        # print average reward (use f strings to make it informative)
        # print average regret (use f strings to make it informative)
        pass


# ---------------------------------------------------------------------------
# Visualization helper
# ---------------------------------------------------------------------------
class Visualization:
    """Create the plots required by the homework."""

    def __init__(self, epsilon_result=None, thompson_result=None):
        self.epsilon_result = epsilon_result
        self.thompson_result = thompson_result
        IMG_DIR.mkdir(parents=True, exist_ok=True)

    def plot1(self, result=None):
        """Visualize the learning process of each bandit on linear/log x scales.

        Parameters
        ----------
        result : dict, optional
            Result dictionary returned by ``experiment()``. If omitted, both
            available algorithm results are plotted.
        """
        results = []
        if result is not None:
            results = [result]
        else:
            if self.epsilon_result is not None:
                results.append(self.epsilon_result)
            if self.thompson_result is not None:
                results.append(self.thompson_result)

        for res in results:
            history = res["estimate_history"]
            trials = np.arange(1, len(history) + 1)
            algorithm_name = res["algorithm"]
            file_stem = algorithm_name.lower().replace(" ", "_").replace("-", "_")

            # Linear x-axis
            plt.figure(figsize=(9, 5))
            for i, true_mean in enumerate(BANDIT_REWARD):
                plt.plot(trials, history[:, i], label=f"Bandit {i + 1} (true mean={true_mean})")
            plt.xlabel("Trial")
            plt.ylabel("Estimated mean")
            plt.title(f"{algorithm_name}: Learning Process (Linear Scale)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(IMG_DIR / f"{file_stem}_learning_linear.png", dpi=200)
            plt.close()

            # Log x-axis
            plt.figure(figsize=(9, 5))
            for i, true_mean in enumerate(BANDIT_REWARD):
                plt.plot(trials, history[:, i], label=f"Bandit {i + 1} (true mean={true_mean})")
            plt.xscale("log")
            plt.xlabel("Trial (log scale)")
            plt.ylabel("Estimated mean")
            plt.title(f"{algorithm_name}: Learning Process (Log Scale)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(IMG_DIR / f"{file_stem}_learning_log.png", dpi=200)
            plt.close()

    def plot2(self):
        """Compare cumulative rewards and cumulative regrets of both algorithms."""
        if self.epsilon_result is None or self.thompson_result is None:
            raise ValueError("Both Epsilon-Greedy and Thompson Sampling results are required.")

        trials = np.arange(1, NUMBER_OF_TRIALS + 1)

        # Cumulative rewards comparison
        plt.figure(figsize=(9, 5))
        plt.plot(trials, self.epsilon_result["cumulative_rewards"], label="Epsilon-Greedy")
        plt.plot(trials, self.thompson_result["cumulative_rewards"], label="Thompson Sampling")
        plt.xlabel("Trial")
        plt.ylabel("Cumulative reward")
        plt.title("Cumulative Rewards: Epsilon-Greedy vs Thompson Sampling")
        plt.legend()
        plt.tight_layout()
        plt.savefig(IMG_DIR / "cumulative_rewards_comparison.png", dpi=200)
        plt.close()

        # Cumulative regret comparison
        plt.figure(figsize=(9, 5))
        plt.plot(trials, self.epsilon_result["cumulative_regrets"], label="Epsilon-Greedy")
        plt.plot(trials, self.thompson_result["cumulative_regrets"], label="Thompson Sampling")
        plt.xlabel("Trial")
        plt.ylabel("Cumulative pseudo-regret")
        plt.title("Cumulative Regrets: Epsilon-Greedy vs Thompson Sampling")
        plt.legend()
        plt.tight_layout()
        plt.savefig(IMG_DIR / "cumulative_regrets_comparison.png", dpi=200)
        plt.close()


# ---------------------------------------------------------------------------
# Epsilon-Greedy
# ---------------------------------------------------------------------------
class EpsilonGreedy(Bandit):
    """Decaying epsilon-greedy experiment with Gaussian rewards.

    Epsilon is set to 1/t at trial t, as required by the homework. Reward noise
    has unit variance, so a pull from a bandit with mean p is N(p, 1).
    """

    def __init__(self, p, n_trials=NUMBER_OF_TRIALS, seed=SEED):
        self.p = np.asarray(p, dtype=float)
        self.n_trials = int(n_trials)
        self.rng = np.random.default_rng(seed)

        self.counts = np.zeros(len(self.p), dtype=int)
        self.estimates = np.zeros(len(self.p), dtype=float)
        self.selected_bandits = []
        self.rewards = []
        self.regrets = []
        self.estimate_history = []
        self.current_bandit = None

    def __repr__(self):
        return f"EpsilonGreedy(bandits={self.p.tolist()}, trials={self.n_trials})"

    def pull(self):
        """Pull the currently selected arm and return a Gaussian reward."""
        if self.current_bandit is None:
            raise RuntimeError("A bandit must be selected before pull().")
        mean = self.p[self.current_bandit]
        return float(self.rng.normal(loc=mean, scale=1.0))

    def update(self, bandit_index=None, reward=None):
        """Update a selected arm's sample-mean reward estimate."""
        if bandit_index is None:
            bandit_index = self.current_bandit
        if reward is None:
            raise ValueError("reward is required for update().")

        self.counts[bandit_index] += 1
        n = self.counts[bandit_index]
        old = self.estimates[bandit_index]
        self.estimates[bandit_index] = old + (reward - old) / n

    def experiment(self):
        """Run the full decaying-epsilon experiment."""
        optimal_mean = float(np.max(self.p))

        for t in range(1, self.n_trials + 1):
            epsilon = 1.0 / t

            if self.rng.random() < epsilon:
                self.current_bandit = int(self.rng.integers(len(self.p)))
            else:
                # Random tie-breaking avoids always favoring the first arm at t=1.
                max_estimate = np.max(self.estimates)
                candidates = np.flatnonzero(np.isclose(self.estimates, max_estimate))
                self.current_bandit = int(self.rng.choice(candidates))

            reward = self.pull()
            self.update(self.current_bandit, reward)

            regret = optimal_mean - self.p[self.current_bandit]
            self.selected_bandits.append(self.current_bandit)
            self.rewards.append(reward)
            self.regrets.append(float(regret))
            self.estimate_history.append(self.estimates.copy())

        rewards = np.asarray(self.rewards)
        regrets = np.asarray(self.regrets)

        return {
            "algorithm": "Epsilon-Greedy",
            "selected_bandits": np.asarray(self.selected_bandits),
            "rewards": rewards,
            "regrets": regrets,
            "cumulative_rewards": np.cumsum(rewards),
            "cumulative_regrets": np.cumsum(regrets),
            "estimate_history": np.asarray(self.estimate_history),
            "final_estimates": self.estimates.copy(),
            "counts": self.counts.copy(),
        }

    def report(self):
        """Save the required CSV and log reward/regret summary statistics."""
        if not self.rewards:
            raise RuntimeError("Run experiment() before report().")

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(
            {
                "Bandit": np.asarray(self.selected_bandits) + 1,
                "Reward": self.rewards,
                "Algorithm": "Epsilon-Greedy",
            }
        )
        df.to_csv(OUTPUT_DIR / "epsilon_greedy_rewards.csv", index=False)

        cumulative_reward = float(np.sum(self.rewards))
        cumulative_regret = float(np.sum(self.regrets))
        average_reward = float(np.mean(self.rewards))
        average_regret = float(np.mean(self.regrets))

        logger.info(f"Epsilon-Greedy cumulative reward: {cumulative_reward:.3f}")
        logger.info(f"Epsilon-Greedy average reward: {average_reward:.3f}")
        logger.info(f"Epsilon-Greedy cumulative regret: {cumulative_regret:.3f}")
        logger.info(f"Epsilon-Greedy average regret: {average_regret:.6f}")

        return df


# ---------------------------------------------------------------------------
# Thompson Sampling with Gaussian rewards and known precision
# ---------------------------------------------------------------------------
class ThompsonSampling(Bandit):
    """Gaussian Thompson Sampling with known observation precision.

    Course-aligned defaults:
    - Likelihood: X ~ N(mu, tau^-1)
    - Known precision tau = 1
    - Prior for each unknown mean: mu ~ N(m0, lambda0^-1)
    - m0 = 0, lambda0 = 1

    The posterior precision and mean are updated after every observed reward.
    """

    def __init__(
        self,
        p,
        n_trials=NUMBER_OF_TRIALS,
        precision=1.0,
        prior_mean=0.0,
        prior_precision=1.0,
        seed=SEED + 1,
    ):
        self.p = np.asarray(p, dtype=float)
        self.n_trials = int(n_trials)
        self.tau = float(precision)
        self.rng = np.random.default_rng(seed)

        self.m = np.full(len(self.p), float(prior_mean))
        self.lambda_ = np.full(len(self.p), float(prior_precision))
        self.sum_x = np.zeros(len(self.p), dtype=float)
        self.counts = np.zeros(len(self.p), dtype=int)

        self.prior_mean = float(prior_mean)
        self.prior_precision = float(prior_precision)

        self.selected_bandits = []
        self.rewards = []
        self.regrets = []
        self.estimate_history = []
        self.current_bandit = None

    def __repr__(self):
        return (
            "ThompsonSampling("
            f"bandits={self.p.tolist()}, trials={self.n_trials}, precision={self.tau})"
        )

    def pull(self):
        """Pull the selected arm: X ~ N(true_mean, tau^-1)."""
        if self.current_bandit is None:
            raise RuntimeError("A bandit must be selected before pull().")
        std = 1.0 / np.sqrt(self.tau)
        return float(self.rng.normal(loc=self.p[self.current_bandit], scale=std))

    def sample(self):
        """Draw one posterior sample for each arm mean."""
        posterior_std = 1.0 / np.sqrt(self.lambda_)
        return self.rng.normal(loc=self.m, scale=posterior_std)

    def update(self, bandit_index=None, reward=None):
        """Update posterior mean and precision for the selected arm."""
        if bandit_index is None:
            bandit_index = self.current_bandit
        if reward is None:
            raise ValueError("reward is required for update().")

        self.counts[bandit_index] += 1
        self.sum_x[bandit_index] += reward

        # Posterior precision: lambda_N = lambda_0 + N * tau
        self.lambda_[bandit_index] = (
            self.prior_precision + self.counts[bandit_index] * self.tau
        )

        # Posterior mean:
        # m_N = (lambda_0*m_0 + tau*sum(x)) / lambda_N
        self.m[bandit_index] = (
            self.prior_precision * self.prior_mean
            + self.tau * self.sum_x[bandit_index]
        ) / self.lambda_[bandit_index]

    def experiment(self):
        """Run Gaussian Thompson Sampling for the configured number of trials."""
        optimal_mean = float(np.max(self.p))

        for _ in range(self.n_trials):
            posterior_samples = self.sample()
            self.current_bandit = int(np.argmax(posterior_samples))

            reward = self.pull()
            self.update(self.current_bandit, reward)

            regret = optimal_mean - self.p[self.current_bandit]
            self.selected_bandits.append(self.current_bandit)
            self.rewards.append(reward)
            self.regrets.append(float(regret))
            self.estimate_history.append(self.m.copy())

        rewards = np.asarray(self.rewards)
        regrets = np.asarray(self.regrets)

        return {
            "algorithm": "Thompson Sampling",
            "selected_bandits": np.asarray(self.selected_bandits),
            "rewards": rewards,
            "regrets": regrets,
            "cumulative_rewards": np.cumsum(rewards),
            "cumulative_regrets": np.cumsum(regrets),
            "estimate_history": np.asarray(self.estimate_history),
            "final_estimates": self.m.copy(),
            "counts": self.counts.copy(),
        }

    def report(self):
        """Save the required CSV and log reward/regret summary statistics."""
        if not self.rewards:
            raise RuntimeError("Run experiment() before report().")

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(
            {
                "Bandit": np.asarray(self.selected_bandits) + 1,
                "Reward": self.rewards,
                "Algorithm": "Thompson Sampling",
            }
        )
        df.to_csv(OUTPUT_DIR / "thompson_sampling_rewards.csv", index=False)

        cumulative_reward = float(np.sum(self.rewards))
        cumulative_regret = float(np.sum(self.regrets))
        average_reward = float(np.mean(self.rewards))
        average_regret = float(np.mean(self.regrets))

        logger.info(f"Thompson Sampling cumulative reward: {cumulative_reward:.3f}")
        logger.info(f"Thompson Sampling average reward: {average_reward:.3f}")
        logger.info(f"Thompson Sampling cumulative regret: {cumulative_regret:.3f}")
        logger.info(f"Thompson Sampling average regret: {average_regret:.6f}")

        return df


def comparison():
    """Run both algorithms, compare them, and create all required outputs.

    Returns
    -------
    tuple
        The Epsilon-Greedy and Thompson Sampling result dictionaries.
    """
    logger.info("Starting Epsilon-Greedy experiment...")
    epsilon = EpsilonGreedy(BANDIT_REWARD)
    epsilon_result = epsilon.experiment()
    epsilon_df = epsilon.report()

    logger.info("Starting Thompson Sampling experiment...")
    thompson = ThompsonSampling(BANDIT_REWARD, precision=1.0)
    thompson_result = thompson.experiment()
    thompson_df = thompson.report()

    # Required combined CSV with columns {Bandit, Reward, Algorithm}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    combined = pd.concat([epsilon_df, thompson_df], ignore_index=True)
    combined.to_csv(OUTPUT_DIR / "bandit_rewards.csv", index=False)

    # Save a compact summary table as an additional helpful output.
    summary = pd.DataFrame(
        [
            {
                "Algorithm": "Epsilon-Greedy",
                "Cumulative Reward": epsilon_result["cumulative_rewards"][-1],
                "Average Reward": np.mean(epsilon_result["rewards"]),
                "Cumulative Regret": epsilon_result["cumulative_regrets"][-1],
                "Average Regret": np.mean(epsilon_result["regrets"]),
                "Best Bandit Pulls": epsilon_result["counts"][-1],
            },
            {
                "Algorithm": "Thompson Sampling",
                "Cumulative Reward": thompson_result["cumulative_rewards"][-1],
                "Average Reward": np.mean(thompson_result["rewards"]),
                "Cumulative Regret": thompson_result["cumulative_regrets"][-1],
                "Average Regret": np.mean(thompson_result["regrets"]),
                "Best Bandit Pulls": thompson_result["counts"][-1],
            },
        ]
    )
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)

    visualizer = Visualization(epsilon_result, thompson_result)
    visualizer.plot1()
    visualizer.plot2()

    logger.info("All CSV files and plots were created successfully.")
    logger.info(f"Epsilon final estimates: {epsilon_result['final_estimates']}")
    logger.info(f"Thompson final estimates: {thompson_result['final_estimates']}")
    logger.info(f"Epsilon arm counts: {epsilon_result['counts']}")
    logger.info(f"Thompson arm counts: {thompson_result['counts']}")

    return epsilon_result, thompson_result


if __name__ == "__main__":
    # The starter file asked to use log levels instead of print().
    logger.info("DS 223 Homework 2: Epsilon-Greedy vs Thompson Sampling")
    comparison()
