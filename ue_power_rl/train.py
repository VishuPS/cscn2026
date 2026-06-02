"""
Training script for the Dueling DQN + PER agent.

Hyperparameters match Table I in the paper exactly.
Trains for 500k steps, saves checkpoints every 50k steps,
logs to results/training_log.csv.

Usage:
    python -m ue_power_rl.train --velocity 10 --seed 42
    python -m ue_power_rl.train --velocity 10 --seed 42 --steps 500000
"""

import os
import sys
import argparse
import csv
import time
import numpy as np
import torch

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

# add parent to path when running as script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ue_power_rl.env import UEPowerEnv


# --------------------------------------------------------------------------- #
# Hyperparameters (Table I in paper)
# --------------------------------------------------------------------------- #
HP = {
    "gamma":          0.99,
    "learning_rate":  3e-4,
    "batch_size":     256,
    "buffer_size":    100_000,
    "tau":            0.005,          # soft target update
    "train_freq":     1,              # update every step
    "gradient_steps": 1,
    "learning_starts": 2_000,         # warm-up with random actions
    "exploration_fraction": 0.1,      # fraction of steps for eps decay
    "exploration_initial_eps": 1.0,
    "exploration_final_eps":   0.05,
    "optimize_memory_usage":   False,
    "net_arch": [64, 64],
}


# --------------------------------------------------------------------------- #
# Metric logging callback
# --------------------------------------------------------------------------- #
class MetricsCallback(BaseCallback):
    """
    Logs per-episode metrics to CSV.
    Fields: step, episode, mean_reward, ep_ee_gbits_j, ep_energy_mj,
            ep_mean_lat_ms, epsilon
    """

    def __init__(self, log_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.log_path = log_path
        self._ep_count = 0
        self._csvfile  = None
        self._writer   = None

    def _on_training_start(self):
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self._csvfile = open(self.log_path, "w", newline="")
        self._writer  = csv.DictWriter(self._csvfile, fieldnames=[
            "step", "episode", "mean_reward",
            "ep_ee_gbits_j", "ep_energy_mj", "ep_mean_lat_ms",
            "epsilon"
        ])
        self._writer.writeheader()

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode_ee_gbits_j" in info:
                self._ep_count += 1
                eps = self.model.exploration_rate
                self._writer.writerow({
                    "step":            self.num_timesteps,
                    "episode":         self._ep_count,
                    "mean_reward":     info.get("episode", {}).get("r", 0),
                    "ep_ee_gbits_j":   info["episode_ee_gbits_j"],
                    "ep_energy_mj":    info["episode_energy_mj"],
                    "ep_mean_lat_ms":  info["episode_mean_lat_ms"],
                    "epsilon":         eps,
                })
                self._csvfile.flush()
        return True

    def _on_training_end(self):
        if self._csvfile:
            self._csvfile.close()


# --------------------------------------------------------------------------- #
# Main training function
# --------------------------------------------------------------------------- #
def train(velocity_kmh: float = 10.0,
          d2d_m: float = 200.0,
          total_steps: int = 500_000,
          seed: int = 42,
          results_dir: str = "results"):

    tag = f"v{int(velocity_kmh)}kmh_d{int(d2d_m)}m_s{seed}"
    model_dir = os.path.join(results_dir, "models", tag)
    log_path  = os.path.join(results_dir, "logs", f"train_{tag}.csv")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Training: v={velocity_kmh} km/h, d={d2d_m} m, seed={seed}")
    print(f"  Steps: {total_steps:,}  |  device: "
          f"{'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"{'='*60}\n")

    # environment
    env = Monitor(UEPowerEnv(
        velocity_kmh=velocity_kmh,
        d2d_m=d2d_m,
        ep_len_slots=5_000,
        seed=seed
    ))

    # model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DQN(
        "MlpPolicy",
        env,
        verbose=0,
        seed=seed,
        device=device,
        policy_kwargs={"net_arch": HP["net_arch"]},
        **{k: v for k, v in HP.items() if k != "net_arch"}
    )

    # callbacks
    callbacks = [
        MetricsCallback(log_path=log_path),
        CheckpointCallback(
            save_freq=50_000,
            save_path=model_dir,
            name_prefix="dqn_checkpoint"
        ),
    ]

    # train
    t0 = time.time()
    model.learn(total_timesteps=total_steps, callback=callbacks,
                progress_bar=True)
    elapsed = time.time() - t0

    # save final model
    final_path = os.path.join(model_dir, "dqn_final")
    model.save(final_path)

    print(f"\nTraining complete in {elapsed/60:.1f} min")
    print(f"Model saved to: {final_path}.zip")
    print(f"Log saved to:   {log_path}")

    return model, env, tag


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--velocity", type=float, default=10.0)
    parser.add_argument("--distance", type=float, default=200.0)
    parser.add_argument("--steps",    type=int,   default=500_000)
    parser.add_argument("--seed",     type=int,   default=42)
    parser.add_argument("--results",  type=str,   default="results")
    args = parser.parse_args()

    train(velocity_kmh=args.velocity,
          d2d_m=args.distance,
          total_steps=args.steps,
          seed=args.seed,
          results_dir=args.results)
